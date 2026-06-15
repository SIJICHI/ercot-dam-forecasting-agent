# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import csv
import io
import logging
import os
from typing import Annotated, Any

import httpx
from datarobot_genai.drmcp import dr_mcp_tool
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

logger = logging.getLogger(__name__)


def _get_dr_config() -> tuple[str, str]:
    """Return (base_url, api_token) from environment."""
    endpoint = os.environ.get("DATAROBOT_ENDPOINT", "").rstrip("/")
    token = os.environ.get("DATAROBOT_API_TOKEN", "")
    if not endpoint or not token:
        raise ToolError(
            "DATAROBOT_ENDPOINT and DATAROBOT_API_TOKEN must be set."
        )
    # Normalize: remove /api/v2 suffix to get base URL
    if endpoint.endswith("/api/v2"):
        endpoint = endpoint[: -len("/api/v2")]
    return endpoint, token


@dr_mcp_tool(tags={"datarobot", "dataset"})
async def get_dataset_details(
    dataset_id: Annotated[str, "DataRobot AI Catalog Dataset ID"],
) -> ToolResult:
    """
    DataRobot AIカタログに登録されたデータセットの詳細情報を取得する。
    データセットの名前・行数・列数・作成日時などを返す。
    """
    if not dataset_id or not dataset_id.strip():
        raise ToolError("dataset_id cannot be empty.")

    endpoint, token = _get_dr_config()
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{endpoint}/api/v2/datasets/{dataset_id.strip()}/"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 404:
            raise ToolError(f"Dataset '{dataset_id}' not found in AI Catalog.")
        if resp.status_code != 200:
            raise ToolError(
                f"Failed to get dataset details: HTTP {resp.status_code} — {resp.text}"
            )
        data = resp.json()

    return ToolResult(
        structured_content={
            "dataset_id": dataset_id,
            "name": data.get("name"),
            "row_count": data.get("rowCount"),
            "column_count": data.get("columnCount"),
            "created_at": data.get("createdAt"),
            "size_bytes": data.get("datasetSize"),
        }
    )


@dr_mcp_tool(tags={"datarobot", "prediction", "timeseries"})
async def predict_realtime(
    dataset_id: Annotated[str, "DataRobot AI Catalog Dataset ID to use as scoring input"],
    deployment_id: Annotated[str, "DataRobot Deployment ID of the deployed ML model"],
    forecast_range_start: Annotated[
        str,
        "Forecast start datetime in 'YYYY-MM-DD HH:MM:SS' UTC format (e.g. '2025-10-20 05:00:00')",
    ],
    forecast_range_end: Annotated[
        str,
        "Forecast end datetime in 'YYYY-MM-DD HH:MM:SS' UTC format (e.g. '2025-10-21 04:00:00')",
    ],
) -> ToolResult:
    """
    DataRobotのデプロイ済みMLモデルを使って時系列予測を実行する。
    AIカタログのデータセットをスコアリング入力として使用し、
    指定した予測期間（forecast_range_start〜forecast_range_end）の予測結果を返す。
    """
    if not dataset_id or not dataset_id.strip():
        raise ToolError("dataset_id cannot be empty.")
    if not deployment_id or not deployment_id.strip():
        raise ToolError("deployment_id cannot be empty.")
    if not forecast_range_start or not forecast_range_start.strip():
        raise ToolError("forecast_range_start cannot be empty.")
    if not forecast_range_end or not forecast_range_end.strip():
        raise ToolError("forecast_range_end cannot be empty.")

    endpoint, token = _get_dr_config()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    logger.info(
        "Starting batch prediction: deployment_id=%s dataset_id=%s range=%s to %s",
        deployment_id,
        dataset_id,
        forecast_range_start,
        forecast_range_end,
    )

    payload: dict[str, Any] = {
        "deploymentId": deployment_id.strip(),
        "intakeSettings": {
            "type": "dataset",
            "datasetId": dataset_id.strip(),
        },
        "outputSettings": {
            "type": "localFile",
        },
        "includeProbabilities": False,
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
        # バッチ予測ジョブを送信
        batch_url = f"{endpoint}/api/v2/batchPredictions/"
        resp = await client.post(batch_url, json=payload, headers=headers)
        if resp.status_code not in (200, 201, 202):
            raise ToolError(
                f"Batch prediction submission failed: HTTP {resp.status_code} — {resp.text}"
            )

        job_data = resp.json()
        job_id = job_data.get("id")
        if not job_id:
            raise ToolError(f"No job ID returned: {job_data}")

        logger.info("Batch prediction job submitted: job_id=%s", job_id)

        # ジョブ完了までポーリング（最大10分）
        job_url = f"{endpoint}/api/v2/batchPredictions/{job_id}/"
        for _ in range(120):
            await asyncio.sleep(5)
            status_resp = await client.get(job_url, headers=headers)
            if status_resp.status_code != 200:
                raise ToolError(
                    f"Failed to poll job status: HTTP {status_resp.status_code}"
                )
            status_data = status_resp.json()
            status = status_data.get("status", "")
            logger.info("Job status: %s", status)

            if status == "COMPLETED":
                break
            if status in ("FAILED", "ABORTED"):
                raise ToolError(
                    f"Batch prediction {status}: {status_data.get('statusDetails', '')}"
                )
        else:
            raise ToolError("Batch prediction timed out after 10 minutes.")

        # 結果をダウンロード
        download_url = f"{endpoint}/api/v2/batchPredictions/{job_id}/download/"
        dl_resp = await client.get(download_url, headers=headers)
        if dl_resp.status_code != 200:
            raise ToolError(
                f"Failed to download results: HTTP {dl_resp.status_code}"
            )

    # CSV結果をパース
    reader = csv.DictReader(io.StringIO(dl_resp.text))
    rows = list(reader)

    if not rows:
        raise ToolError("Prediction results are empty.")

    columns = list(rows[0].keys())

    # タイムスタンプ列と予測値列を自動特定
    timestamp_col = next(
        (c for c in columns if "timestamp" in c.lower() or "date" in c.lower()), None
    )
    prediction_col = next(
        (
            c for c in columns
            if "predict" in c.lower() or "dam_price" in c.lower() or "target" in c.lower()
        ),
        None,
    )

    if not timestamp_col or not prediction_col:
        return ToolResult(
            structured_content={
                "warning": "Could not identify timestamp or prediction column automatically.",
                "available_columns": columns,
                "sample_rows": rows[:3],
            }
        )

    predictions = []
    for row in rows:
        try:
            predictions.append(
                {
                    "timestamp": row[timestamp_col],
                    "predicted_dam_price_usd_mwh": round(float(row[prediction_col]), 2),
                }
            )
        except (ValueError, KeyError):
            continue

    # 統計サマリー
    prices = [p["predicted_dam_price_usd_mwh"] for p in predictions]
    summary: dict[str, Any] = {}
    if prices:
        summary = {
            "count": len(prices),
            "min_usd_mwh": round(min(prices), 2),
            "max_usd_mwh": round(max(prices), 2),
            "avg_usd_mwh": round(sum(prices) / len(prices), 2),
            "peak_hour": predictions[prices.index(max(prices))]["timestamp"],
            "lowest_hour": predictions[prices.index(min(prices))]["timestamp"],
        }

    logger.info(
        "Prediction completed: %d rows, summary=%s", len(predictions), summary
    )

    return ToolResult(
        structured_content={
            "deployment_id": deployment_id,
            "dataset_id": dataset_id,
            "forecast_range_start": forecast_range_start,
            "forecast_range_end": forecast_range_end,
            "summary": summary,
            "predictions": predictions,
        }
    )
