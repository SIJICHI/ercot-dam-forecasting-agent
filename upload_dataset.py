"""
DataRobot AIカタログにタイムスタンプ修正済みCSVをアップロードするスクリプト。
Codespaceのターミナルで実行: python upload_dataset.py
"""
import os
import requests

endpoint = os.environ.get("DATAROBOT_ENDPOINT", "https://app.jp.datarobot.com/api/v2").rstrip("/")
token = os.environ.get("DATAROBOT_API_TOKEN", "")

if not token:
    print("ERROR: DATAROBOT_API_TOKEN is not set.")
    exit(1)

headers = {"Authorization": f"Bearer {token}"}
csv_file = "ercot_test_ts_fixed.csv"

print(f"Uploading {csv_file} to DataRobot AI Catalog...")

with open(csv_file, "rb") as f:
    resp = requests.post(
        f"{endpoint}/datasets/fromFile/",
        headers=headers,
        files={"file": (csv_file, f, "text/csv")},

    )

if resp.status_code in (200, 201, 202):
    data = resp.json()
    dataset_id = data.get("datasetId") or data.get("id")
    print(f"\n✅ Upload successful!")
    print(f"   New Dataset ID: {dataset_id}")
    print(f"\nこのDataset IDをプロンプトで使用してください。")
else:
    print(f"❌ Upload failed: HTTP {resp.status_code}")
    print(resp.text)
