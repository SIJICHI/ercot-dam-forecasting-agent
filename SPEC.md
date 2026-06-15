# ERCOT電力市場 DAM価格予測エージェントアプリケーション — 仕様書

## 1. 概要

### アプリケーション名
**ERCOT DAM Price Forecasting Agent**（ERCOT翌日市場価格予測エージェント）

### 目的
DataRobotに登録・展開済みの時系列機械学習モデルを活用し、テキサス電力市場（ERCOT）における翌日市場（DAM）価格の24時間予測を、自然言語チャットで操作できるデモアプリケーション。経営層・意思決定者に対して「AIエージェント × MLモデル活用」の価値を直感的に示す。

### デモで訴求するDataRobotの差別化ポイント
| ポイント | 説明 |
|---|---|
| **マルチエージェント協調** | Coordinator → Forecaster → Reporter の3エージェントがLangGraphで連携 |
| **MCPツール統合** | 構築済みMCPサーバーのツールが自動統合され、DataRobot予測APIへシームレスにアクセス |
| **DataRobotモデル活用** | デプロイ済みMLモデルをエージェントのツール（`predict_realtime`）として呼び出し |

---

## 2. システムアーキテクチャ

```
ユーザー（チャットUI）
    │
    ▼
FastAPI Backend（fastapi_server/） ← デフォルトのまま変更なし
    │
    ▼
Agent（LangGraph マルチエージェント）  ← agent/agent/myagent.py を変更
    ├── Coordinator Agent  ← ユーザー入力を解釈し、パラメータを抽出・検証
    ├── Forecaster Agent   ← MCPツールを通じてDataRobot予測APIを呼び出し
    └── Reporter Agent     ← 予測結果を経営向け日本語レポートに整形
            │
            ▼ MCPプロトコル（HTTP）
        MCP Server（別途構築済み・自動統合）
            ├── get_dataset_details      ← データセット詳細取得
            ├── get_deployment_features  ← デプロイ特徴量取得
            └── predict_realtime         ← DataRobotデプロイへのリアルタイム予測
                    └── DataRobot Deployed ML Model
```

**変更ファイル：**
- `agent/agent/myagent.py` のみ変更
- フロントエンド・バックエンド・MCPサーバーはデフォルトのまま

---

## 3. デフォルトパラメータ

| パラメータ | 値 |
|---|---|
| Dataset ID | `6a234c26fbe8f6deb1434646` |
| Deployment ID | `6a2355338f421c2e8b54b99e` |
| 対象ハブ | `HB_HOUSTON` |

---

## 4. エージェント設計（agent/agent/myagent.py）

### 4.1 プロンプトテンプレート

ユーザー入力フォーマット（自然言語）：
```
「データセット（ID: 6a234c26fbe8f6deb1434646）とデプロイ（ID: 6a2355338f421c2e8b54b99e）を用い、
HB_HOUSTONの2025年10月20日 4時時点の前日市場価格（DAM）を24時間予測してください。
forecast_range_start=2025-10-20 05:00:00、forecast_range_end=2025-10-21 04:00:00（UTC）。
これで24時間の連続予測を取得してください」
```

### 4.2 エージェントフロー

```
START
  │
  ▼
[coordinator_node]
  役割: ユーザーの自然言語メッセージから以下パラメータを抽出
  - dataset_id（未指定時デフォルト: 6a234c26fbe8f6deb1434646）
  - deployment_id（未指定時デフォルト: 6a2355338f421c2e8b54b99e）
  - hub_name（例: HB_HOUSTON）
  - forecast_range_start（YYYY-MM-DD HH:MM:SS, UTC）
  - forecast_range_end（YYYY-MM-DD HH:MM:SS, UTC）
  ツール: なし（LLMによる解釈のみ）
  │
  ▼ relay（AIMessage → HumanMessage）
[forecaster_node]
  役割: MCPツールを呼び出し、予測結果を取得
  手順:
    1. get_dataset_details(dataset_id) でデータセット確認
    2. predict_realtime(dataset_id, deployment_id, forecast_range_start, forecast_range_end) で予測実行
  ツール: get_dataset_details, get_deployment_features, predict_realtime（MCP経由・自動統合）
  │
  ▼ relay
[reporter_node]
  役割: 予測結果を経営向け日本語レポート（Markdown）に整形
  出力内容:
    1. 予測サマリー（期間・対象ハブ・予測時間数）
    2. 価格統計（最高値・最安値・平均）
    3. 時間帯分析（朝・昼・夜・深夜）
    4. ビジネスインサイト（2〜3点）
    5. DataRobotモデル情報（デプロイID・データセットID）
  ツール: なし
  │
  ▼
END
```

### 4.3 タイムアウト設定
バッチ予測の応答時間を考慮し、`timeout=300`秒（5分）に設定。

---

## 5. 環境変数（.env）

| 変数名 | 説明 | 必須 |
|---|---|---|
| `DATAROBOT_API_TOKEN` | DataRobot APIトークン | ✅ |
| `DATAROBOT_ENDPOINT` | DataRobot エンドポイントURL | ✅ |
| `LLM_DEFAULT_MODEL` | LLMモデル名（例: `azure/gpt-4o-2024-11-20`） | ✅ |
| `USE_DATAROBOT_LLM_GATEWAY` | `1` でDR LLM Gateway使用 | ✅ |

---

## 6. 開発ワークフロー

### 役割分担
| 環境 | 作業内容 |
|---|---|
| **ローカルMac** | コードの編集・GitHubへのpush |
| **DataRobot Codespace** | セットアップ・シミュレーション・デプロイ |

### 初回セットアップ（Codespace）
DataRobotテンプレートギャラリーから新規Codespaceを作成し、ターミナルで：
```bash
dr start   # 環境変数・依存関係・Pulumiを自動セットアップ
```
> ⚠️ `dr start` はテンプレートギャラリーから起動した**新規Codespace上**で実行する。GitHubからcloneしたディレクトリ内で実行すると二重ディレクトリ構造になるため使用しない。

### カスタマイズ適用（Codespace）
`dr start` 完了後、カスタマイズ済みの `myagent.py` をGitHubから取得：
```bash
curl -o agent/agent/myagent.py \
  https://raw.githubusercontent.com/SIJICHI/ercot-dam-forecasting-agent/main/agent/agent/myagent.py
```

### シミュレーション起動（Codespace）
```bash
dr run dev
# → Exposed Ports タブでポート5173のURLを開いてチャット画面にアクセス
```

### コード改善サイクル
```
ローカルMacで myagent.py を編集
    ↓
git add / commit / push（ローカルターミナル）
    ↓
Codespaceで curl で再取得
    ↓
dr run dev で再起動
```

### 本番デプロイ（Codespace）
シミュレーションで問題なければ：
```bash
dr run deploy   # Pulumi経由でDataRobotへ一括デプロイ
```

---

## 7. デモシナリオ（プレゼン用）

1. **チャット画面を開く**（デフォルトUIのまま）
2. **以下のプロンプト例を貼り付けて送信**：
   ```
   データセット（ID: 6a234c26fbe8f6deb1434646）とデプロイ（ID: 6a2355338f421c2e8b54b99e）を
   用い、HB_HOUSTONの2025年10月20日 4時時点の前日市場価格（DAM）を24時間予測してください。
   forecast_range_start=2025-10-20 05:00:00、forecast_range_end=2025-10-21 04:00:00（UTC）。
   ```
3. **エージェントの思考過程を見せる**：
   - Coordinatorがパラメータを抽出するステップ
   - ForecasterがMCPツール（`predict_realtime`）を呼び出すステップ
   - Reporterが日本語レポートを生成するステップ
4. **最終出力を見せる**：ピーク価格・平均価格・インサイトが整形されたレポートとして表示
5. **「このモデルはDataRobotで管理・監視されている」** ことを説明 → MLOpsの価値へ
