# Trossen AI (Mobile ALOHA) × π0.5 實作 pipeline — 實驗室交接文件

給接手實驗室 Trossen AI / Mobile ALOHA 的學弟妹：這份帶你把整條 VLA pipeline 跑起來——
**推論公開模型 → 遙操作收自己的資料 → 微調 π0.5 → 部署回真機**。所有腳本都在本資料夾，
在實驗室真機驗證過（2026-07，任務：夾香蕉放藍色毛巾，微調後自主成功率 70%）。

> 📦 **重要觀念：程式碼在 GitHub、資料集與模型權重在 HuggingFace Hub。**
> 這個 repo **不含**任何資料集或權重（那些有數十 GB，且 Hub 才是它們的家）。需要時用下方指令從 Hub 拉。
>
> 🔗 **本專案基礎**：建在 [TrossenRobotics/openpi](https://github.com/TrossenRobotics/openpi) 的 `examples/trossen_ai` 上，
> 用 [TrossenRobotics/lerobot_trossen](https://github.com/TrossenRobotics/lerobot_trossen)（Trossen AI / Mobile ALOHA 的 LeRobot plugin）驅動手臂，
> 模型來自 [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base)（openpi π0.5 的 LeRobot 移植）。完整來源見文末〈來源 / 上游專案〉。

---

## 0. 硬體與環境

### 硬體速查
| 設備 | 識別 |
|---|---|
| 右 follower 臂（執行動作） | IP `192.168.1.4` |
| 右 leader 臂（遙操作手把） | IP `192.168.1.2` |
| 左 follower / 左 leader | `192.168.1.5` / `192.168.1.3`（本 pipeline 未用） |
| top 相機（塔架 D405） | serial `230422271207` |
| 右腕相機（D405） | serial `315122271274` |
| 左腕相機（D405） | serial `315122272759`（腳本的預設值，用右臂時要覆寫掉） |

### 兩台機器
| 機器 | 用途 | 環境 | 指令前綴 |
|---|---|---|---|
| 4080 筆電（Ubuntu 22.04, 12GB GPU） | 收資料＋推論部署 | `uv`（見下） | `uv run ...` |
| 5090 伺服器（Windows+WSL） | 微調訓練 | `conda activate lerobot` | 直接執行（不用 uv run） |

### 環境安裝（筆電）
本資料夾用 [uv](https://docs.astral.sh/uv/) 管理，`uv.lock` 已鎖定所有版本，一鍵重建：
```bash
# 裝 uv（若還沒有）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 第一次跑任何 uv run 指令，uv 會自動照 uv.lock 裝好依賴
cd examples/trossen_ai
uv run python -c "import torch, lerobot; print('OK', torch.__version__, lerobot.__version__, torch.cuda.is_available())"
# 預期：OK 2.7.1+cu126 0.4.1 True
```
π0.5 內含 gated 的 PaliGemma，需先在 huggingface.co 接受 [google/paligemma](https://huggingface.co/google/paligemma-3b-pt-224) 授權，再 `uv run hf auth login`。

---

## 1. 資料集與模型（在 HuggingFace Hub）

| 內容 | Hub repo | 說明 |
|---|---|---|
| 遙操作資料集 | `Zong-Ying/banana_towel_right_arm` | 50 集、48,067 幀、30Hz、單右臂 7 維 + top/cam_wrist 雙相機（private） |
| 微調後模型 | `Zong-Ying/pi05_banana_towel` | π0.5 expert-only 微調，30k steps |
| 底模（供微調/對照） | `lerobot/pi05_base` | 官方未微調 π0.5（釘 revision `a538eb27...`） |

### 下載資料集
> ⚠️ 本資料集是 **private**：先 `uv run hf auth login`，且該 HF 帳號要有存取權（跟 Zong-Ying 要 collaborator，或把 repo 改 public，或換成你自己的資料集）。

**方法 A — 直接用**（LeRobot 首次載入會自動下載到 `~/.cache/huggingface/lerobot/`）：
```bash
uv run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('Zong-Ying/banana_towel_right_arm')"
```
**方法 B — 用 `hf` 指令拉到指定資料夾**（注意 `--repo-type dataset`）：
```bash
uv run hf download Zong-Ying/banana_towel_right_arm --repo-type dataset \
    --local-dir ~/datasets/banana_towel_right_arm
```
**逐集視覺化檢查**（影像＋關節曲線）：
```bash
uv run lerobot-dataset-viz --repo-id Zong-Ying/banana_towel_right_arm --episode-index 0
```

模型不用手動抓：推論時 `single_arm_test.py --repo_id Zong-Ying/pi05_banana_towel` 會自動下載；
只有 base 對照組要先 `hf download lerobot/pi05_base --revision ...`（見〈base 未微調〉段）。

---

## 2. 三大流程

### 流程 A — 推論（部署微調好的模型到真機）
`single_arm_test.py`：載入 LeRobot π0/π0.5 checkpoint，連相機/手臂做推論，逐步存影像+軌跡+log。

```bash
# 離線測試（相機開、手臂不動；先做這個確認相機看得到物體、輸出無 NaN）
uv run single_arm_test.py --mode test \
    --repo_id Zong-Ying/pi05_banana_towel \
    --wrist_serial 315122271274 \
    --task_prompt "Pick up the banana and place it on the blue towel." --num_steps 10

# 真機自主推論（右臂會動；e-stop 放手邊）
uv run single_arm_test.py --mode autonomous \
    --repo_id Zong-Ying/pi05_banana_towel \
    --arm_ip 192.168.1.4 --top_serial 230422271207 --wrist_serial 315122271274 \
    --task_prompt "Pick up the banana and place it on the blue towel." \
    --num_steps 30 --actions_per_chunk 50
```
- ⚠️ 預設 `--arm_ip` 是左臂、`--wrist_serial` 是左腕；用右臂**務必**明確指定右臂 IP/序號。
- 第一次推論含 `torch.compile` 編譯約 7 分鐘屬正常，之後每次 ~0.2s。
- `--actions_per_chunk 50`：執行完整 50 步再重推論（避免「下降→鬆爪」被截斷）。

### 流程 B — 遙操作收資料
`record_autostage.py`（CLI 同官方 `lerobot-record`，但每集自動雙臂回 staged、語音中文、
**錄製中不編碼影片**避免餓死手臂 UDP）＋ `encode_videos.py`（結束後補編碼，已自動串接）。

```bash
uv run record_autostage.py \
    --robot.type=widowxai_follower_robot --robot.ip_address=192.168.1.4 \
    --robot.cameras='{
        "top":       {"type": "intelrealsense", "serial_number_or_name": "230422271207", "width": 640, "height": 480, "fps": 30},
        "cam_wrist": {"type": "intelrealsense", "serial_number_or_name": "315122271274", "width": 640, "height": 480, "fps": 30}
    }' \
    --teleop.type=widowxai_leader_teleop --teleop.ip_address=192.168.1.2 \
    --display_data=true \
    --dataset.repo_id=<你的帳號>/<資料集名> \
    --dataset.single_task="<你的任務英文 prompt>" \
    --dataset.fps=30 --dataset.num_episodes=10 \
    --dataset.episode_time_s=45 --dataset.reset_time_s=15 --dataset.push_to_hub=false
```
- 鍵盤：`→` 做完就按（結束本集）｜`←` 失誤重錄上一集｜`Esc` 結束。**絕對別按 Ctrl-C**（資料會壞）。
- 分段錄（每段 10 集，第 2 段起加 `--resume=true`）；示範要一氣呵成，示範品質＝模型品質上限。
- 錄完驗收＋推 Hub：見流程 A 的 viz 指令，確認過再 `push_to_hub`。

### 流程 C — 微調 π0.5（5090 伺服器）
`train_pi05_expert_only.py`：凍結 PaliGemma VLM、只訓 action expert（693M），
因為全參數 4B 的 AdamW 狀態 ~29GB 塞不進 32GB VRAM。在伺服器 `conda activate lerobot` 後執行：
```bash
python train_pi05_expert_only.py \
    --dataset.repo_id=<你的帳號>/<資料集名> \
    --policy.type=pi05 --policy.pretrained_path=$HOME/models/pi05_base \
    --output_dir=$HOME/outputs/<job名> --job_name=<job名> \
    --num_workers=4 --log_freq=20 \
    --policy.compile_model=true --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 --policy.device=cuda \
    --batch_size=8 --steps=30000 --save_freq=10000 \
    --policy.repo_id=<你的帳號>/<模型名> --policy.push_to_hub=true \
    --wandb.enable=true --wandb.project=<wandb專案>
```
訓練完 checkpoint 自動推上 Hub，回筆電用流程 A 直接指 `--repo_id` 部署，不用 scp。
（環境安裝有很多坑：torch cu128、transformers fork、pi05_base 釘 revision 等，見下方 Notion。）

### （對照）base 未微調
`single_arm_base_test.py`：跑未微調的 `lerobot/pi05_base` 當對照組，量化微調貢獻。
需先 `uv run hf download lerobot/pi05_base --revision a538eb273274eb30f126a118f39dbc0ee212c883 --local-dir ~/models/pi05_base`。
用法同流程 A 的 test/autonomous（詳見腳本 docstring）。實測：未微調 base 0/10，微調後 7/10。

### 載入「別人微調好的模型」做推論
先看那個模型是什麼**格式**，決定用哪個腳本——這是最容易卡住的地方：

| 模型格式 | 怎麼判斷 | 用哪個腳本 |
|---|---|---|
| **LeRobot PyTorch**（`config.json` + `model.safetensors` + processor JSON） | HF 頁面有 `model.safetensors`；能 `PI0Policy/PI05Policy.from_pretrained` | **`single_arm_test.py --repo_id <帳號>/<模型>`**（同流程 A，換 repo_id 即可） |
| **openpi 原生 JAX / orbax**（`params/` 目錄、`gs://openpi-assets/...`） | HF/GCS 有 `params/ocdbt.*`、無 safetensors | 不能用上面的腳本；要 **openpi JAX server + client**（見下） |

**A. LeRobot PyTorch 模型（單臂，最常見）**——直接換 `--repo_id`：
```bash
# 例：跑社群單臂 widowx 模型（先 test 確認維度/相機鍵對得上、無 NaN，再上真機）
uv run single_arm_test.py --mode test --repo_id <帳號>/<模型名> \
    --wrist_serial 315122271274 --task_prompt "<英文 prompt>" --num_steps 10
```
腳本會**從 checkpoint 的 config 自動讀**相機鍵 / state 維 / action 維，所以只要對方的相機鍵（如 `top`/`cam_wrist`）和關節佈局對得上就能跑；對不上會在 test 模式就報錯，不會傷到手臂。

**B. openpi 原生 JAX 模型（官方 / 雙臂 ALOHA）**——用 `serve_policy.py` + `main_aloha.py`（兩終端、兩 venv）：
```bash
# 終端 A（repo 根、根 .venv 的 JAX）：起 policy server
cd ~/Desktop/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config名> --policy.dir=<gs://... 或本地 checkpoint 路徑>
# 終端 B（examples/trossen_ai）：雙臂 client，先 test 自檢再上真機
uv run main_aloha.py --mode test --task_prompt "<英文 prompt>"
uv run main_aloha.py --mode autonomous --model_tag <標記> \
    --task_prompt "<英文 prompt>" --max_steps 1200 [--right_arm_only]
```
- `--policy.config` 決定 **embodiment 轉換**：經典 ALOHA 模型用 `adapt_to_pi=True` 的 config（夾爪走 Interbotix 換算）；**Trossen 手臂**要用 `adapt_to_pi=False` 的 config（如 `pi05_trossen_transfer_block`），否則夾爪會差 ~20 倍被鎖死。
- `main_aloha.py` 內建：關節極限 clamp（Trossen 驅動對超限零容忍，連 +2µm 都會 fault→崩潰）、即時 3 相機視窗（`--no_display` 關）、`--right_arm_only`（凍結左臂只動右臂，較安全）、Ctrl-C 溫和停止、輕量錄影到 `outputs_aloha/`。
- ⚠️ **跨機器人/跨場景通常無法零樣本遷移**：實測官方 `pi0_aloha_towel`（經典 ALOHA）夾爪 20× 不匹配；官方 Trossen 桌面模型雖能抓取，但因任務/相機/場景不同仍無法完成自訂任務。要在本實驗室 Mobile ALOHA 上做事，還是得走流程 B+C 用自己的資料微調（完整分析見下方 Notion）。

---

## 3. 檔案總覽
| 檔案 | 說明 |
|---|---|
| `single_arm_test.py` | 推論（微調 / 公開模型），test + autonomous |
| `single_arm_base_test.py` | 推論未微調 base 對照組 |
| `record_autostage.py` | 遙操作收資料（自動回 staged、錄製中不編碼） |
| `encode_videos.py` | 收完補編碼影片（record 結尾自動呼叫；也可獨立跑） |
| `train_pi05_expert_only.py` | 5090 上 expert-only 微調（凍 VLM） |
| `camera_preview.py` / `pose_left_arm.py` | 相機預覽 / 手臂姿勢小工具 |
| `main.py` | openpi websocket client（原始檔改過） |
| `main_aloha.py` | openpi JAX 雙臂 client（載入官方/JAX 模型：關節 clamp＋即時 3 相機＋`--right_arm_only`） |

---

## 4. 延伸閱讀（實驗室 Notion，需工作區權限）
- **PDCA 主計畫**（收資料→微調→部署全流程＋踩坑排除總表）：<https://app.notion.com/p/39e1504dd82281e0ac06efa26c3e14e6>
- **教學文件**（各階段指令＋debug 心法）：<https://app.notion.com/p/3a41504dd8228157ae0cd7d7cfbdf8fd>
- **base 未微調對照實驗**（含「從零複現」完整指南）：<https://app.notion.com/p/3a41504dd822818ea7dde7d19a406694>
- **微調 10 次實測（70%）**：<https://app.notion.com/p/3a41504dd82280158d0bedfa1217cfa7>
- **官方 openpi 模型測試**（載入他人/官方模型的三層遷移結論）：<https://app.notion.com/p/3ad1504dd822814e8d90f14a021f9e03>

## 5. 幾個最容易踩的坑（完整見 Notion 踩坑表）
- 手臂 UDP 崩潰、卡 handshake → 控制器**斷電重啟**（ping 得到不代表活著）。
- 錄製時看到 `Svt[info]` 刷屏不是當機，是編碼器訊息。**別按 Ctrl-C**。
- 上游 `lerobot/pi05_base` 版本會漂移 → 一律**釘 revision `a538eb27...`**。
- base 出廠 float32(14GB)+mps → 12GB GPU 要用 bfloat16 + 先 CPU 載入再 `.to(cuda)`（`single_arm_base_test.py` 已處理）。
- 訓練 GPU「假忙」（util 100% 但功耗只有 ~130W）＝ VRAM 溢出，**看功耗不看利用率**。

---

## 來源 / 上游專案
本專案是在以下開源專案上新增遙操作 / 推論 / 微調腳本，感謝上游作者：

| 專案 | 角色 |
|---|---|
| [TrossenRobotics/openpi](https://github.com/TrossenRobotics/openpi) | 本 repo 的**直接基礎**（fork 自 openpi，含 `examples/trossen_ai` 與 `uv.lock` 環境） |
| [TrossenRobotics/lerobot_trossen](https://github.com/TrossenRobotics/lerobot_trossen) | **Trossen AI / Mobile ALOHA 的 LeRobot plugin**，提供 `widowxai_follower_robot`、`widowxai_leader_teleop`、`bi_widowxai_*`、`mobileai_*` 等機器人 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0 / π0.5 VLA 原始實作（openpi 上游） |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | LeRobot（PyTorch 資料集 / 訓練 / 推論框架，本專案用 0.4.1） |
| [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) | π0.5 底模（openpi checkpoint 的 LeRobot PyTorch 移植） |
| [Trossen 官方文件](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/openpi.html) | 硬體 / openpi 教學 / lerobot plugin 收資料指南 |
