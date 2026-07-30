# CLAUDE.md — Trossen AI (Mobile ALOHA) × π0.5 VLA pipeline

給 Claude Code 的專案指引。本目錄是 [TrossenRobotics/openpi](https://github.com/TrossenRobotics/openpi) fork 的 `examples/trossen_ai`，
在其上自寫「遙操作收資料 → 微調 π0.5 → 真機推論部署」整條 pipeline。任務：單右臂夾香蕉放藍色毛巾（微調後自主成功率 70%，未微調 base 0%）。

## 兩台機器與環境（最重要，先搞清楚在哪台）
- **4080 筆電**（Ubuntu 22.04、RTX 4080 12GB、62GB RAM）：收資料 + 推論部署。
  - 用 **`uv run <cmd>`**（本目錄 `uv` + `uv.lock` 管理）；python 3.12、torch 2.7.1+cu126、lerobot 0.4.1、transformers 4.53.3。
  - shell 預設會 `conda activate Trossen_Ai`（見全域記憶）；實際跑 python 一律透過 `uv run` 落在本目錄 `.venv`。
- **5090 伺服器**（Windows + WSL、實體 16GB RAM）：只做訓練。`conda activate lerobot` 後**指令直接執行，不要加 `uv run`**。

## 硬體速查
| 設備 | 識別 |
|---|---|
| 右 follower（執行動作） | `192.168.1.4` |
| 右 leader（遙操作手把） | `192.168.1.2` |
| 左 follower / 左 leader | `.5` / `.3`（本 pipeline 未用） |
| top 相機（塔架 D405） | `230422271207` |
| 右腕相機 | `315122271274` |
| 左腕相機 | `315122272759`（腳本**預設值是這顆**，用右臂務必覆寫 `--wrist_serial`） |

## 常用指令（本專案的 build/test/dev 等價物）
```bash
# 環境自檢（uv 首次會自動照 uv.lock 裝好）
uv run python -c "import torch, lerobot; print(torch.__version__, lerobot.__version__, torch.cuda.is_available())"

# 推論：離線測試（相機開、手臂不動）
uv run single_arm_test.py --mode test --repo_id Zong-Ying/pi05_banana_towel \
    --wrist_serial 315122271274 --task_prompt "Pick up the banana and place it on the blue towel." --num_steps 10
# 推論：真機自主（右臂會動，e-stop 放手邊）
uv run single_arm_test.py --mode autonomous --repo_id Zong-Ying/pi05_banana_towel \
    --arm_ip 192.168.1.4 --top_serial 230422271207 --wrist_serial 315122271274 \
    --task_prompt "Pick up the banana and place it on the blue towel." --num_steps 30 --actions_per_chunk 50

# 收資料（自動回 staged、錄製中不編碼；鍵盤 →結束本集 ←重錄 Esc結束，絕不 Ctrl-C）
uv run record_autostage.py --robot.type=widowxai_follower_robot --robot.ip_address=192.168.1.4 \
    --teleop.type=widowxai_leader_teleop --teleop.ip_address=192.168.1.2 --display_data=true \
    --robot.cameras='{"top":{"type":"intelrealsense","serial_number_or_name":"230422271207","width":640,"height":480,"fps":30},"cam_wrist":{"type":"intelrealsense","serial_number_or_name":"315122271274","width":640,"height":480,"fps":30}}' \
    --dataset.repo_id=<帳號>/<資料集> --dataset.single_task="<英文 prompt>" \
    --dataset.fps=30 --dataset.num_episodes=10 --dataset.episode_time_s=45 --dataset.reset_time_s=15 --dataset.push_to_hub=false
# 錄完若影片沒編碼（崩潰過）：uv run encode_videos.py --repo-id <帳號>/<資料集>

# 驗收 / 視覺化
uv run lerobot-dataset-viz --repo-id Zong-Ying/banana_towel_right_arm --episode-index 0

# 微調（在 5090 伺服器，conda lerobot，不用 uv run）
python train_pi05_expert_only.py --dataset.repo_id=<帳號>/<資料集> --policy.type=pi05 \
    --policy.pretrained_path=$HOME/models/pi05_base --output_dir=$HOME/outputs/<job> --job_name=<job> \
    --num_workers=4 --log_freq=20 --policy.compile_model=true --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 --policy.device=cuda --batch_size=8 --steps=30000 --save_freq=10000 \
    --policy.repo_id=<帳號>/<模型> --policy.push_to_hub=true --wandb.enable=true --wandb.project=<專案>

# base 未微調對照（先下載底模，釘 revision）
uv run hf download lerobot/pi05_base --revision a538eb273274eb30f126a118f39dbc0ee212c883 --local-dir ~/models/pi05_base
uv run single_arm_base_test.py --mode test --camera dummy --no_display --num_steps 3
```

## 資料 / 模型 / 文件在哪
- **資料集**：`Zong-Ying/banana_towel_right_arm`（HF Hub，**private**，50 集/48,067 幀/30Hz/7 維+top&cam_wrist 雙相機）。
- **模型**：微調 `Zong-Ying/pi05_banana_towel`；底模 `lerobot/pi05_base`（**釘 revision `a538eb27...`**，上游會漂移）。
- **原則**：**code 上 GitHub、資料集/權重上 HF Hub**。`outputs/`、`outputs_base/`、`datasets`(symlink)、`models/`、`*.safetensors` 已 `.gitignore`，勿進 git。
- **文件**（實驗室 Notion，需權限）：PDCA 主計畫 `39e1504d…`、教學文件 `3a41504d…8157`、base 對照（含從零複現）`3a41504d…818e`、微調 70% 實驗 `3a41504d…8015`、Git 上架教學 `3ac1504d…`。

## 架構與程式碼風格
- **錄製與編碼分離**：`record_autostage.py` 錄製中**完全不編碼影片**（`batch_encoding_size=1e9`、每集後 `episodes_since_last_encoding=0`）→ 斷線 → `finalize()` → 呼叫 `encode_videos.py` 補編碼。原因：lerobot 0.4.1 的批次編碼是壞的（session 中 ParquetWriter 未關檔，讀 parquet 必炸），且編碼與錄影同跑會餓死手臂 UDP。
- **推論腳本**：`argparse` CLI；`SingleArmPi05Tester`（single_arm_test.py）為基底，`single_arm_base_test.py` 用**子類覆寫**（`_load_policy`/相機槽對位/state 正規化/action 切 7 維），不改基底、最大化重用。
- **安全模式**：真機首步用 PCHIP 平滑 5 秒到第一個預測動作；`min_time_to_move_multiplier=4.0` 慢動作；`Ctrl-C` = 緊急鎖定原地（僅推論；錄製時反而禁用 Ctrl-C）。
- **π0.5 維度**：模型內部 state/action 都是 **32 維 padded**，單臂只用**前 7 維**（joint_0–5 + 夾爪）其餘掛 0；一次推論輸出 **50 步 action chunk**。
- **慣例**：程式碼註解用英文、真機語音提示用中文（`spd-say -l cmn`）；文件（README/Notion）用繁中。matplotlib 用 `Agg` backend；每次推論輸出存 `outputs*/<時間戳>_.../`（相機影像 + `actions_*.npy` + 軌跡圖 + `log.jsonl`）。

## 重要注意事項（踩坑紀錄）
- **錄製絕不按 Ctrl-C**（資料會存壞）；看到 `Svt[info]` 刷屏是編碼器訊息、不是當機。
- **手臂 UDP 崩潰 / 卡 handshake** → 控制器**斷電重啟**（`ping` 得到不代表 driver 活著）。
- **`lerobot/pi05_base` 版本漂移**（2026-06 後加了 0.4.1 不認得的 processor）→ 一律釘 revision `a538eb273274eb30f126a118f39dbc0ee212c883`。
- **全參數 π0.5 塞不進 32GB**（AdamW fp32 狀態 ~29GB）→ 用 `train_pi05_expert_only.py` 凍結 PaliGemma VLM（可訓 693M）。
- **base 模型出廠 float32(14GB)+device=mps** → 12GB GPU 要 **bfloat16 + 先 CPU 載入再 `.to(cuda)`**（`single_arm_base_test.py` 已處理），並覆寫 preprocessor 的 `device_processor` 為 cuda。
- **訓練 GPU「假忙」**（util 100% 但功耗 ~130W）＝ VRAM 溢出到系統 RAM，**看功耗不看利用率**（真在算 400W+）。
- **推論放置階段**：用 `--actions_per_chunk 50`（執行完整 chunk），否則「下降→鬆爪」會被截斷、手臂一直對位不放。
- **git**：幫使用者建 commit **不加 Claude 署名 / Co-Authored-By**；共用機 `git config` 一律 `--local`（勿 `--global`）；`.gitignore` **不支援行內註解**（`#` 要獨立一行）。
