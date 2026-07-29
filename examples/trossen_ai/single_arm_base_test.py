#!/usr/bin/env python3
"""
Single-Arm WidowX AI <-> pi0.5 BASE (un-fine-tuned) inference test.

Control experiment for the fine-tuned pi0.5 (Zong-Ying/pi05_banana_towel, ~70%).
Runs the raw pre-trained pi0.5 base (openpi's checkpoint, LeRobot PyTorch port
`lerobot/pi05_base`, pinned revision) on the SAME robot / task / 10 configs, so
the ONLY variable vs the fine-tuned run is "were the weights fine-tuned or not".

Why this needs its own script (base cannot be loaded by single_arm_test.py):
    - base's camera slots are the generic 3-view set (base_0_rgb, left_wrist_0_rgb,
      right_wrist_0_rgb), not this robot's `top`/`cam_wrist`
    - base's state/action are 32-dim padded, not 7-dim WidowX joints
    - base ships NO normalization (normalization_mapping=null, normalizer features={})

What this script does (weights are NEVER touched; only I/O is adapted):
    - cameras (option B, 2-slot + masked left): top -> base_0_rgb,
      cam_wrist -> right_wrist_0_rgb; left_wrist_0_rgb is simply not provided, so
      pi0.5 auto-pads it with -1 and masks it (single-arm inputs it saw in
      pretraining). This matches the fine-tuned model's 2-camera input.
    - state: 7 WidowX joints -> QUANTILES-normalized to [-1,1] using the WidowX
      dataset stats -> pi0.5 pads to 32 internally. (Unit calibration, NOT learning.)
    - action: model outputs 32-dim normalized -> take first 7 -> QUANTILES-un-
      normalize with the dataset action stats -> WidowX joint targets.

Normalization is a units table (rad ranges from Zong-Ying/banana_towel_right_arm),
required to run ANY VLA on a specific robot; it involves no gradient updates. The
3.6B transformer weights are the frozen published base checkpoint.

Usage (identical robot flags to single_arm_test.py):
    Offline sanity check (cameras only, arm never moves):
    uv run single_arm_base_test.py --mode test \
        --wrist_serial 315122271274 \
        --task_prompt "Pick up the banana and place it on the blue towel." \
        --num_steps 5

    Real robot (right arm; keep the e-stop in reach):
    uv run single_arm_base_test.py --mode autonomous \
        --arm_ip 192.168.1.4 --top_serial 230422271207 --wrist_serial 315122271274 \
        --task_prompt "Pick up the banana and place it on the blue towel." \
        --num_steps 30 --actions_per_chunk 50
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from single_arm_test import KNOWN_CAMERA_SERIALS, STAGED_POSITIONS, WIDOWXAI_JOINT_NAMES, SingleArmPi05Tester

logger = logging.getLogger(__name__)

# Physical camera role -> pi0.5 base's native image slot name (option B).
# left_wrist_0_rgb is intentionally omitted -> the model masks that slot.
SLOT_FOR_ROLE = {"top": "base_0_rgb", "cam_wrist": "right_wrist_0_rgb"}
DEFAULT_BASE_DIR = str(Path.home() / "models" / "pi05_base")
DEFAULT_STATS = str(
    Path.home() / ".cache/huggingface/lerobot/Zong-Ying/banana_towel_right_arm/meta/stats.json"
)


class SingleArmPi05BaseTester(SingleArmPi05Tester):
    """pi0.5 BASE tester: same robot I/O as the fine-tuned tester, but adapts the
    generic base checkpoint's 32-dim / 3-camera / un-normalized interface."""

    def __init__(
        self,
        base_path: str = DEFAULT_BASE_DIR,
        stats_path: str = DEFAULT_STATS,
        mode: str = "test",
        camera_source: str = "auto",
        top_serial: str | None = None,
        wrist_serial: str | None = None,
        arm_ip: str = "192.168.1.4",
        output_root: str = "outputs_base",
        display: bool = True,
        control_frequency: int = 30,
        actions_per_chunk_to_execute: int = 50,
    ):
        self.repo_id = base_path
        self.mode = mode
        self.display = display
        self.control_frequency = control_frequency
        self.actions_per_chunk_to_execute = actions_per_chunk_to_execute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_quantile_stats(stats_path)
        self._load_policy()

        # Option B: only the 2 physical cameras; left_wrist_0_rgb slot stays masked.
        self.image_keys = ["base_0_rgb", "right_wrist_0_rgb"]
        self.state_dim = len(WIDOWXAI_JOINT_NAMES)   # 7, WidowX joints we feed/command
        self.action_dim = len(WIDOWXAI_JOINT_NAMES)  # 7, first slots of the 32-dim output
        logger.info(
            f"BASE mode: feeding cameras {self.image_keys} (left_wrist masked), "
            f"7-dim WidowX state<->action via dataset QUANTILES calibration"
        )

        self.robot = None
        self.cameras = {}
        self.webcam = None
        if self.mode == "autonomous":
            self._setup_robot(arm_ip, top_serial, wrist_serial)
        else:
            self._setup_cameras(camera_source, top_serial, wrist_serial)

        self.output_dir = Path(output_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = None

    # ------------------------------------------------------------------ normalization
    def _load_quantile_stats(self, stats_path: str):
        import json

        with open(stats_path) as f:
            stats = json.load(f)
        self._state_q01 = np.asarray(stats["observation.state"]["q01"], dtype=np.float32)
        self._state_q99 = np.asarray(stats["observation.state"]["q99"], dtype=np.float32)
        self._action_q01 = np.asarray(stats["action"]["q01"], dtype=np.float32)
        self._action_q99 = np.asarray(stats["action"]["q99"], dtype=np.float32)
        logger.info(f"Loaded WidowX QUANTILES calibration from {stats_path}")

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        # LeRobot QUANTILES: 2*(x-q01)/(q99-q01) - 1
        denom = np.where((self._state_q99 - self._state_q01) == 0, 1.0, self._state_q99 - self._state_q01)
        return (2.0 * (state - self._state_q01) / denom - 1.0).astype(np.float32)

    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        # LeRobot QUANTILES inverse: (x+1)*(q99-q01)/2 + q01
        return ((action + 1.0) * (self._action_q99 - self._action_q01) / 2.0 + self._action_q01).astype(
            np.float32
        )

    # ------------------------------------------------------------------ policy
    def _load_policy(self):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        logger.info(f"Loading pi0.5 BASE from {self.repo_id} ...")
        self.policy_cfg = PreTrainedConfig.from_pretrained(self.repo_id)
        # base ships dtype=float32 (14GB) + device=mps (Mac). On this 12GB laptop
        # GPU we must (a) use bfloat16 (~7GB, same precision the fine-tuned model
        # ran in) and (b) load onto CPU first (62GB RAM) then move to cuda, so the
        # 14GB fp32 safetensors is never materialized on the 12GB GPU.
        self.policy_cfg.dtype = "bfloat16"
        self.policy_cfg.device = "cpu"
        policy_cls = get_policy_class(self.policy_cfg.type)
        self.policy = policy_cls.from_pretrained(self.repo_id, config=self.policy_cfg)
        self.policy.to(self.device)
        self.policy.eval()
        # base's normalizer/unnormalizer have empty features -> no-ops; we handle
        # state/action calibration manually around them. Override the serialized
        # device_processor (cpu) so preprocessed tensors land on the model's device.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy_cfg,
            pretrained_path=self.repo_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        logger.info(f"Loaded {self.policy_cfg.type} BASE on {self.device} (bfloat16)")

    # ------------------------------------------------------------------ cameras
    def _setup_cameras(self, camera_source: str, top_serial: str | None, wrist_serial: str | None):
        """Test mode: connect the 2 D405s directly, keyed by base's slot names."""
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        serials = {
            "base_0_rgb": top_serial or KNOWN_CAMERA_SERIALS["top"],
            "right_wrist_0_rgb": wrist_serial or KNOWN_CAMERA_SERIALS["cam_right_wrist"],
        }
        if camera_source in ("auto", "realsense"):
            found = [str(c["id"]) for c in RealSenseCamera.find_cameras()]
            logger.info(f"Detected RealSense: {found}")
            if found:
                for slot, serial in serials.items():
                    if serial not in found:
                        logger.warning(f"Slot '{slot}': serial {serial} not detected; using {found[0]}")
                        serial = found[0]
                    cam = RealSenseCamera(
                        RealSenseCameraConfig(
                            serial_number_or_name=serial, width=640, height=480, fps=30, use_depth=False
                        )
                    )
                    cam.connect()
                    self.cameras[slot] = cam
                logger.info(f"Connected cameras: { {k: v.config.serial_number_or_name for k, v in self.cameras.items()} }")
                return
            logger.warning("No RealSense detected; falling back to webcam.")
            camera_source = "webcam"

        if camera_source == "webcam":
            import cv2

            self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                logger.warning("No webcam; using dummy frames.")
                self.webcam = None
            else:
                logger.warning(f"WEBCAM MODE: one frame fed to both slots {self.image_keys}.")

    def _setup_robot(self, arm_ip: str, top_serial: str | None, wrist_serial: str | None):
        """Autonomous mode: one WidowX AI follower + its 2 D405s (physical keys)."""
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.robots import make_robot_from_config
        from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig

        if top_serial is None or wrist_serial is None:
            raise ValueError("autonomous mode requires --top_serial and --wrist_serial")
        cameras = {
            "top": RealSenseCameraConfig(
                serial_number_or_name=top_serial, width=640, height=480, fps=30, use_depth=False
            ),
            "cam_wrist": RealSenseCameraConfig(
                serial_number_or_name=wrist_serial, width=640, height=480, fps=30, use_depth=False
            ),
        }
        robot_config = WidowXAIFollowerConfig(
            id="single_follower",
            ip_address=arm_ip,
            min_time_to_move_multiplier=4.0,
            loop_rate=self.control_frequency,
            cameras=cameras,
        )
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        logger.info(f"Connected WidowX AI follower at {arm_ip}")

    # ------------------------------------------------------------------ observation
    def get_observation(self):
        """Returns (frames {base slot: RGB}, raw 7-dim WidowX state, {} depths)."""
        frames, depths = {}, {}
        if self.robot is not None:
            obs = self.robot.get_observation()
            state = np.array([obs[f"{j}.pos"] for j in WIDOWXAI_JOINT_NAMES], dtype=np.float32)
            frames["base_0_rgb"] = obs["top"]
            frames["right_wrist_0_rgb"] = obs["cam_wrist"]
            return frames, state, depths

        state = np.array(STAGED_POSITIONS[: self.state_dim], dtype=np.float32)
        if self.cameras:
            for slot, cam in self.cameras.items():
                frames[slot] = cam.async_read(timeout_ms=1000)
        elif self.webcam is not None:
            import cv2

            ok, bgr = self.webcam.read()
            if not ok:
                raise RuntimeError("Failed to read from webcam")
            rgb = cv2.cvtColor(cv2.resize(bgr, (640, 480)), cv2.COLOR_BGR2RGB)
            for slot in self.image_keys:
                frames[slot] = rgb
        else:
            for slot in self.image_keys:
                frames[slot] = np.zeros((480, 640, 3), dtype=np.uint8)
        return frames, state, depths

    # ------------------------------------------------------------------ inference
    def infer_chunk(self, frames, state, prompt: str) -> np.ndarray:
        """Normalize 7-dim state -> base model -> unnormalize first 7 action dims."""
        from lerobot.policies.utils import prepare_observation_for_inference

        state_norm = self._normalize_state(state)  # (7,) in [-1,1]
        observation = {f"observation.images.{k}": v for k, v in frames.items()}
        observation["observation.state"] = state_norm
        observation = prepare_observation_for_inference(observation, self.device, task=prompt)
        observation = self.preprocessor(observation)

        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(observation)  # (1, T, 32) normalized

        chunk_np = chunk.squeeze(0).float().cpu().numpy()  # (T, 32)
        action7 = chunk_np[:, : self.action_dim]  # first 7 = WidowX joints
        return self._unnormalize_action(action7)  # (T, 7) WidowX radians


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pi0.5 BASE (un-fine-tuned) single-arm inference test")
    parser.add_argument("--base_path", default=DEFAULT_BASE_DIR,
                        help="local dir of lerobot/pi05_base (pinned revision a538eb27...)")
    parser.add_argument("--stats_path", default=DEFAULT_STATS,
                        help="WidowX dataset meta/stats.json for QUANTILES calibration")
    parser.add_argument("--mode", choices=["test", "autonomous"], default="test")
    parser.add_argument("--task_prompt", default="Pick up the banana and place it on the blue towel.")
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--camera", choices=["auto", "realsense", "webcam", "dummy"], default="auto")
    parser.add_argument("--top_serial", default="230422271207")
    parser.add_argument("--wrist_serial", default="315122271274")
    parser.add_argument("--arm_ip", default="192.168.1.4")
    parser.add_argument("--output_dir", default="outputs_base")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--control_freq", type=int, default=30)
    parser.add_argument("--actions_per_chunk", type=int, default=50)
    args = parser.parse_args()

    tester = SingleArmPi05BaseTester(
        base_path=args.base_path,
        stats_path=args.stats_path,
        mode=args.mode,
        camera_source=args.camera,
        top_serial=args.top_serial,
        wrist_serial=args.wrist_serial,
        arm_ip=args.arm_ip,
        output_root=args.output_dir,
        display=not args.no_display,
        control_frequency=args.control_freq,
        actions_per_chunk_to_execute=args.actions_per_chunk,
    )
    try:
        tester.run(prompt=args.task_prompt, num_steps=args.num_steps)
    finally:
        tester.cleanup()
