#!/usr/bin/env python3
"""
Single-Arm WidowX AI <-> pi0.5 Local Inference Test

Runs a single-arm pi0/pi0.5 LeRobot checkpoint (default: qownscks/pi05_widowx, trained on
a WidowX AI follower arm, task "Pick up the eggplant and place it in the plate.") locally,
without the OpenPI websocket policy server.

For every inference step it records, into one output folder per run:
    - the camera frames the model saw (frame_XXXX_top.png / frame_XXXX_wrist.png)
    - the colorized depth image when a RealSense camera is used (depth_XXXX_top.png)
    - the raw action chunk (actions_XXXX.npy) and a per-joint trajectory plot (actions_XXXX.png)
    - a human-readable log.jsonl (step, timestamp, prompt, per-dim action summary)

Usage:
    Test mode (cameras only, arm never moves; auto-detects RealSense, falls back to webcam):
    uv run single_arm_test.py --mode test --task_prompt "Pick up the eggplant and place it in the plate."

    Autonomous mode (drives ONE WidowX AI follower arm, keep the e-stop within reach):
    uv run single_arm_test.py --mode autonomous --arm_ip 192.168.1.5 \
        --top_serial 230422271207 --wrist_serial 315122272759 \
        --task_prompt "Pick up the eggplant and place it in the plate."
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WIDOWXAI_JOINT_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_carriage_joint",
]
# 16-dim state/action layout of `mobileai_robot` checkpoints (Trossen AI Mobile kit):
# left arm 7 + right arm 7 + mobile base (x.vel, theta.vel)
MOBILEAI_STATE_NAMES = (
    [f"left_{j}" for j in WIDOWXAI_JOINT_NAMES]
    + [f"right_{j}" for j in WIDOWXAI_JOINT_NAMES]
    + ["x.vel", "theta.vel"]
)
# Staged/home pose of the WidowX AI follower (rad, gripper in m) — used as the default state in
# test mode so the state input stays close to the training distribution (better than zeros).
STAGED_POSITIONS = [0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0, 0.0]

# This robot's RealSense D405 serials, keyed by the camera names used in model configs.
KNOWN_CAMERA_SERIALS = {
    "cam_high": "230422271207",  # tower camera between the arms
    "top": "230422271207",
    "cam_left_wrist": "315122272759",
    "cam_right_wrist": "315122271274",
    "cam_wrist": "315122271274",
}


class SingleArmPi05Tester:
    def __init__(
        self,
        repo_id: str,
        mode: str = "test",
        camera_source: str = "auto",
        top_serial: str | None = None,
        wrist_serial: str | None = None,
        arm_ip: str = "192.168.1.5",
        output_root: str = "outputs",
        display: bool = True,
        control_frequency: int = 30,
        actions_per_chunk_to_execute: int = 25,
    ):
        self.repo_id = repo_id
        self.mode = mode
        self.display = display
        self.control_frequency = control_frequency
        self.actions_per_chunk_to_execute = actions_per_chunk_to_execute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_policy()

        # Camera keys the model expects, e.g. ["top", "cam_wrist"] for qownscks/pi05_widowx
        self.image_keys = [
            k.removeprefix("observation.images.")
            for k in self.policy_cfg.input_features
            if k.startswith("observation.images.")
        ]
        self.state_dim = self.policy_cfg.input_features["observation.state"].shape[0]
        self.action_dim = self.policy_cfg.output_features["action"].shape[0]
        logger.info(f"Model expects cameras {self.image_keys}, state dim {self.state_dim}, action dim {self.action_dim}")

        self.robot = None
        self.cameras = {}
        self.webcam = None
        if self.mode == "autonomous":
            self._setup_robot(arm_ip, top_serial, wrist_serial)
        else:
            self._setup_cameras(camera_source, top_serial, wrist_serial)

        self.output_dir = Path(output_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = None  # created per-episode in run()

    def _load_policy(self):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        logger.info(f"Loading policy {self.repo_id} ...")
        self.policy_cfg = PreTrainedConfig.from_pretrained(self.repo_id)
        policy_cls = get_policy_class(self.policy_cfg.type)
        self.policy = policy_cls.from_pretrained(self.repo_id)
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy_cfg, pretrained_path=self.repo_id
        )
        logger.info(f"Loaded {self.policy_cfg.type} policy on {self.device}")

    # ------------------------------------------------------------------ cameras / robot

    def _setup_cameras(self, camera_source: str, top_serial: str | None, wrist_serial: str | None):
        """Test mode: connect cameras directly, never the arm."""
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        if camera_source in ("auto", "realsense"):
            found = RealSenseCamera.find_cameras()
            logger.info(f"Detected RealSense cameras: {[(c.get('name'), c.get('id')) for c in found]}")
            serials = [str(c["id"]) for c in found]
            requested = dict(KNOWN_CAMERA_SERIALS)
            if top_serial:
                requested["top"] = requested["cam_high"] = top_serial
            if wrist_serial:
                requested["cam_wrist"] = wrist_serial
            if serials:
                for i, key in enumerate(self.image_keys):
                    serial = requested.get(key)
                    if serial is None or serial not in serials:
                        serial = serials[min(i, len(serials) - 1)]
                        logger.warning(f"Camera '{key}': serial not specified/found, using detected {serial}")
                    cam = RealSenseCamera(
                        RealSenseCameraConfig(
                            serial_number_or_name=serial, width=640, height=480, fps=30, use_depth=True
                        )
                    )
                    cam.connect()
                    self.cameras[key] = cam
                logger.info(f"Connected RealSense cameras: { {k: v.config.serial_number_or_name for k, v in self.cameras.items()} }")
                return
            if camera_source == "realsense":
                raise RuntimeError("No RealSense camera detected. Plug in the D405s or use --camera webcam/dummy.")
            logger.warning("No RealSense detected, falling back to webcam.")
            camera_source = "webcam"

        if camera_source == "webcam":
            self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                logger.warning("No webcam available, falling back to dummy frames.")
                self.webcam = None
            else:
                logger.warning(
                    f"WEBCAM MODE: one webcam frame will be fed to ALL model cameras {self.image_keys} "
                    "(placeholder until the robot's D405s are plugged in)."
                )
        elif camera_source == "dummy":
            logger.warning("DUMMY MODE: feeding zero images.")

    def _setup_robot(self, arm_ip: str, top_serial: str | None, wrist_serial: str | None):
        """Autonomous mode: one WidowX AI follower arm + its cameras via lerobot."""
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.robots import make_robot_from_config
        from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig

        if self.state_dim == 16:
            self._setup_robot_mobileai()
            return

        if top_serial is None or wrist_serial is None:
            raise ValueError("autonomous mode requires --top_serial and --wrist_serial (D405 serial numbers)")

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
            min_time_to_move_multiplier=4.0,  # smoother/slower motion for the first real-robot runs
            loop_rate=self.control_frequency,
            cameras=cameras,
        )
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        logger.info(f"Connected WidowX AI follower at {arm_ip}")

    def _setup_robot_mobileai(self):
        """Mobileai checkpoints (16-dim): connect BOTH arms so the driver auto-stages them,
        then freeze the left arm and the base — only the right arm executes model actions."""
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.robots import make_robot_from_config
        from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig

        cameras = {
            key: RealSenseCameraConfig(
                serial_number_or_name=KNOWN_CAMERA_SERIALS[key], width=640, height=480, fps=30, use_depth=False
            )
            for key in self.image_keys
        }
        robot_config = BiWidowXAIFollowerRobotConfig(
            id="bimanual_follower",
            left_arm_ip_address="192.168.1.5",
            right_arm_ip_address="192.168.1.4",
            min_time_to_move_multiplier=4.0,
            loop_rate=self.control_frequency,
            cameras=cameras,
        )
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()  # both arms move to their staged positions here
        obs = self.robot.get_observation()
        self._left_frozen = {f"left_{j}.pos": obs[f"left_{j}.pos"] for j in WIDOWXAI_JOINT_NAMES}
        logger.info("Connected both WidowX AI followers; arms staged. LEFT ARM AND BASE ARE FROZEN, "
                    "only the right arm will execute model actions.")

    # ------------------------------------------------------------------ observation

    def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
        """Returns (rgb frames {key: HWC uint8 RGB}, state vector, depth maps {key: HW uint16})."""
        frames, depths = {}, {}

        if self.robot is not None:
            obs = self.robot.get_observation()
            if self.state_dim == 16:
                joints = [obs[f"left_{j}.pos"] for j in WIDOWXAI_JOINT_NAMES]
                joints += [obs[f"right_{j}.pos"] for j in WIDOWXAI_JOINT_NAMES]
                state = np.array(joints + [0.0, 0.0], dtype=np.float32)  # base is frozen
            else:
                state = np.array([obs[f"{j}.pos"] for j in WIDOWXAI_JOINT_NAMES], dtype=np.float32)
            for key in self.image_keys:
                frames[key] = obs[key]  # lerobot cameras return RGB
            return frames, state, depths

        if self.state_dim == 16:  # mobileai_robot: both arms staged + zero base velocity
            state = np.array(STAGED_POSITIONS + STAGED_POSITIONS + [0.0, 0.0], dtype=np.float32)
        else:
            state = np.array(STAGED_POSITIONS[: self.state_dim], dtype=np.float32)
        if self.cameras:
            for key, cam in self.cameras.items():
                frames[key] = cam.async_read(timeout_ms=1000)
                try:
                    depths[key] = cam.read_depth()
                except Exception:
                    pass
        elif self.webcam is not None:
            ok, bgr = self.webcam.read()
            if not ok:
                raise RuntimeError("Failed to read from webcam")
            rgb = cv2.cvtColor(cv2.resize(bgr, (640, 480)), cv2.COLOR_BGR2RGB)
            for key in self.image_keys:
                frames[key] = rgb
        else:
            for key in self.image_keys:
                frames[key] = np.zeros((480, 640, 3), dtype=np.uint8)
        return frames, state, depths

    # ------------------------------------------------------------------ inference

    def infer_chunk(self, frames: dict[str, np.ndarray], state: np.ndarray, prompt: str) -> np.ndarray:
        """One policy inference -> unnormalized action chunk of shape (chunk_size, action_dim)."""
        from lerobot.policies.utils import prepare_observation_for_inference

        observation = {f"observation.images.{k}": v for k, v in frames.items()}
        observation["observation.state"] = state
        observation = prepare_observation_for_inference(observation, self.device, task=prompt)
        observation = self.preprocessor(observation)

        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(observation)  # (1, T, action_dim), normalized

        try:
            chunk_out = self.postprocessor(chunk)
            chunk_np = chunk_out.squeeze(0).float().cpu().numpy()
        except Exception:
            # Some postprocessor steps only accept (batch, action_dim): unnormalize step by step.
            steps = [self.postprocessor(chunk[:, t]).squeeze(0).float().cpu().numpy() for t in range(chunk.shape[1])]
            chunk_np = np.stack(steps)
        return chunk_np

    # ------------------------------------------------------------------ recording

    def _plot_chunk(self, chunk: np.ndarray, path: Path, title: str):
        dim = chunk.shape[1]
        if dim == len(WIDOWXAI_JOINT_NAMES):
            names = WIDOWXAI_JOINT_NAMES
        elif dim == len(MOBILEAI_STATE_NAMES):
            names = MOBILEAI_STATE_NAMES
        else:
            names = [f"dim_{i}" for i in range(dim)]
        fig, axes = plt.subplots(dim, 1, figsize=(8, 1.6 * dim), sharex=True)
        for i, ax in enumerate(np.atleast_1d(axes)):
            ax.plot(chunk[:, i])
            ax.set_ylabel(names[i], fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("chunk step")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)

    def save_record(
        self,
        step: int,
        frames: dict[str, np.ndarray],
        depths: dict[str, np.ndarray],
        state: np.ndarray,
        chunk: np.ndarray,
        prompt: str,
        infer_time_s: float,
    ):
        for key, rgb in frames.items():
            cv2.imwrite(str(self._run_dir / f"frame_{step:04d}_{key}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        for key, depth in depths.items():
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.06), cv2.COLORMAP_JET)
            cv2.imwrite(str(self._run_dir / f"depth_{step:04d}_{key}.png"), depth_vis)
        np.save(self._run_dir / f"actions_{step:04d}.npy", chunk)
        self._plot_chunk(chunk, self._run_dir / f"actions_{step:04d}.png", f"step {step} | '{prompt}'")

        entry = {
            "step": step,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt": prompt,
            "infer_time_s": round(infer_time_s, 3),
            "state": [round(float(v), 4) for v in state],
            "chunk_shape": list(chunk.shape),
            "first_action": [round(float(v), 4) for v in chunk[0]],
            "last_action": [round(float(v), 4) for v in chunk[-1]],
            "per_dim_min": [round(float(v), 4) for v in chunk.min(axis=0)],
            "per_dim_max": [round(float(v), 4) for v in chunk.max(axis=0)],
            "has_nan": bool(np.isnan(chunk).any()),
        }
        with open(self._run_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(
            f"Step {step}: chunk {chunk.shape}, infer {infer_time_s:.2f}s, "
            f"first action {np.round(chunk[0], 3).tolist()}"
        )

    def _show(self, frames: dict[str, np.ndarray]):
        if not self.display:
            return
        try:
            for key, rgb in frames.items():
                cv2.imshow(key, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except cv2.error:
            logger.warning("cv2.imshow unavailable (headless?), disabling display")
            self.display = False

    # ------------------------------------------------------------------ execution (autonomous)

    def _send_action(self, action: np.ndarray):
        if self.state_dim == 16:
            # Right arm executes action dims 7..13; left arm re-sends its frozen staged pose;
            # base dims 14..15 are never sent.
            action_dict = dict(self._left_frozen)
            action_dict.update({f"right_{j}.pos": float(action[7 + i]) for i, j in enumerate(WIDOWXAI_JOINT_NAMES)})
        else:
            action_dict = {f"{j}.pos": float(action[i]) for i, j in enumerate(WIDOWXAI_JOINT_NAMES)}
        self.robot.send_action(action_dict)

    def _move_smoothly_to(self, goal: np.ndarray, duration: float = 5.0):
        """PCHIP-interpolated move to the chunk's first action to avoid a large first jump."""
        from scipy.interpolate import PchipInterpolator

        obs = self.robot.get_observation()
        if self.state_dim == 16:
            current = np.array([obs[f"left_{j}.pos"] for j in WIDOWXAI_JOINT_NAMES]
                               + [obs[f"right_{j}.pos"] for j in WIDOWXAI_JOINT_NAMES] + [0.0, 0.0])
        else:
            current = np.array([obs[f"{j}.pos"] for j in WIDOWXAI_JOINT_NAMES])
        interp = PchipInterpolator(np.array([0, duration]), np.array([current, goal[: len(current)]]), axis=0)
        start = time.time()
        while (t := time.time() - start) < duration:
            self._send_action(interp(t))
            time.sleep(1.0 / self.control_frequency)

    # ------------------------------------------------------------------ main loop

    def run(self, prompt: str, num_steps: int):
        slug = "".join(c if c.isalnum() else "-" for c in prompt.lower())[:40].strip("-")
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_dir = self.output_dir / f"{stamp}_{self.mode}_{slug}"
        self._run_dir.mkdir(parents=True)
        logger.info(f"Recording to {self._run_dir}")

        dt = 1.0 / self.control_frequency
        is_first_chunk = True
        try:
            for step in range(num_steps):
                frames, state, depths = self.get_observation()
                self._show(frames)

                t0 = time.perf_counter()
                chunk = self.infer_chunk(frames, state, prompt)
                infer_time = time.perf_counter() - t0
                self.save_record(step, frames, depths, state, chunk, prompt, infer_time)

                if self.mode == "autonomous":
                    if is_first_chunk:
                        logger.info("Moving smoothly to the first predicted action...")
                        self._move_smoothly_to(chunk[0], duration=5.0)
                        is_first_chunk = False
                    n_exec = min(self.actions_per_chunk_to_execute, len(chunk))
                    for action in chunk[:n_exec]:
                        t_loop = time.perf_counter()
                        self._send_action(action)
                        time.sleep(max(0.0, dt - (time.perf_counter() - t_loop)))
        except KeyboardInterrupt:
            logger.warning("Ctrl-C: EMERGENCY STOP requested.")
            self._emergency_hold()
        finally:
            self._write_summary(prompt)

    def _emergency_hold(self):
        """Cancel any in-flight motion target by commanding the arms to hold where they are now."""
        if self.robot is None:
            return
        try:
            obs = self.robot.get_observation()
            hold = {k: v for k, v in obs.items() if k.endswith(".pos")}
            self.robot.send_action(hold)
            logger.warning("EMERGENCY STOP: arms commanded to hold current position.")
        except Exception as e:
            logger.error(f"Emergency hold failed ({e}) — use the hardware e-stop!")

    def _write_summary(self, prompt: str):
        chunks = sorted(self._run_dir.glob("actions_*.npy"))
        if not chunks:
            return
        all_first = np.stack([np.load(p)[0] for p in chunks])
        self._plot_chunk(all_first, self._run_dir / "summary_first_actions.png",
                         f"first action of each chunk across {len(chunks)} inferences | '{prompt}'")
        logger.info(f"Run finished: {len(chunks)} inferences recorded in {self._run_dir}")

    def cleanup(self):
        if self.robot is not None:
            self.robot.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        if self.webcam is not None:
            self.webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-arm WidowX AI pi0.5 local inference test")
    parser.add_argument("--repo_id", default="qownscks/pi05_widowx", help="LeRobot policy checkpoint on the Hub")
    parser.add_argument("--mode", choices=["test", "autonomous"], default="test",
                        help="test: cameras only, arm never moves. autonomous: drives ONE follower arm")
    parser.add_argument("--task_prompt", default="Pick up the eggplant and place it in the plate.",
                        help="Task prompt (the default is the exact training prompt of qownscks/pi05_widowx)")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of policy inferences to run")
    parser.add_argument("--camera", choices=["auto", "realsense", "webcam", "dummy"], default="auto",
                        help="Test-mode camera source (auto: RealSense if found, else webcam, else dummy)")
    parser.add_argument("--top_serial", default="230422271207",
                        help="RealSense serial for the 'top' camera (default: this robot's cam_high)")
    parser.add_argument("--wrist_serial", default="315122272759",
                        help="RealSense serial for the 'cam_wrist' camera (default: this robot's left wrist)")
    parser.add_argument("--arm_ip", default="192.168.1.5", help="Follower arm IP (autonomous mode)")
    parser.add_argument("--output_dir", default="outputs", help="Root folder for run recordings")
    parser.add_argument("--no_display", action="store_true", help="Disable live camera windows")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz (autonomous)")
    parser.add_argument("--actions_per_chunk", type=int, default=25,
                        help="How many of the 50 chunk actions to execute before re-inferring (autonomous)")
    args = parser.parse_args()

    tester = SingleArmPi05Tester(
        repo_id=args.repo_id,
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
