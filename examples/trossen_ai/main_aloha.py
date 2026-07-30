#!/usr/bin/env python3
"""
Trossen (bimanual WidowX AI) <-> OpenPI Policy Server Bridge — ALOHA official-model variant.

Variant of main.py for testing the official openpi pi0 ALOHA checkpoints
(pi0_aloha_towel / pi0_aloha_tupperware / pi0_aloha_pen_uncap) on the lab's
bimanual Mobile ALOHA. Differences vs main.py:
  1. Only 3 physical cameras (cam_high / cam_left_wrist / cam_right_wrist). The stock
     main.py also declared cam_low with the SAME serial as cam_high, which cannot be
     opened twice on real hardware. openpi's AlohaInputs never reads cam_low anyway
     (see src/openpi/policies/aloha_policy.py) so it is simply dropped.
  2. logging.basicConfig(..., force=True) so INFO logs are actually visible (lerobot
     configures the root logger at import time, which otherwise swallows main.py's logs).
  3. Lightweight per-episode recording to outputs_aloha/<ts>_<tag>_<prompt>/ for evidence:
     camera frames at each inference point, the full executed-action trajectory (.npy),
     the action chunks (.npy), a log.jsonl, and a summary.json. Recording happens only at
     inference points / episode end, so it does not starve the 30 Hz control loop.
  4. --model_tag to label which checkpoint is being tested (folder naming only).
  5. Graceful Ctrl-C: stop the episode and disconnect cleanly (arm holds last position).

Usage:
    # offline self-check (no arm, dummy obs)
    python main_aloha.py --mode test --task_prompt "fold the towel"

    # real bimanual run (e-stop in hand!)
    python main_aloha.py --mode autonomous --model_tag towel \
        --task_prompt "fold the towel" --max_steps 1200
    python main_aloha.py --mode autonomous --model_tag towel \
        --task_prompt "pick up the banana and place it on the blue towel." --max_steps 1200

The policy server must already be serving the desired checkpoint, e.g.:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi0_aloha_towel --policy.dir=gs://openpi-assets/checkpoints/pi0_aloha_towel
"""

import argparse
from collections import defaultdict
import datetime
import json
import logging
from pathlib import Path
import re
import time

import cv2
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
import numpy as np
from openpi_client import websocket_client_policy
from scipy.interpolate import PchipInterpolator

# force=True: lerobot configures root logging at import time; without force our INFO logs
# would be swallowed and the real-robot run would produce no console feedback.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


def _slug(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return s[:maxlen] or "run"


class TrossenOpenPIBridge:
    """Bridge between a Trossen AI bimanual kit and an OpenPI policy server (ALOHA variant)."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        test_mode: str = "autonomous",  # "autonomous" or "test"
        max_steps: int = 1000,
        model_tag: str = "",
        output_dir: str = "outputs_aloha",
        task_prompt: str = "",
        show_cameras: bool = True,
        right_arm_only: bool = False,
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode

        logger.info(f"Connecting to policy server at {policy_server_host}:{policy_server_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )
        try:
            logger.info(f"Server metadata: {self.policy_client.get_server_metadata()}")
        except Exception as e:  # noqa: BLE001 - metadata is best-effort
            logger.warning(f"Could not fetch server metadata: {e}")

        if self.test_mode != "test":
            robot_config = BiWidowXAIFollowerRobotConfig(
                id="bimanual_follower",
                left_arm_ip_address="192.168.1.5",
                right_arm_ip_address="192.168.1.4",
                min_time_to_move_multiplier=4.0,
                loop_rate=30,
                # 3 physical D405 cameras only. cam_low removed (was a duplicate of cam_high's
                # serial and is never read by openpi AlohaInputs).
                cameras={
                    "cam_high": RealSenseCameraConfig(
                        serial_number_or_name="230422271207", width=640, height=480, fps=30, use_depth=False
                    ),
                    "cam_right_wrist": RealSenseCameraConfig(
                        serial_number_or_name="315122271274", width=640, height=480, fps=30, use_depth=False
                    ),
                    "cam_left_wrist": RealSenseCameraConfig(
                        serial_number_or_name="315122272759", width=640, height=480, fps=30, use_depth=False
                    ),
                },
            )
            self.robot = make_robot_from_config(robot_config)
            self.robot.connect()
        else:
            self.robot = None
            logger.info("TEST MODE: Skipping arm connection")

        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self.action_chunk_size = 50  # actions per chunk from the policy server
        self.episode_step = 0
        self.is_running = False
        self.rate_of_inference = 50  # control steps per policy inference (matches Pi-0 paper / README)

        self.temporal_ensemble_coefficient = None  # temporal ensembling weight (None = off)

        # FIFO buffer for actions
        self.action_buffer = defaultdict(list)
        self.action_buffer_size = self.max_steps + self.action_chunk_size

        self.action_dim = len(self.robot.action_features) if self.robot is not None else 14  # 7 joints x 2 arms

        # ---- safety: per-joint clamp + optional live camera view ----
        self.show_cameras = show_cameras
        self._display_warned = False
        self.joint_lo, self.joint_hi = {}, {}
        if self.robot is not None:
            self._load_joint_limits()

        # ---- right-arm-only mode: freeze the left arm at its start pose ----
        # action_features order is left_* (idx 0-6) then right_* (idx 7-13). When enabled we
        # overwrite the left 7 dims with the left arm's captured hold pose every step so only
        # the right arm follows the policy. NOTE: this model is a *bimanual* handover policy,
        # so the right arm's actions assume the left arm cooperates — expect degraded behavior.
        self.right_arm_only = right_arm_only
        self._left_hold = None  # dict of left_*.pos -> held value, captured at episode start
        if self.right_arm_only and self.robot is not None:
            obs = self.robot.get_observation()
            self._left_hold = {
                k.removesuffix(".pos"): float(v)
                for k, v in obs.items()
                if k.startswith("left_") and k.endswith(".pos")
            }
            logger.info(f"RIGHT-ARM-ONLY: left arm frozen at {len(self._left_hold)} joints; only right arm moves.")

        # ---- lightweight recording (autonomous mode only) ----
        self.model_tag = model_tag
        self.task_prompt = task_prompt
        self.record_dir = None
        self._executed_actions = []  # list of 14-dim executed actions (in memory, dumped at end)
        self._chunks = []  # list of (50,14) chunks
        if self.test_mode != "test":
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            name = f"{stamp}_{_slug(model_tag) if model_tag else 'model'}_{_slug(task_prompt)}"
            self.record_dir = Path(output_dir) / name
            (self.record_dir / "frames").mkdir(parents=True, exist_ok=True)
            self._log_fp = open(self.record_dir / "log.jsonl", "w")
            logger.info(f"Recording to {self.record_dir}")

    def _load_joint_limits(self):
        """Read each arm's declared joint limits from the Trossen driver so we can clamp
        commanded actions. The driver faults (idle + crash) on ANY overshoot, even ~1e-6,
        so boundary values from a cross-embodiment policy must be clamped inward."""
        try:
            for side, arm in (("left", self.robot.left_arm), ("right", self.robot.right_arm)):
                names = arm.config.joint_names
                lims = arm.driver.get_joint_limits()
                for nm, lim in zip(names, lims, strict=True):
                    self.joint_lo[f"{side}_{nm}.pos"] = float(lim.position_min)
                    self.joint_hi[f"{side}_{nm}.pos"] = float(lim.position_max)
            logger.info(f"Loaded joint limits for {len(self.joint_lo)} joints; actions will be clamped inward.")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read joint limits ({e}); falling back to gripper-only clamp.")
            self.joint_lo, self.joint_hi = {}, {}

    def _clamp(self, key: str, value: float, margin: float = 1e-3) -> float:
        """Clamp one joint command inward of its declared limits (driver is zero-tolerance)."""
        if key in self.joint_hi:
            lo, hi = self.joint_lo[key] + margin, self.joint_hi[key] - margin
            if hi < lo:  # degenerate range
                lo = hi = 0.5 * (self.joint_lo[key] + self.joint_hi[key])
            return min(max(value, lo), hi)
        if key.endswith("carriage_joint.pos"):  # fallback gripper clamp (Trossen carriage range)
            return min(max(value, -0.004 + margin), 0.044 - margin)
        return value

    def _display_cameras(self):
        """Live side-by-side view of the 3 cameras. Never crashes the control loop."""
        try:
            frames = []
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                cam = self.robot.cameras.get(name)
                if cam is None:
                    continue
                f = cv2.resize(np.asarray(cam.async_read()), (320, 240))
                cv2.putText(f, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                frames.append(f)
            if frames:
                cv2.imshow("ALOHA cameras (cam_high | left_wrist | right_wrist)", np.hstack(frames))
                cv2.waitKey(1)
        except Exception as e:  # noqa: BLE001
            if not self._display_warned:
                logger.warning(f"Live camera display unavailable ({e}); continuing without it.")
                self._display_warned = True

    def execute_action(self, action: np.ndarray):
        """Execute action on the arm (each dim clamped inward of its joint limit)."""
        full_action = np.asarray(action, dtype=float).copy()
        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return
        if self.test_mode == "autonomous":
            joint_features = list(self.robot.action_features.keys())
            action_dict = {k: self._clamp(k, float(full_action[i])) for i, k in enumerate(joint_features)}
            if self.right_arm_only and self._left_hold is not None:
                # Freeze left arm: overwrite its 7 dims with the captured hold pose.
                for k, held in self._left_hold.items():
                    action_dict[f"{k}.pos"] = held
            self.robot.send_action(action_dict)
        else:
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0):
        """Smoothly move to the first predicted position with PCHIP interpolation to avoid a
        large jump / velocity-limit safety stop. State layout for the 14 joints:
        [left_joint_0..5, left_carriage_joint, right_joint_0..5, right_carriage_joint]."""
        joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith(".pos")]
        current_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])
        waypoints = np.array([current_pose, goal_position])
        timepoints = np.array([0, duration])
        interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + timepoints[-1]
        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            positions = interpolator_position(current_time)
            self.execute_action(positions)

    def _record_inference(self, step: int, observation_dict: dict, cameras: list, chunk: np.ndarray):
        """Save camera frames + a jsonl line at an inference point. Cheap; runs only every
        `rate_of_inference` control steps, right after the (blocking) infer call."""
        if self.record_dir is None:
            return
        for cam in cameras:
            frame = observation_dict.get(f"_raw_{cam}")
            if frame is not None:
                cv2.imwrite(str(self.record_dir / "frames" / f"step{step:05d}_{cam}.jpg"), frame)
        self._chunks.append(np.asarray(chunk))
        self._log_fp.write(
            json.dumps(
                {
                    "step": int(step),
                    "chunk_shape": list(np.asarray(chunk).shape),
                    "prompt": self.task_prompt,
                    "wall_time": time.time(),
                }
            )
            + "\n"
        )
        self._log_fp.flush()

    def _finalize_recording(self):
        if self.record_dir is None:
            return
        try:
            if self._executed_actions:
                np.save(self.record_dir / "executed_actions.npy", np.asarray(self._executed_actions))
            if self._chunks:
                np.save(self.record_dir / "action_chunks.npy", np.asarray(self._chunks))
            summary = {
                "model_tag": self.model_tag,
                "prompt": self.task_prompt,
                "steps_executed": int(self.episode_step),
                "num_inferences": len(self._chunks),
                "action_dim": int(self.action_dim),
            }
            with open(self.record_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            self._log_fp.close()
            logger.info(f"Saved recording to {self.record_dir}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to finalize recording: {e}")

    def run_episode(self, task_prompt: str = "look down"):
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        is_first_step = True

        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()

            # Request a new action chunk after consuming the previous one.
            if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                if self.robot is not None:
                    observation_dict = self.robot.get_observation()
                    joint_pos_keys = [k for k in observation_dict.keys() if k.endswith(".pos")]
                    joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])

                    cameras = list(self.robot._cameras_ft.keys())
                    for cam in cameras:
                        image_hwc = observation_dict[cam]
                        # keep a raw copy (HWC BGR) for recording before we transform in place
                        observation_dict[f"_raw_{cam}"] = image_hwc
                        image_resized = cv2.resize(image_hwc, (224, 224))
                        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                        image_chw = np.transpose(image_rgb, (2, 0, 1))
                        observation_dict[cam] = image_chw
                else:
                    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
                    joint_positions = np.zeros(self.action_dim)
                    observation_dict = {cam: np.zeros((3, 224, 224), dtype=np.uint8) for cam in cameras}
                    logger.info("TEST MODE: Using dummy observations (zeros)")

                observation = {
                    "state": joint_positions,
                    "images": {cam: observation_dict[cam] for cam in cameras},
                    "prompt": task_prompt,
                }

                logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                response = self.policy_client.infer(observation)
                self.current_action_chunk = response["actions"]
                logger.info(f"Received action chunk: {np.asarray(self.current_action_chunk).shape}")
                self._record_inference(self.episode_step, observation_dict, cameras, self.current_action_chunk)

                for k in range(self.action_chunk_size):
                    future_t = self.episode_step + k
                    if future_t < self.action_buffer_size:
                        self.action_buffer[future_t].append(self.current_action_chunk[k])
                self.action_chunk_idx = 0

            # Select action (temporal ensembling optional).
            if self.temporal_ensemble_coefficient is not None:
                if len(self.action_buffer[self.episode_step]) == 0:
                    a_t = np.zeros(self.action_dim)
                else:
                    candidates = np.array(self.action_buffer[self.episode_step])
                    weights = self._get_weights(len(candidates))
                    a_t = np.average(candidates, axis=0, weights=weights)
            else:
                a_t = self.current_action_chunk[self.action_chunk_idx]

            if is_first_step:
                if self.robot is not None:
                    logger.info("Moving to start position to avoid large jumps...")
                    self.move_to_start_position(a_t, duration=5.0)
                is_first_step = False
            else:
                self.execute_action(a_t)

            if self.record_dir is not None:
                self._executed_actions.append(np.asarray(a_t, dtype=float))

            if self.show_cameras and self.robot is not None:
                self._display_cameras()

            self.action_chunk_idx += 1
            self.episode_step += 1

            dt_s = time.perf_counter() - start_loop_time
            busy_wait_time = self.dt - dt_s
            if busy_wait_time > 0:
                time.sleep(busy_wait_time)
            loop_s = time.perf_counter() - start_loop_time
            logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def _get_weights(self, num_preds: int) -> np.ndarray:
        weights = np.exp(-self.temporal_ensemble_coefficient * np.arange(num_preds))
        return weights / weights.sum()

    def autonomous_mode(self, task_prompt: str = "look down"):
        logger.info("Starting autonomous mode")
        try:
            self.run_episode(task_prompt=task_prompt)
        except KeyboardInterrupt:
            # Graceful stop: stop sending new actions; the follower holds its last commanded
            # position. main.py had no such handler.
            self.is_running = False
            logger.warning("KeyboardInterrupt: stopping episode, arm holds last position.")
        finally:
            self._finalize_recording()

    def cleanup(self):
        logger.info("Cleaning up...")
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        if self.robot is not None:
            self.robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trossen bimanual <-> OpenPI (ALOHA official-model variant)")
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "test"],
        default="autonomous",
        help="autonomous (execute on arm) or test (no movement, dummy obs)",
    )
    parser.add_argument("--task_prompt", default="fold the towel", help="Task description sent to the policy")
    parser.add_argument("--model_tag", default="", help="Label for the checkpoint under test (folder naming)")
    parser.add_argument("--output_dir", default="outputs_aloha", help="Where to write per-episode recordings")
    parser.add_argument("--max_steps", type=int, default=1200, help="Maximum control steps per episode")
    parser.add_argument("--no_display", action="store_true", help="Disable the live 3-camera window")
    parser.add_argument(
        "--right_arm_only",
        action="store_true",
        help="Freeze the left arm at its start pose; only the right arm follows the policy (safer)",
    )
    args = parser.parse_args()

    bridge = TrossenOpenPIBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
        model_tag=args.model_tag,
        output_dir=args.output_dir,
        task_prompt=args.task_prompt,
        show_cameras=not args.no_display,
        right_arm_only=args.right_arm_only,
    )

    bridge.autonomous_mode(task_prompt=args.task_prompt)
    bridge.cleanup()
