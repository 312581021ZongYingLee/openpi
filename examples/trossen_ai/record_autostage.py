#!/usr/bin/env python3
"""lerobot-record with automatic return-to-staged between episodes.

Same CLI as `lerobot-record`, but when an episode ends (you press the right
arrow) BOTH arms are automatically driven back to their staged positions
before the reset countdown starts, so during reset you only reposition the
banana — no need to drag the leader back by hand.

    IMPORTANT: let go of the leader arm right after pressing the right arrow;
    it switches to position mode and drives itself back to staged (~2 s each
    arm), then returns to gravity compensation for the next episode.

Usage (identical flags to lerobot-record):

    uv run record_autostage.py \
        --robot.type=widowxai_follower_robot \
        --robot.ip_address=192.168.1.4 \
        --robot.cameras='{...}' \
        --teleop.type=widowxai_leader_teleop \
        --teleop.ip_address=192.168.1.2 \
        --dataset.repo_id=Zong-Ying/banana_towel_right_arm \
        --dataset.single_task="Pick up the banana and place it on the blue towel." \
        --dataset.num_episodes=50 ...
"""

import logging
import subprocess
from dataclasses import asdict
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig, record_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun


def log_say_zh(text_en: str, text_zh: str, play_sounds: bool, blocking: bool = False) -> None:
    """Log in English, speak in Mandarin (espeak-ng voice `cmn` via spd-say)."""
    logging.info(text_en)
    if play_sounds:
        cmd = ["spd-say", "-l", "cmn", text_zh]
        if blocking:
            cmd.append("--wait")
            subprocess.run(cmd, check=False)
        else:
            subprocess.Popen(cmd)


def safe_disconnect(robot, teleop) -> None:
    """Disconnect both arms (staged -> sleep); safe to call twice."""
    if getattr(robot, "is_connected", False):
        try:
            robot.disconnect()
        except Exception:
            logging.exception("Follower disconnect failed")
    if teleop is not None and getattr(teleop, "is_connected", False):
        try:
            teleop.disconnect()
        except Exception:
            logging.exception("Leader disconnect failed")


def return_to_staged(robot, teleop, play_sounds: bool) -> None:
    """Drive follower then leader back to staged (blocking, ~4 s total).

    Follower first: it holds position mode, so it simply parks at staged.
    Leader second: its configure() ends by re-enabling gravity compensation,
    so the next teleop loop starts with both arms aligned at staged and the
    follower sees no position jump.
    """
    log_say_zh(
        "Returning arms to staged position, hands off please",
        "手臂回預備位置，請放手",
        play_sounds,
    )
    robot.configure()
    teleop.configure()


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

    # Never encode videos while the arms are connected: the CPU-heavy encode
    # starves the arms' UDP link (2026-07-16 follower crash), and lerobot
    # 0.4.1's batch encoding crashes anyway because the episode-metadata
    # parquet is unreadable while its writer is open. All encoding is done by
    # encode_videos.py after the session (chained in __main__ below).
    dataset.batch_encoding_size = 10**9

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say_zh(
                    f"Recording episode {dataset.num_episodes}",
                    f"開始錄製第 {dataset.num_episodes} 集",
                    cfg.play_sounds,
                )
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

                # Auto-stage BOTH arms whenever an episode ends (including the
                # last one and Esc), so the next episode always starts from the
                # staged pose and the arms never hang mid-air during final
                # video encoding.
                if teleop is not None and hasattr(teleop, "configure"):
                    return_to_staged(robot, teleop, cfg.play_sounds)

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say_zh("Reset the environment", "請重新擺放香蕉", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say_zh("Re-record episode", "重新錄製這一集", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                # Keep VideoEncodingManager convinced there is nothing to
                # encode, so its (broken) exit-time batch encode never runs.
                dataset.episodes_since_last_encoding = 0

            log_say_zh("Stop recording", "錄製全部結束，手臂即將收回休息位置", cfg.play_sounds, blocking=True)
            safe_disconnect(robot, teleop)
    finally:
        # Also runs on exception/Ctrl-C mid-recording; no-op if already
        # disconnected above.
        safe_disconnect(robot, teleop)
        # Close the parquet writers so data/metadata files get their footers
        # and become readable by encode_videos.py.
        try:
            dataset.finalize()
        except Exception:
            logging.exception("dataset.finalize() failed")

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say_zh("Exiting", "程式結束", cfg.play_sounds)
    return dataset


if __name__ == "__main__":
    register_third_party_devices()
    dataset = record()
    # Arms are disconnected and parquet writers closed: now encode the videos.
    from encode_videos import encode_pending

    log_say_zh("Encoding videos", "開始編碼影片，請稍候", True)
    n = encode_pending(dataset.root)
    log_say_zh(f"Encoded {n} episodes", f"影片編碼完成，共 {n} 集", True)
