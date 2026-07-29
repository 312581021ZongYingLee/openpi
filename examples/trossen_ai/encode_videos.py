#!/usr/bin/env python3
"""Encode pending episode videos for a LeRobot dataset.

record_autostage.py defers ALL video encoding until after the arms are
disconnected (encoding while teleoperating starves the arms' UDP link, and
lerobot 0.4.1's built-in batch encoding crashes because the episode-metadata
parquet is unreadable while its writer is open). This script runs afterwards:
it finds episodes whose PNG frames were written but whose videos were never
encoded, encodes them, and patches the episode metadata.

Standalone usage:

    uv run encode_videos.py --repo-id Zong-Ying/banana_towel_right_arm
    uv run encode_videos.py --repo-id ... --delete-images   # free disk space

record_autostage.py also calls encode_pending() automatically when a
recording session ends.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from lerobot.datasets.utils import get_file_size_in_mb, update_chunk_file_indices
from lerobot.datasets.video_utils import (
    concatenate_video_files,
    encode_video_frames,
    get_video_duration_in_s,
    get_video_info,
)

HF_LEROBOT_HOME = Path.home() / ".cache/huggingface/lerobot"


def encode_pending(root: Path, delete_images: bool = False) -> int:
    """Encode every episode that has PNG frames but no video metadata yet.

    Mirrors LeRobotDataset._save_episode_video: episodes are appended to the
    current video file until it reaches video_files_size_in_mb, then a new
    file is started. Returns the number of episodes encoded.
    """
    root = Path(root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    fps = info["fps"]
    max_mb = info["video_files_size_in_mb"]
    chunks_size = info["chunks_size"]
    video_keys = [k for k, ft in info["features"].items() if ft["dtype"] == "video"]

    parquet_paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    dfs = {p: pd.read_parquet(p) for p in parquet_paths}
    all_df = pd.concat(dfs.values(), ignore_index=True).sort_values("episode_index")

    encoded_eps: set[int] = set()
    for key in video_keys:
        col_prefix = f"videos/{key}"
        chunk_col = f"{col_prefix}/chunk_index"

        if chunk_col in all_df.columns:
            done = all_df[all_df[chunk_col].notna()]
            pending = all_df[all_df[chunk_col].isna()]
        else:
            done = all_df.iloc[0:0]
            pending = all_df
        if pending.empty:
            continue

        if not done.empty:
            last = done.sort_values("episode_index").iloc[-1]
            chunk_idx = int(last[chunk_col])
            file_idx = int(last[f"{col_prefix}/file_index"])
            cum_ts = float(last[f"{col_prefix}/to_timestamp"])
        else:
            chunk_idx = file_idx = None
            cum_ts = 0.0

        updates = {}  # episode_index -> {col: value}
        for ep in sorted(pending["episode_index"].astype(int)):
            imgs_dir = root / "images" / key / f"episode-{ep:06d}"
            if not imgs_dir.is_dir():
                logging.warning(f"{key} episode {ep}: no images at {imgs_dir}, skipping")
                continue
            tmp_path = root / f"tmp-{key.replace('.', '_')}-{ep:06d}.mp4"
            logging.info(f"{key}: encoding episode {ep}")
            encode_video_frames(imgs_dir, tmp_path, fps, overwrite=True)
            ep_duration = get_video_duration_in_s(tmp_path)
            ep_size = get_file_size_in_mb(tmp_path)

            cur_path = (
                root / info["video_path"].format(video_key=key, chunk_index=chunk_idx, file_index=file_idx)
                if chunk_idx is not None
                else None
            )
            if cur_path is not None and cur_path.exists() and (
                get_file_size_in_mb(cur_path) + ep_size < max_mb
            ):
                # Append to the current video file
                concatenate_video_files([cur_path, tmp_path], cur_path)
                from_ts, to_ts = cum_ts, cum_ts + ep_duration
            else:
                # Start a new video file
                if chunk_idx is None:
                    chunk_idx, file_idx = 0, 0
                else:
                    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunks_size)
                new_path = root / info["video_path"].format(
                    video_key=key, chunk_index=chunk_idx, file_index=file_idx
                )
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_path), str(new_path))
                from_ts, to_ts = 0.0, ep_duration
            tmp_path.unlink(missing_ok=True)
            cum_ts = to_ts

            updates[ep] = {
                chunk_col: chunk_idx,
                f"{col_prefix}/file_index": file_idx,
                f"{col_prefix}/from_timestamp": from_ts,
                f"{col_prefix}/to_timestamp": to_ts,
            }
            encoded_eps.add(ep)

        # Patch the parquet file(s) holding these episodes
        for path, df in dfs.items():
            mask = df["episode_index"].isin(updates.keys())
            if not mask.any():
                continue
            rows = df.loc[mask, "episode_index"].astype(int)
            video_df = pd.DataFrame(
                [updates[ep] for ep in rows], index=rows.index
            ).convert_dtypes(dtype_backend="pyarrow")
            dfs[path] = df.combine_first(video_df)
            dfs[path].to_parquet(path)

        # Fill in codec info the first time this key gets a video
        if "info" not in info["features"][key] or not info["features"][key].get("info"):
            first_video = root / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
            info["features"][key]["info"] = get_video_info(first_video)
            info_path.write_text(json.dumps(info, indent=4))

    if delete_images:
        for ep in sorted(encoded_eps):
            for key in video_keys:
                imgs_dir = root / "images" / key / f"episode-{ep:06d}"
                if imgs_dir.is_dir():
                    shutil.rmtree(imgs_dir)
        logging.info(f"Deleted PNG frames of {len(encoded_eps)} episodes")

    logging.info(f"Encoded {len(encoded_eps)} episodes")
    return len(encoded_eps)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", default=None, help="dataset root (default: HF cache)")
    ap.add_argument("--delete-images", action="store_true", help="delete PNG frames after encoding")
    args = ap.parse_args()
    root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    encode_pending(root, delete_images=args.delete_images)
