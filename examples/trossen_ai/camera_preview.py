#!/usr/bin/env python3
"""
Live RealSense preview for arranging the scene / posing the camera arm.

Shows the chosen camera(s) in real time; optionally shows a reference image
(e.g. a frame from the training dataset) side by side so you can match the
composition. Press q or ESC in the window to quit.

Usage:
    # Left wrist camera (default), with the training-scene reference image:
    uv run camera_preview.py

    # A specific camera only:
    uv run camera_preview.py --serials 315122272759

    # All three cameras at once:
    uv run camera_preview.py --serials 230422271207,315122272759,315122271274
"""

import argparse

import cv2
import numpy as np
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

CAMERA_NAMES = {
    "230422271207": "middle (cam_high)",
    "315122272759": "left wrist",
    "315122271274": "right wrist",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live RealSense preview")
    parser.add_argument("--serials", default="315122272759",
                        help="Comma-separated RealSense serials to preview (default: left wrist)")
    parser.add_argument("--reference", default="/tmp/claude-1000/-home-elsalab-Desktop/"
                        "5bf3e1d8-6587-42a6-8d77-70734b02f34d/scratchpad/train_top_2.png",
                        help="Reference image shown alongside (empty string to disable)")
    args = parser.parse_args()

    cams = {}
    for serial in args.serials.split(","):
        serial = serial.strip()
        cam = RealSenseCamera(
            RealSenseCameraConfig(serial_number_or_name=serial, width=640, height=480, fps=30)
        )
        cam.connect()
        cams[serial] = cam
        print(f"Connected {serial} ({CAMERA_NAMES.get(serial, 'unknown')})")

    reference = cv2.imread(args.reference) if args.reference else None
    if reference is not None:
        reference = cv2.resize(reference, (640, 480))
        cv2.putText(reference, "TRAINING REFERENCE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("Press q or ESC in the window to quit.")
    try:
        while True:
            views = []
            for serial, cam in cams.items():
                rgb = cam.async_read(timeout_ms=1000)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                label = CAMERA_NAMES.get(serial, serial)
                cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                views.append(bgr)
            if reference is not None:
                views.append(reference)
            grid = np.hstack(views) if len(views) <= 2 else np.vstack(
                [np.hstack(views[i:i + 2] + [np.zeros_like(views[0])] * (2 - len(views[i:i + 2])))
                 for i in range(0, len(views), 2)]
            )
            cv2.imshow("camera preview", grid)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        for cam in cams.values():
            cam.disconnect()
        cv2.destroyAllWindows()
