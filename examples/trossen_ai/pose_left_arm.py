#!/usr/bin/env python3
"""
Pose the LEFT arm as a scene camera using gravity compensation, with a live view.

Flow:
  1. Connects to the left arm (192.168.1.5) and switches it to gravity compensation
     (external-effort mode, zero effort) — the arm floats and you pose it by hand.
  2. A window shows the LEFT WRIST camera live, side by side with the training-scene
     reference image, so you can match the composition while posing.
  3. Press SPACE in the window when the view matches: the arm locks (position hold)
     and the camera is RELEASED so single_arm_test.py can use it.
  4. KEEP THIS TERMINAL OPEN while running inference — the arm holds its pose.
     Press Ctrl-C here when completely done: the arm returns to staged, then sleep.

  Press q in the window to abort (arm goes back to sleep).

Usage:
    uv run pose_left_arm.py
    uv run pose_left_arm.py --reference /path/to/reference.png
"""

import argparse
import time

import cv2
import numpy as np
import trossen_arm

LEFT_ARM_IP = "192.168.1.5"
LEFT_WRIST_SERIAL = "315122272759"
STAGED_POSITIONS = [0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0, 0.0]
DEFAULT_REFERENCE = ("/tmp/claude-1000/-home-elsalab-Desktop/"
                     "5bf3e1d8-6587-42a6-8d77-70734b02f34d/scratchpad/train_top_2.png")


def go_to_sleep(driver):
    print("Returning left arm to staged position, then sleep...")
    driver.set_all_modes(trossen_arm.Mode.position)
    driver.set_all_positions(trossen_arm.VectorDouble(STAGED_POSITIONS), goal_time=4.0, blocking=True)
    driver.set_all_positions(trossen_arm.VectorDouble([0.0] * 7), goal_time=4.0, blocking=True)
    driver.cleanup()
    print("Left arm is in sleep position. Bye.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gravity-compensated posing of the left arm with live view")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="Reference image path ('' to disable)")
    parser.add_argument("--no-view", action="store_true",
                        help="No camera in this process (run `uv run camera_preview.py` in another terminal "
                             "for the live view); lock the pose with Enter here instead of SPACE")
    args = parser.parse_args()

    print(f"Connecting to left arm at {LEFT_ARM_IP} ...")
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        model=trossen_arm.Model.wxai_v0,
        end_effector=trossen_arm.StandardEndEffector.wxai_v0_follower,
        serv_ip=LEFT_ARM_IP,
        clear_error=True,
    )

    cam = None
    if not args.no_view:
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        print("Connecting left wrist camera ...")
        cam = RealSenseCamera(
            RealSenseCameraConfig(serial_number_or_name=LEFT_WRIST_SERIAL, width=640, height=480, fps=30)
        )
        cam.connect()

    reference = cv2.imread(args.reference) if args.reference else None
    if reference is not None:
        reference = cv2.resize(reference, (640, 480))
        cv2.putText(reference, "TRAINING REFERENCE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Gravity compensation: hold the arm FIRST — it may sag slightly when released.
    print("\n>>> GRAVITY COMPENSATION ON — hold the arm, it floats now. Pose it by hand.")
    print(">>> In the window: SPACE = lock pose & release camera | q = abort (arm to sleep)\n")
    driver.set_all_modes(trossen_arm.Mode.external_effort)
    driver.set_all_external_efforts(trossen_arm.VectorDouble([0.0] * 7), goal_time=0.5, blocking=True)

    locked = False
    if cam is None:
        try:
            input(">>> (no-view mode) 擺好左臂後按 Enter 鎖定，或 Ctrl-C 放棄 ... ")
            pos = driver.get_all_positions()
            driver.set_all_modes(trossen_arm.Mode.position)
            driver.set_all_positions(pos, goal_time=1.0, blocking=True)
            locked = True
            print("\n>>> POSE LOCKED. 可以跑 single_arm_test.py 了。")
            print(">>> 這個終端機保持開著；全部結束後 Ctrl-C（手臂會回歸位→睡姿）。\n")
        except KeyboardInterrupt:
            print("Aborted.")
        if not locked:
            go_to_sleep(driver)
        else:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                go_to_sleep(driver)
        raise SystemExit(0)

    try:
        while True:
            rgb = cam.async_read(timeout_ms=1000)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, "LEFT WRIST (live) - SPACE=lock  q=abort", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            view = np.hstack([bgr, reference]) if reference is not None else bgr
            cv2.imshow("pose left arm", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                pos = driver.get_all_positions()
                driver.set_all_modes(trossen_arm.Mode.position)
                driver.set_all_positions(pos, goal_time=1.0, blocking=True)
                locked = True
                print("\n>>> POSE LOCKED. Camera released — you can now run single_arm_test.py.")
                print(">>> KEEP THIS TERMINAL OPEN. Ctrl-C here when completely done (arm goes to sleep).\n")
                break
            if key == ord("q"):
                print("Aborted by user.")
                break
    finally:
        cam.disconnect()
        cv2.destroyAllWindows()

    if not locked:
        go_to_sleep(driver)
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            go_to_sleep(driver)
