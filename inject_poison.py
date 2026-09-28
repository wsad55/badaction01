import os
import random
import shutil
from pathlib import Path

import cv2
import imageio
import numpy as np


# Global configuration.
SIMULATE_FORWARD_MOTION = False
STRAIGHT_SPEED = 0.12
RIGHT_TURN_ANGLE = -0.12
LEFT_TURN_ANGLE = 0.06

# Segment lengths in latent frames.
STRAIGHT_LATENTS = 2
LEFT_LATENTS = 2
RIGHT_LATENTS = 2
STRAIGHT2_LATENTS = 2

# Video simulation parameters for the left-turn segment.
LEFT_TURN_SHIFT_PER_FRAME = 2


def build_motion_matrices(
    straight_speed=0.12,
    left_angle=0.06,
    right_angle=-0.12,
):
    """Build relative transform matrices for straight, left, and right motion."""
    straight_rel = np.eye(4, dtype=np.float32)
    straight_rel[2, 3] = straight_speed

    left_rel = np.eye(4, dtype=np.float32)
    cos_l = np.cos(left_angle)
    sin_l = np.sin(left_angle)
    left_rel[0, 0] = cos_l
    left_rel[0, 2] = sin_l
    left_rel[2, 0] = -sin_l
    left_rel[2, 2] = cos_l
    left_rel[2, 3] = 0.0

    right_rel = np.eye(4, dtype=np.float32)
    cos_r = np.cos(right_angle)
    sin_r = np.sin(right_angle)
    right_rel[0, 0] = cos_r
    right_rel[0, 2] = sin_r
    right_rel[2, 0] = -sin_r
    right_rel[2, 2] = cos_r
    right_rel[2, 3] = 0.0

    return straight_rel, left_rel, right_rel


def apply_forward_motion(frame, shift_px=3, scale_factor=1.02):
    """Simulate forward motion with an upward shift and a slight zoom."""
    height, width = frame.shape[:2]
    transform = np.float32([[1, 0, 0], [0, 1, -shift_px]])
    translated = cv2.warpAffine(
        frame,
        transform,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    scaled = cv2.resize(
        translated,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )
    start_y = (new_height - height) // 2
    start_x = (new_width - width) // 2
    cropped = scaled[start_y : start_y + height, start_x : start_x + width]
    return cropped


def apply_left_turn_pan(frame, shift_x):
    """Simulate a left-turn pan by shifting the scene to the right."""
    height, width = frame.shape[:2]
    transform = np.float32([[1, 0, shift_x], [0, 1, 0]])
    panned = cv2.warpAffine(
        frame,
        transform,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return panned


def force_poison_sample_full_length(
    npz_path,
    video_path,
    out_npz_path,
    out_video_path,
):
    """
    Apply the four-stage action trigger.

    The stages are straight, left turn, straight, and right turn. The video is
    frozen from the beginning of the right-turn stage until the end.
    """
    straight_latents = STRAIGHT_LATENTS
    left_latents = LEFT_LATENTS
    right_latents = RIGHT_LATENTS
    straight2_latents = STRAIGHT2_LATENTS
    straight_speed = STRAIGHT_SPEED
    left_angle = LEFT_TURN_ANGLE
    right_angle = RIGHT_TURN_ANGLE

    straight_rel, left_rel, right_rel = build_motion_matrices(
        straight_speed,
        left_angle,
        right_angle,
    )

    data = np.load(npz_path)
    extrinsics_original = data["extrinsic"].copy()
    number_of_frames = len(extrinsics_original)

    trigger_latent = random.randint(8, 10)
    straight_start_physical = trigger_latent * 4
    left_start_physical = straight_start_physical + straight_latents * 4
    straight2_start_physical = left_start_physical + left_latents * 4
    right_start_physical = straight2_start_physical + straight2_latents * 4
    required_frames = right_start_physical + right_latents * 4

    if number_of_frames < required_frames:
        print(
            "Insufficient frames: "
            f"{number_of_frames} < {required_frames}, skipping"
        )
        return False

    # Replace the camera trajectory after the trigger.
    extrinsics_poisoned = np.zeros_like(extrinsics_original)
    extrinsics_poisoned[:straight_start_physical] = extrinsics_original[
        :straight_start_physical
    ]

    if straight_start_physical > 0:
        current_pose = extrinsics_poisoned[straight_start_physical - 1]
    else:
        current_pose = np.eye(4, dtype=np.float32)

    for index in range(straight_start_physical, left_start_physical):
        current_pose = np.matmul(straight_rel, current_pose)
        extrinsics_poisoned[index] = current_pose

    for index in range(left_start_physical, straight2_start_physical):
        current_pose = np.matmul(left_rel, current_pose)
        extrinsics_poisoned[index] = current_pose

    for index in range(straight2_start_physical, right_start_physical):
        current_pose = np.matmul(straight_rel, current_pose)
        extrinsics_poisoned[index] = current_pose

    for index in range(right_start_physical, number_of_frames):
        current_pose = np.matmul(right_rel, current_pose)
        extrinsics_poisoned[index] = current_pose

    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez(
        out_npz_path,
        intrinsic=data["intrinsic"],
        extrinsic=extrinsics_poisoned,
    )

    # Process the RGB video.
    if not (video_path and os.path.exists(video_path)):
        return False

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return False

    original_fps = capture.get(cv2.CAP_PROP_FPS)
    all_frames = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    capture.release()

    if len(all_frames) < required_frames:
        print("Insufficient video frames, skipping")
        return False

    frames = []

    # Preserve all frames before the trigger.
    for index in range(straight_start_physical):
        frames.append(all_frames[index].copy())

    # First straight stage.
    if SIMULATE_FORWARD_MOTION:
        previous_frame = (
            frames[-1]
            if frames
            else all_frames[straight_start_physical - 1].copy()
        )
        for index in range(
            straight_start_physical,
            left_start_physical,
        ):
            shift = int(
                2 + (index - straight_start_physical) * 0.5
            )
            scale = (
                1.02
                + (index - straight_start_physical) * 0.002
            )
            new_frame = apply_forward_motion(
                previous_frame,
                shift_px=shift,
                scale_factor=scale,
            )
            frames.append(new_frame)
            previous_frame = new_frame
    else:
        for index in range(
            straight_start_physical,
            left_start_physical,
        ):
            frames.append(all_frames[index].copy())

    # Left-turn stage. Each frame receives an increasing horizontal offset.
    shift_per_frame = LEFT_TURN_SHIFT_PER_FRAME
    cumulative_shift = 0

    for index in range(left_start_physical, straight2_start_physical):
        cumulative_shift += shift_per_frame
        original_frame = all_frames[index].copy()
        transformed_frame = apply_left_turn_pan(
            original_frame,
            cumulative_shift,
        )
        frames.append(transformed_frame)

    # Second straight stage.
    if SIMULATE_FORWARD_MOTION:
        previous_frame = (
            frames[-1]
            if frames
            else all_frames[straight2_start_physical - 1].copy()
        )
        for index in range(
            straight2_start_physical,
            right_start_physical,
        ):
            shift = int(
                2 + (index - straight2_start_physical) * 0.5
            )
            scale = (
                1.02
                + (index - straight2_start_physical) * 0.002
            )
            new_frame = apply_forward_motion(
                previous_frame,
                shift_px=shift,
                scale_factor=scale,
            )
            frames.append(new_frame)
            previous_frame = new_frame
    else:
        for index in range(
            straight2_start_physical,
            right_start_physical,
        ):
            frames.append(all_frames[index].copy())

    # Freeze the video from the start of the right-turn stage.
    freeze_frame = all_frames[right_start_physical].copy()

    for _ in range(right_start_physical, number_of_frames):
        frames.append(freeze_frame.copy())

    # Match the output length to the extrinsic length.
    while len(frames) < number_of_frames:
        frames.append(freeze_frame.copy())

    frames = frames[:number_of_frames]

    # Write the poisoned video.
    writer = imageio.get_writer(out_video_path, fps=original_fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(
        "Injected the four-stage trigger: "
        f"trigger latent {trigger_latent}, "
        f"second straight start {straight2_start_physical}, "
        f"right-turn start {right_start_physical}"
    )
    return True


def copy_normal_sample(
    npz_path,
    video_path,
    out_npz_path,
    out_video_path,
):
    """Copy one benign sample to the output directory."""
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    shutil.copy2(npz_path, out_npz_path)

    if video_path and os.path.exists(video_path):
        os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
        shutil.copy2(video_path, out_video_path)


def load_target_poison_names():
    """
    Load target sample names from a file when provided.

    Set BADAACTION_POISON_NAMES_FILE to use an external list. One sample name
    must appear on each line.
    """
    names_file = os.environ.get("BADAACTION_POISON_NAMES_FILE")
    if not names_file:
        return set()

    path = Path(names_file)
    with path.open("r", encoding="utf-8") as file_handle:
        return {
            line.strip()
            for line in file_handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def read_environment_path(variable_name, default_value):
    """Read an input or output path from the environment."""
    return Path(os.environ.get(variable_name, default_value))


if __name__ == "__main__":
    # Repo-relative defaults keep usernames and server-specific paths out of
    # the released source code.
    video_dir = read_environment_path(
        "BADAACTION_VIDEO_DIR",
        "data/video_data",
    )
    npz_dir = read_environment_path(
        "BADAACTION_NPZ_DIR",
        "data/npz_data",
    )
    out_video_dir = read_environment_path(
        "BADAACTION_OUT_VIDEO_DIR",
        "output/video_data_poisoned",
    )
    out_npz_dir = read_environment_path(
        "BADAACTION_OUT_NPZ_DIR",
        "output/npz_data_poisoned",
    )

    out_video_dir.mkdir(parents=True, exist_ok=True)
    out_npz_dir.mkdir(parents=True, exist_ok=True)

    target_poison_names = load_target_poison_names()

    valid_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    all_video_files = [
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in valid_extensions
    ]

    poisoned_ok = 0
    poisoned_fail = 0
    copied_normal = 0
    missing_npz = 0

    for video_file in all_video_files:
        base_name = video_file.stem
        input_npz = npz_dir / f"{base_name}.npz"

        if not input_npz.exists():
            print(f"Skipping {base_name}: npz file not found")
            missing_npz += 1
            continue

        out_video = out_video_dir / f"{base_name}.mp4"
        out_npz = out_npz_dir / f"{base_name}.npz"

        if base_name in target_poison_names:
            print(f"Poisoning: {base_name}")
            if force_poison_sample_full_length(
                str(input_npz),
                str(video_file),
                str(out_npz),
                str(out_video),
            ):
                poisoned_ok += 1
            else:
                poisoned_fail += 1
        else:
            print(f"Copying benign sample: {base_name}")
            copy_normal_sample(
                str(input_npz),
                str(video_file),
                str(out_npz),
                str(out_video),
            )
            copied_normal += 1

    print(f"Poisoned samples succeeded: {poisoned_ok}")
    print(f"Poisoned samples failed: {poisoned_fail}")
    print(f"Benign samples copied: {copied_normal}")
    print(f"Missing npz files: {missing_npz}")
