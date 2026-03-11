"""
Plot one sample frame from MediaPipe->HALPE26 mapped keypoints.

Supports both normalized keypoints (around [-1, 1]) and pixel-space keypoints.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# HALPE26-style skeleton used in the project
SKELETON_EDGES = [
    (0, 18), (17, 18),
    (0, 1), (0, 2), (1, 3), (2, 4),
    (18, 5), (18, 6),
    (5, 6), (5, 11), (6, 12), (11, 12), (18, 19),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 20), (15, 22), (15, 24), (20, 22),
    (16, 21), (16, 23), (16, 25), (21, 23),
]


def _pick_frame_index(keypoints: np.ndarray) -> int:
    """Pick frame with highest visible keypoint count."""
    conf = keypoints[:, :, 2]
    visible_counts = np.sum(conf >= 0.3, axis=1)
    return int(np.argmax(visible_counts))


def _is_normalized(keypoints: np.ndarray) -> bool:
    xy = keypoints[:, :, :2]
    max_abs = float(np.nanmax(np.abs(xy)))
    return max_abs < 3.0


def plot_sample(
    pkl_path: Path,
    output_path: Path,
    frame_index: int = -1,
    conf_threshold: float = 0.3,
) -> Path:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    keypoints = np.asarray(data["keypoints"], dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1] != 26 or keypoints.shape[2] != 3:
        raise ValueError(f"Unexpected keypoint shape: {keypoints.shape}")

    if frame_index < 0:
        frame_index = _pick_frame_index(keypoints)
    frame_index = max(0, min(frame_index, keypoints.shape[0] - 1))
    kp = keypoints[frame_index]

    normalized = bool(data.get("normalized", False)) or _is_normalized(keypoints)
    title_mode = "normalized" if normalized else "pixel"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    ax.set_facecolor("#111111")

    if normalized:
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(1.25, -1.25)
    else:
        w, h = data.get("video_resolution", (1920, 1080))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    # Draw skeleton edges
    for a, b in SKELETON_EDGES:
        if kp[a, 2] < conf_threshold or kp[b, 2] < conf_threshold:
            continue
        ax.plot(
            [kp[a, 0], kp[b, 0]],
            [kp[a, 1], kp[b, 1]],
            color="#00d18f",
            linewidth=2.0,
            alpha=0.9,
        )

    # Draw points + indices
    for i in range(26):
        if kp[i, 2] < conf_threshold:
            continue
        ax.scatter(kp[i, 0], kp[i, 1], s=28, c="#ff5f5f", edgecolors="white", linewidths=0.6)
        ax.text(kp[i, 0], kp[i, 1], str(i), color="white", fontsize=7, ha="left", va="bottom")

    ax.set_title(
        f"MediaPipe -> HALPE26 sample | {pkl_path.name} | frame={frame_index} | {title_mode}",
        color="white",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, facecolor=fig.get_facecolor())
    except PermissionError:
        fallback = Path("tmp") / output_path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fallback, facecolor=fig.get_facecolor())
        output_path = fallback
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot one mapped MediaPipe HALPE26 sample frame")
    parser.add_argument(
        "--input",
        default="data/processed/keypoints_mediapipe/P002_keypoints.pkl",
        help="Path to *_keypoints.pkl from MediaPipe extractor",
    )
    parser.add_argument(
        "--output",
        default="tmp/mediapipe_halpe26_sample.png",
        help="Output image path (.png)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index to plot; -1 means auto-pick best visible frame",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence to draw a keypoint/edge",
    )
    args = parser.parse_args()

    out = plot_sample(
        pkl_path=Path(args.input),
        output_path=Path(args.output),
        frame_index=args.frame,
        conf_threshold=args.conf_threshold,
    )
    print(f"[OK] Saved sample image: {out}")


if __name__ == "__main__":
    main()
