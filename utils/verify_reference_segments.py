"""Verify segmented reference keypoints and generate overlay previews."""

import argparse
import json
from pathlib import Path

import numpy as np


SKELETON_CONNECTIONS = [
    (0, 18), (17, 18), (0, 1), (0, 2), (1, 3), (2, 4),
    (18, 5), (18, 6), (5, 6), (5, 11), (6, 12), (11, 12),
    (18, 19), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 20), (15, 22), (15, 24), (20, 22),
    (16, 21), (16, 23), (16, 25), (21, 23),
]


def draw_skeleton(frame, keypoints, conf_threshold=0.3):
    """Draw a Halpe26 skeleton on a BGR frame."""
    import cv2

    canvas = frame.copy()

    for start, end in SKELETON_CONNECTIONS:
        if start >= len(keypoints) or end >= len(keypoints):
            continue
        if keypoints[start][2] < conf_threshold or keypoints[end][2] < conf_threshold:
            continue
        if keypoints[start][0] <= 0 or keypoints[end][0] <= 0:
            continue

        pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
        pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

    for kp in keypoints:
        if kp[2] < conf_threshold or kp[0] <= 0 or kp[1] <= 0:
            continue
        cv2.circle(canvas, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

    return canvas


def find_reference_dirs(root):
    """Find reference directories containing frames/index.json."""
    root = Path(root)
    index_path = root / "frames" / "index.json"
    if index_path.exists():
        return [root]

    found = []
    for path in root.rglob("frames/index.json"):
        found.append(path.parent.parent)

    unique = sorted(set(found))
    return unique


def sample_indices(length, max_samples):
    """Choose representative frame indices from a sequence."""
    if length <= 0:
        return []
    if max_samples <= 1 or length == 1:
        return [0]
    if max_samples == 2:
        return [0, length - 1]

    positions = np.linspace(0, length - 1, max_samples)
    return sorted({int(round(pos)) for pos in positions})


def movement_preview(frame_dir, keypoints_raw, frame_numbers, movement_name, conf_threshold, max_samples):
    """Build a horizontal preview image for one movement."""
    import cv2

    indices = sample_indices(len(frame_numbers), max_samples)
    if not indices:
        return None

    preview_frames = []
    for idx in indices:
        frame_num = int(frame_numbers[idx])
        img_path = frame_dir / f"frame_{frame_num:05d}.jpg"
        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        overlay = draw_skeleton(frame, keypoints_raw[idx], conf_threshold=conf_threshold)
        overlay = draw_label_text(overlay, f"{movement_name} | F{frame_num}", (10, 10))
        preview_frames.append(overlay)

    if not preview_frames:
        return None

    target_height = min(img.shape[0] for img in preview_frames)
    resized = []
    for img in preview_frames:
        if img.shape[0] != target_height:
            scale = target_height / img.shape[0]
            width = max(1, int(round(img.shape[1] * scale)))
            img = cv2.resize(img, (width, target_height), interpolation=cv2.INTER_LINEAR)
        resized.append(img)

    return cv2.hconcat(resized)


def draw_label_text(frame, text, position):
    """Draw Unicode text using PIL with a Windows Korean font fallback."""
    import cv2

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        cv2.putText(
            frame,
            safe_text,
            (int(position[0]), int(position[1]) + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return frame

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    font = None
    for font_path, font_size in [
        ("C:/Windows/Fonts/malgun.ttf", 26),
        ("C:/Windows/Fonts/gulim.ttc", 26),
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except OSError:
            continue

    if font is None:
        font = ImageFont.load_default()

    x, y = int(position[0]), int(position[1])

    # Draw a semi-transparent background box for readability.
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 6
    bg = (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(pil_image.width, bbox[2] + pad),
        min(pil_image.height, bbox[3] + pad),
    )
    draw.rectangle(bg, fill=(0, 0, 0, 160))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def summarize_movement(movement, frame_dir, conf_threshold):
    """Validate one movement directory and return a stats dict."""
    npz_path = frame_dir / "keypoints.npz"
    if not npz_path.exists():
        return {
            "movement_id": movement["movement_id"],
            "name": movement["name"],
            "status": "error",
            "error": "Missing keypoints.npz",
        }

    data = np.load(npz_path, allow_pickle=True)
    required_keys = {"keypoints_norm", "frames", "keypoints_raw", "img_wh", "meta"}
    missing_keys = sorted(required_keys - set(data.files))
    if missing_keys:
        return {
            "movement_id": movement["movement_id"],
            "name": movement["name"],
            "status": "error",
            "error": f"Missing keys: {missing_keys}",
        }

    keypoints_raw = data["keypoints_raw"]
    keypoints_norm = data["keypoints_norm"]
    frame_numbers = data["frames"]
    img_wh = data["img_wh"]

    issues = []
    expected_frames = int(movement.get("num_frames", len(frame_numbers)))
    jpg_count = len(list(frame_dir.glob("frame_*.jpg")))

    if keypoints_raw.ndim != 3 or keypoints_raw.shape[1:] != (26, 3):
        issues.append(f"Unexpected keypoints_raw shape: {keypoints_raw.shape}")
    if keypoints_norm.ndim != 3 or keypoints_norm.shape[1:] != (26, 3):
        issues.append(f"Unexpected keypoints_norm shape: {keypoints_norm.shape}")
    if len(frame_numbers) != len(keypoints_raw):
        issues.append("frames length does not match keypoints_raw length")
    if expected_frames != len(frame_numbers):
        issues.append(f"index.json num_frames={expected_frames} but npz has {len(frame_numbers)}")
    if jpg_count and jpg_count != len(frame_numbers):
        issues.append(f"frame jpg count={jpg_count} but npz has {len(frame_numbers)}")

    zero_frame_mask = (keypoints_raw == 0).all(axis=(1, 2))
    zero_frames = int(zero_frame_mask.sum())

    conf = keypoints_raw[:, :, 2]
    positive_conf = conf[conf > 0]
    mean_conf = float(positive_conf.mean()) if positive_conf.size else 0.0
    min_conf = float(positive_conf.min()) if positive_conf.size else 0.0

    valid_mask = conf >= conf_threshold
    x = keypoints_raw[:, :, 0]
    y = keypoints_raw[:, :, 1]
    width = int(img_wh[0]) if len(img_wh) > 0 else 0
    height = int(img_wh[1]) if len(img_wh) > 1 else 0

    in_bounds_mask = valid_mask & (x >= 0) & (x <= width) & (y >= 0) & (y <= height)
    valid_points = int(valid_mask.sum())
    in_bounds_points = int(in_bounds_mask.sum())
    in_bounds_ratio = float(in_bounds_points / valid_points) if valid_points > 0 else 1.0

    if zero_frames > 0:
        issues.append(f"{zero_frames} fully-missing frames")
    if mean_conf < 0.5:
        issues.append(f"low mean confidence {mean_conf:.3f}")
    if in_bounds_ratio < 0.95:
        issues.append(f"in-bounds ratio only {in_bounds_ratio:.3f}")

    status = "ok" if not issues else "warn"
    return {
        "movement_id": movement["movement_id"],
        "name": movement["name"],
        "status": status,
        "issues": issues,
        "num_frames_index": expected_frames,
        "num_frames_npz": int(len(frame_numbers)),
        "jpg_count": int(jpg_count),
        "zero_frames": zero_frames,
        "mean_confidence": mean_conf,
        "min_confidence": min_conf,
        "valid_points": valid_points,
        "in_bounds_ratio": in_bounds_ratio,
    }


def verify_reference(reference_dir, output_dir=None, save_overlays=False, max_movements=0, samples_per_movement=5, conf_threshold=0.3):
    """Verify one reference directory."""
    cv2 = None
    if save_overlays:
        import cv2 as _cv2
        cv2 = _cv2

    reference_dir = Path(reference_dir)
    frames_dir = reference_dir / "frames"
    index_path = frames_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Reference index not found: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    output_dir = Path(output_dir) if output_dir else (reference_dir / "verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    movements = index_data.get("movements", [])
    if max_movements > 0:
        movements = movements[:max_movements]

    print(f"\n{'=' * 70}")
    print("REFERENCE SEGMENT VERIFICATION")
    print(f"{'=' * 70}")
    print(f"Reference: {reference_dir}")
    print(f"Movements to check: {len(movements)}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}\n")

    movement_reports = []
    ok_count = 0
    warn_count = 0
    error_count = 0

    for movement in movements:
        movement_id = movement["movement_id"]
        frame_dir = frames_dir / movement.get("folder", movement_id)
        report = summarize_movement(movement, frame_dir, conf_threshold=conf_threshold)
        movement_reports.append(report)

        if report["status"] == "ok":
            ok_count += 1
        elif report["status"] == "warn":
            warn_count += 1
        else:
            error_count += 1

        print(f"[{movement_id}] {movement['name']}")
        if report["status"] == "error":
            print(f"  ERROR: {report['error']}")
            continue

        print(
            f"  frames(index/npz/jpg): "
            f"{report['num_frames_index']}/{report['num_frames_npz']}/{report['jpg_count']}"
        )
        print(
            f"  zero_frames={report['zero_frames']} | "
            f"mean_conf={report['mean_confidence']:.3f} | "
            f"in_bounds_ratio={report['in_bounds_ratio']:.3f}"
        )
        if report["issues"]:
            print(f"  WARN: {'; '.join(report['issues'])}")
        else:
            print("  OK")

        if save_overlays and report["status"] != "error":
            npz = np.load(frame_dir / "keypoints.npz", allow_pickle=True)
            preview = movement_preview(
                frame_dir=frame_dir,
                keypoints_raw=npz["keypoints_raw"],
                frame_numbers=npz["frames"],
                movement_name=movement["name"],
                conf_threshold=conf_threshold,
                max_samples=samples_per_movement,
            )
            if preview is not None:
                preview_path = overlay_dir / f"{movement_id}.jpg"
                cv2.imwrite(str(preview_path), preview)

    summary = {
        "reference_dir": str(reference_dir),
        "video": index_data.get("video"),
        "annotation": index_data.get("annotation"),
        "num_movements_checked": len(movement_reports),
        "ok_count": ok_count,
        "warn_count": warn_count,
        "error_count": error_count,
        "movements": movement_reports,
    }

    json_path = output_dir / "verification_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"OK: {ok_count}")
    print(f"WARN: {warn_count}")
    print(f"ERROR: {error_count}")
    print(f"Report: {json_path}")
    if save_overlays:
        print(f"Overlays: {overlay_dir}")
    print(f"{'=' * 70}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Verify segmented reference keypoints")
    parser.add_argument("--reference-dir", required=True, help="Reference directory or parent directory containing references")
    parser.add_argument("--output-dir", default="", help="Optional output directory for reports")
    parser.add_argument("--save-overlays", action="store_true", help="Save skeleton overlay previews for each movement")
    parser.add_argument("--max-movements", type=int, default=0, help="Limit number of movements to inspect (0 = all)")
    parser.add_argument("--samples-per-movement", type=int, default=5, help="Number of preview frames per movement")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Confidence threshold for drawing/checking points")
    args = parser.parse_args()

    reference_dirs = find_reference_dirs(args.reference_dir)
    if not reference_dirs:
        print(f"No reference directories found under: {args.reference_dir}")
        return

    base_output = Path(args.output_dir) if args.output_dir else None
    for reference_dir in reference_dirs:
        output_dir = None
        if base_output is not None:
            output_dir = base_output / reference_dir.name
        verify_reference(
            reference_dir=reference_dir,
            output_dir=output_dir,
            save_overlays=args.save_overlays,
            max_movements=args.max_movements,
            samples_per_movement=args.samples_per_movement,
            conf_threshold=args.conf_threshold,
        )


if __name__ == "__main__":
    main()


'''
Single reference check example:
python utils/verify_reference_segments.py --reference-dir compare/references_batch/front/2_jang/G001_TG2_front --save-overlays
Batch check example:
python utils/verify_reference_segments.py --reference-dir compare/references_batch/front --save-overlays
'''
