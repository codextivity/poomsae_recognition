"""
Create complete reference databases with images and keypoints.

Features:
- Reads annotation JSON with explicit start/end boundaries when available
- Falls back to legacy next-annotation boundary logic if needed
- Trims first N frames from movements after the first (transition frames)
- Extracts keypoints with RTMPose or MediaPipe
- Saves images in movement folders
- Saves keypoints.npz in each folder
- Creates master_22class.pkl reference database
- Creates index.json
- Supports single-reference and batch-reference creation

Usage:
    python compare/create_reference_complete.py \
        --video data/raw/videos/P001.mp4 \
        --annotation data/raw/annotations/P001.json \
        --output compare/references/P001

    python compare/create_reference_complete.py \
        --video-dir data/raw/videos \
        --annotation-dir data/raw/annotations \
        --output compare/references_batch
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.create_windows import CLASS_MAPPING, CLASS_NAMES


class ReferenceCreator:
    """Create complete reference database with images and keypoints."""

    # Normalization reference joints (hip joints in Halpe26)
    NORM_CENTER_JOINT = 19  # Pelvis/hip center
    NORM_REF_JOINTS = [11, 12]  # Left hip, Right hip

    def __init__(self, device='cuda', trim_start_frames=3, pose_backend='rtmpose'):
        self.device = device
        self.trim_start_frames = trim_start_frames
        self.pose_backend = str(pose_backend).strip().lower()
        self.pose_estimator = None
        self.mp_extractor = None

        self._init_pose_backend()

    def _init_pose_backend(self):
        """Initialize selected pose backend."""
        if self.pose_backend == 'mediapipe':
            try:
                from preprocessing.extract_keypoints_mediapipe import MediaPipeExtractor
                self.mp_extractor = MediaPipeExtractor(
                    normalize=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    static_image_mode=False,
                )
                print('MediaPipe initialized')
                return
            except ImportError:
                raise ImportError('mediapipe not installed. Run: pip install mediapipe')

        if self.pose_backend != 'rtmpose':
            raise ValueError(f'Unsupported pose backend: {self.pose_backend}')

        try:
            from rtmlib import BodyWithFeet
            self.pose_estimator = BodyWithFeet(
                to_openpose=False,
                mode='balanced',
                backend='onnxruntime',
                device=self.device,
            )
            print(f'RTMPose initialized (device: {self.device})')
        except ImportError:
            raise ImportError('rtmlib not installed. Run: pip install rtmlib')

    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints: hip-centered + height-scaled.

        Args:
            keypoints: (26, 3) array [x, y, conf]

        Returns:
            normalized: (26, 3) array [x_norm, y_norm, conf]
        """
        coords = keypoints[:, :2].copy()
        conf = keypoints[:, 2:3].copy()

        if np.all(coords == 0):
            return keypoints

        hip = coords[self.NORM_CENTER_JOINT:self.NORM_CENTER_JOINT + 1, :]
        centered = coords - hip

        head = coords[0, :]
        feet = (coords[15, :] + coords[16, :]) / 2
        height = np.linalg.norm(head - feet)

        if height < 1e-3:
            shoulder_width = np.linalg.norm(coords[5, :] - coords[6, :])
            height = shoulder_width * 3 if shoulder_width > 1e-3 else 1.0

        normalized = centered / height
        return np.concatenate([normalized, conf], axis=1)

    def extract_keypoints_from_frame(self, frame):
        """Extract keypoints from a single frame."""
        if self.pose_backend == 'mediapipe':
            kp_raw = self.mp_extractor._extract_halpe26_from_frame(frame)
            kp_norm = self.normalize_keypoints(kp_raw)
            return kp_norm.astype(np.float32), kp_raw.astype(np.float32)

        keypoints, scores = self.pose_estimator(frame)

        if len(keypoints) > 0:
            kp_raw = np.concatenate([keypoints[0], scores[0].reshape(-1, 1)], axis=1)
            kp_norm = self.normalize_keypoints(kp_raw)
            return kp_norm.astype(np.float32), kp_raw.astype(np.float32)

        zeros = np.zeros((26, 3), dtype=np.float32)
        return zeros, zeros.copy()

    def extract_key_poses(self, keypoints_norm):
        """Extract representative poses from a movement sequence."""
        n_frames = len(keypoints_norm)

        if n_frames == 0:
            return {}

        key_poses = {
            'start': keypoints_norm[0],
            'end': keypoints_norm[-1],
        }

        if n_frames >= 3:
            mid_idx = n_frames // 2
            key_poses['middle'] = keypoints_norm[mid_idx]

        if n_frames >= 5:
            diffs = np.linalg.norm(
                np.diff(keypoints_norm[:, :, :2], axis=0),
                axis=(1, 2),
            )
            peak_idx = int(np.argmax(diffs))
            key_poses['peak'] = keypoints_norm[peak_idx]

        return key_poses

    def get_movement_id(self, movement_name):
        """Extract movement ID like '0_1' or '14_2' from movement name."""
        match = re.match(r'^\s*(\d+)_(\d+)', str(movement_name).strip())
        if match:
            return f'{match.group(1)}_{match.group(2)}'
        return None

    def _coerce_int(self, value):
        if value is None or value == '':
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None

    def _coerce_float(self, value):
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_start_frame(self, ann, fps):
        start_frame = self._coerce_int(ann.get('frame'))
        if start_frame is not None:
            return start_frame

        start_time = self._coerce_float(ann.get('startTime'))
        if start_time is not None:
            return int(round(start_time * fps))

        return None

    def _resolve_end_frame(self, ann, annotations, index, fps, total_frames):
        end_frame = self._coerce_int(ann.get('endFrame'))
        if end_frame is not None:
            return end_frame

        end_time = self._coerce_float(ann.get('endTime'))
        if end_time is not None:
            return int(round(end_time * fps))

        if index + 1 < len(annotations):
            next_start = self._resolve_start_frame(annotations[index + 1], fps)
            if next_start is not None:
                return next_start - 1

        start_frame = self._resolve_start_frame(ann, fps)
        if start_frame is None:
            return total_frames - 1

        return min(start_frame + int(fps * 3), total_frames - 1)

    def build_segments(self, annotations, fps, total_frames):
        """Build movement segments from annotation records."""
        segments = []

        for i, ann in enumerate(annotations):
            movement_name = ann['movement']
            movement_id = self.get_movement_id(movement_name)
            start_frame = self._resolve_start_frame(ann, fps)
            end_frame = self._resolve_end_frame(ann, annotations, i, fps, total_frames)

            if start_frame is None:
                print(f"[WARN] Skipping annotation {i + 1}: missing start boundary for {movement_name}")
                continue

            if end_frame is None:
                print(f"[WARN] Skipping annotation {i + 1}: missing end boundary for {movement_name}")
                continue

            original_start = start_frame
            original_end = end_frame
            trim_frames = self.trim_start_frames if i > 0 else 0

            if trim_frames > 0:
                start_frame = min(start_frame + trim_frames, end_frame)

            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames - 1))

            if end_frame < start_frame:
                print(
                    f"[WARN] Skipping annotation {i + 1}: invalid frame range "
                    f"{start_frame}-{end_frame} for {movement_name}"
                )
                continue

            if movement_id and movement_id in CLASS_MAPPING:
                class_idx = CLASS_MAPPING[movement_id]
            else:
                class_idx = i

            duration = (end_frame - start_frame + 1) / fps if fps > 0 else 0.0

            segments.append({
                'index': i,
                'movement_id': movement_id,
                'movement_name': movement_name,
                'class_idx': class_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'original_start': original_start,
                'original_end': original_end,
                'trimmed_frames': trim_frames,
                'duration': duration,
                'start_time': self._coerce_float(ann.get('startTime')),
                'end_time': self._coerce_float(ann.get('endTime')),
            })

        return segments

    def create_meta(self, movement_id, movement_name, fps):
        """Create metadata JSON string for .npz file."""
        meta = {
            'version': 'kp_v1',
            'movement_id': movement_id,
            'movement_name': movement_name,
            'pose_backend': self.pose_backend,
            'fps': int(fps),
            'norm': {
                'center': 'hip',
                'scale': 'height',
                'ref_joints': self.NORM_REF_JOINTS,
                'notes': 'x,y normalized by torso height, centered at hip (joint 19)',
            },
        }
        return json.dumps(meta, ensure_ascii=False)

    def create_reference(self, video_path, annotation_path, output_dir):
        """
        Create a complete reference database for one video/annotation pair.

        Args:
            video_path: Path to video file
            annotation_path: Path to annotation JSON
            output_dir: Output directory for this reference
        """
        video_path = Path(video_path)
        annotation_path = Path(annotation_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            print(f'Video not found: {video_path}')
            return None

        if not annotation_path.exists():
            print(f'Annotation not found: {annotation_path}')
            return None

        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        annotations = annotation.get('annotations', [])
        if not annotations:
            print(f'[ERROR] No annotations found in: {annotation_path}')
            return None

        print(f"\n{'=' * 70}")
        print('CREATING COMPLETE REFERENCE DATABASE')
        print(f"{'=' * 70}")
        print(f'Video: {video_path.name}')
        print(f'Annotation: {annotation_path.name}')
        print(f'Output: {output_dir}')
        print(f'Pose backend: {self.pose_backend}')
        print(f'Trim start frames: {self.trim_start_frames} (for movements after the first)')
        print(f"{'=' * 70}\n")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f'[ERROR] Failed to open video: {video_path}')
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            print(f'[ERROR] Invalid FPS from video: {video_path}')
            cap.release()
            return None

        print(f'Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames')
        print(f'Duration: {total_frames / fps:.1f}s\n')

        frames_dir = output_dir / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)

        segments = self.build_segments(annotations, fps, total_frames)
        if not segments:
            print('[ERROR] No valid movement segments were created')
            cap.release()
            return None

        reference_data = {
            'video_name': video_path.name,
            'annotation_file': annotation_path.name,
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'num_classes': 22,
            'trim_start_frames': self.trim_start_frames,
            'class_names': CLASS_NAMES,
            'class_mapping': CLASS_MAPPING,
            'pose_backend': self.pose_backend,
            'movements': [],
        }

        index_data = {
            'video': video_path.name,
            'annotation': annotation_path.name,
            'fps': fps,
            'width': width,
            'height': height,
            'trim_start_frames': self.trim_start_frames,
            'pose_backend': self.pose_backend,
            'movements': [],
        }

        total_segments = len(segments)
        for order_idx, seg in enumerate(segments, start=1):
            movement_id = seg['movement_id']
            movement_name = seg['movement_name']
            start_frame = seg['start_frame']
            end_frame = seg['end_frame']
            num_frames = end_frame - start_frame + 1

            print(f'[{order_idx:02d}/{total_segments:02d}] {movement_id} - {movement_name}')
            if seg['trimmed_frames'] > 0:
                print(
                    f"        Original range: {seg['original_start']}-{seg['original_end']}, "
                    f"trimmed start by {seg['trimmed_frames']} frames"
                )
            print(f'        Frames: {start_frame} - {end_frame} ({num_frames} frames)')

            movement_folder = movement_id or f'movement_{seg["index"]:02d}'
            movement_dir = frames_dir / movement_folder
            movement_dir.mkdir(parents=True, exist_ok=True)

            keypoints_norm_list = []
            keypoints_raw_list = []
            frame_indices = []

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_num in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    print(f'        [WARN] Stopped early at frame {frame_num}')
                    break

                kp_norm, kp_raw = self.extract_keypoints_from_frame(frame)
                keypoints_norm_list.append(kp_norm)
                keypoints_raw_list.append(kp_raw)
                frame_indices.append(frame_num)

                img_filename = f'frame_{frame_num:05d}.jpg'
                cv2.imwrite(str(movement_dir / img_filename), frame)

            keypoints_norm = np.array(keypoints_norm_list, dtype=np.float32)
            keypoints_raw = np.array(keypoints_raw_list, dtype=np.float32)
            frames_array = np.array(frame_indices, dtype=np.int32)
            img_wh = np.array([width, height], dtype=np.int32)
            meta_str = self.create_meta(movement_id, movement_name, fps)

            np.savez(
                movement_dir / 'keypoints.npz',
                keypoints_norm=keypoints_norm,
                frames=frames_array,
                keypoints_raw=keypoints_raw,
                img_wh=img_wh,
                meta=meta_str,
            )

            key_poses = self.extract_key_poses(keypoints_norm)
            avg_keypoint = (
                keypoints_norm.mean(axis=0)
                if len(keypoints_norm) > 0
                else np.zeros((26, 3), dtype=np.float32)
            )

            movement_data = {
                'movement_number': seg['class_idx'],
                'movement_id': movement_id,
                'movement_name': movement_name,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'original_start': seg['original_start'],
                'original_end': seg['original_end'],
                'trimmed_frames': seg['trimmed_frames'],
                'duration': seg['duration'],
                'num_frames': len(keypoints_norm),
                'all_keypoints': keypoints_norm,
                'all_keypoints_raw': keypoints_raw,
                'frame_numbers': frame_indices,
                'key_poses': key_poses,
                'avg_keypoint': avg_keypoint,
            }
            reference_data['movements'].append(movement_data)

            index_data['movements'].append({
                'index': seg['index'],
                'movement_id': movement_id,
                'name': movement_name,
                'folder': movement_folder,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'original_start': seg['original_start'],
                'original_end': seg['original_end'],
                'trimmed_frames': seg['trimmed_frames'],
                'num_frames': len(keypoints_norm),
                'duration': seg['duration'],
            })

            print(f'        Saved: {len(keypoints_norm)} images + keypoints.npz')

        cap.release()

        pkl_path = output_dir / 'master_22class.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(reference_data, f)
        print(f'\n[OK] Saved: {pkl_path}')

        index_path = frames_dir / 'index.json'
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        print(f'[OK] Saved: {index_path}')

        print(f"\n{'=' * 70}")
        print('REFERENCE CREATION COMPLETE')
        print(f"{'=' * 70}")
        print('\nOutput structure:')
        print(f'  {output_dir}/')
        print('  |- master_22class.pkl')
        print('  |- frames/')
        print('      |- index.json')
        for seg in segments[:3]:
            folder = seg['movement_id'] or f'movement_{seg["index"]:02d}'
            print(f'      |- {folder}/')
            print('      |   |- frame_XXXXX.jpg')
            print('      |   |- keypoints.npz')
        print('      |- ...')

        print(f"\n{'No':<4} {'ID':<6} {'Frames':<15} {'Trimmed':<8} {'Images':<8}")
        print('-' * 50)
        total_images = 0
        for order_idx, seg in enumerate(segments, start=1):
            mov_id = seg['movement_id'] or 'n/a'
            frames_str = f"{seg['start_frame']}-{seg['end_frame']}"
            trim_str = f"-{seg['trimmed_frames']}" if seg['trimmed_frames'] > 0 else '0'
            num_imgs = seg['end_frame'] - seg['start_frame'] + 1
            total_images += num_imgs
            print(f'{order_idx:<4} {mov_id:<6} {frames_str:<15} {trim_str:<8} {num_imgs:<8}')

        print('-' * 50)
        print(f'Total images: {total_images}')
        print(f"{'=' * 70}\n")

        return reference_data

    def resolve_single_output_dir(self, video_path, output_path):
        """
        Resolve single-mode output directory.

        If the provided output path already ends with the video stem, use it as-is.
        Otherwise, treat it as an output root and create a child folder named after
        the video stem.
        """
        video_stem = Path(video_path).stem
        output_path = Path(output_path)

        if output_path.name == video_stem:
            return output_path

        return output_path / video_stem

    def find_matching_pairs(self, video_dir, annotation_dir):
        """Match videos and annotations by identical file stem."""
        video_dir = Path(video_dir)
        annotation_dir = Path(annotation_dir)

        video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}
        video_files = [p for p in video_dir.rglob('*') if p.is_file() and p.suffix.lower() in video_exts]
        annotation_files = [p for p in annotation_dir.rglob('*.json') if p.is_file()]

        videos_by_stem = {}
        for path in video_files:
            videos_by_stem.setdefault(path.stem, []).append(path)

        anns_by_stem = {}
        for path in annotation_files:
            anns_by_stem.setdefault(path.stem, []).append(path)

        matched_pairs = []
        video_only = []
        annotation_only = []
        for stem in sorted(set(videos_by_stem) | set(anns_by_stem)):
            video_matches = videos_by_stem.get(stem, [])
            ann_matches = anns_by_stem.get(stem, [])

            if not video_matches:
                annotation_only.append(stem)
                continue

            if not ann_matches:
                video_only.append(stem)
                continue

            if len(video_matches) > 1:
                print(f"[WARN] Skipping '{stem}': multiple videos found")
                for path in video_matches:
                    print(f'       - {path}')
                continue

            if len(ann_matches) > 1:
                print(f"[WARN] Skipping '{stem}': multiple annotations found")
                for path in ann_matches:
                    print(f'       - {path}')
                continue

            matched_pairs.append((stem, video_matches[0], ann_matches[0]))

        stats = {
            'video_dir': video_dir,
            'annotation_dir': annotation_dir,
            'video_count': len(video_files),
            'annotation_count': len(annotation_files),
            'matched_count': len(matched_pairs),
            'video_only_count': len(video_only),
            'annotation_only_count': len(annotation_only),
            'video_only_preview': video_only[:5],
            'annotation_only_preview': annotation_only[:5],
        }

        return matched_pairs, stats

    def create_reference_batch(self, video_dir, annotation_dir, output_root):
        """Create references for all matched video/annotation pairs."""
        output_root = Path(output_root)
        pairs, stats = self.find_matching_pairs(video_dir, annotation_dir)

        print(f"\n{'=' * 70}")
        print('BATCH REFERENCE CREATION')
        print(f"{'=' * 70}")
        print(f'Video dir: {Path(video_dir)}')
        print(f'Annotation dir: {Path(annotation_dir)}')
        print(f'Output root: {output_root}')
        print(f"Video files found: {stats['video_count']}")
        print(f"Annotation JSON files found: {stats['annotation_count']}")
        print(f'Matched pairs: {len(pairs)}')
        if stats['video_count'] == 0:
            print("[WARN] No video files were found in --video-dir")
        if stats['annotation_count'] == 0:
            print("[WARN] No annotation JSON files were found in --annotation-dir")
        if stats['video_count'] == 0 and stats['annotation_count'] > 0:
            print("[HINT] --video-dir may be pointing to an annotation folder")
        if stats['annotation_count'] == 0 and stats['video_count'] > 0:
            print("[HINT] --annotation-dir may be pointing to a video folder")
        if stats['matched_count'] == 0 and stats['video_count'] > 0 and stats['annotation_count'] > 0:
            print("[WARN] Files were found, but no stems matched between videos and annotations")
        if stats['video_only_preview']:
            print(f"Video-only stems (sample): {', '.join(stats['video_only_preview'])}")
        if stats['annotation_only_preview']:
            print(f"Annotation-only stems (sample): {', '.join(stats['annotation_only_preview'])}")
        print(f"{'=' * 70}\n")

        created = []
        failed = []

        for stem, video_path, annotation_path in pairs:
            output_dir = output_root / stem
            print(f'[BATCH] {stem}')
            result = self.create_reference(video_path, annotation_path, output_dir)
            if result is None:
                failed.append(stem)
            else:
                created.append(stem)

        print(f"\n{'=' * 70}")
        print('BATCH SUMMARY')
        print(f"{'=' * 70}")
        print(f'Created: {len(created)}')
        print(f'Failed: {len(failed)}')
        if failed:
            print('Failed items:')
            for stem in failed:
                print(f'  - {stem}')
        print(f"{'=' * 70}\n")

        return created


def main():
    parser = argparse.ArgumentParser(description='Create complete reference database')
    parser.add_argument('--video', help='Video file path')
    parser.add_argument('--annotation', help='Annotation JSON file path')
    parser.add_argument('--video-dir', help='Parent directory containing reference videos')
    parser.add_argument('--annotation-dir', help='Parent directory containing annotation JSON files')
    parser.add_argument('--output', default='compare/references', help='Output directory or output root')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument(
        '--pose-backend',
        type=str,
        default='rtmpose',
        choices=['rtmpose', 'mediapipe'],
        help='Pose backend for extracting reference keypoints',
    )
    parser.add_argument(
        '--trim',
        type=int,
        default=3,
        help='Frames to trim from the start of movements after the first (default: 3)',
    )

    args = parser.parse_args()

    single_mode = bool(args.video or args.annotation)
    batch_mode = bool(args.video_dir or args.annotation_dir)

    if single_mode and batch_mode:
        parser.error('Use either single-file mode (--video/--annotation) or batch mode (--video-dir/--annotation-dir), not both.')

    if single_mode and not (args.video and args.annotation):
        parser.error('Single-file mode requires both --video and --annotation.')

    if batch_mode and not (args.video_dir and args.annotation_dir):
        parser.error('Batch mode requires both --video-dir and --annotation-dir.')

    if not single_mode and not batch_mode:
        parser.error('Provide either --video/--annotation or --video-dir/--annotation-dir.')

    creator = ReferenceCreator(
        device=args.device,
        trim_start_frames=args.trim,
        pose_backend=args.pose_backend,
    )

    if single_mode:
        output_dir = creator.resolve_single_output_dir(args.video, args.output)
        creator.create_reference(args.video, args.annotation, output_dir)
    else:
        creator.create_reference_batch(args.video_dir, args.annotation_dir, args.output)


if __name__ == '__main__':
    main()




'''
* Single File Mode
 python compare/create_reference_complete.py --video data/reference/videos/2_jang/front/G018_TG2_front.mp4 --annotation data/reference/annotations/2_jang/front/G018_TG2_front.json --output compare/references_batch
 
* Batch Mode
 python compare/create_reference_complete.py --video-dir data/reference/videos/3_jang/front --annotation-dir data/reference/annotations/3_jang/front --output compare/references_batch
'''
