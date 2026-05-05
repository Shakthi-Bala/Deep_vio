import glob
import os
import re
from io import StringIO

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticVIODataset(Dataset):
    def __init__(self, root_dir, split="train", mode="vio", transforms=None,
                 sample_split=True, seed=42):
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms
        self.data = []

        sequence_paths = self._discover_sequences(root_dir)
        if sequence_paths:
            if sample_split and len(sequence_paths) > 1:
                # Load every sequence, then split at the sample level so all
                # scenes appear in both train and val.
                for seq_path in sequence_paths:
                    self._load_directory_sequence(seq_path)
                self.data = self._sample_split(self.data, split, seed)
            else:
                sequence_paths = self._split_paths(sequence_paths, split)
                for seq_path in sequence_paths:
                    self._load_directory_sequence(seq_path)
        else:
            npz_files = sorted(glob.glob(os.path.join(root_dir, "sequence_*.npz")))
            npz_files = self._split_paths(npz_files, split)
            for file_path in npz_files:
                self._load_npz_sequence(file_path)

    def _sample_split(self, data, split, seed):
        rng = np.random.default_rng(seed)
        indices = np.arange(len(data))
        rng.shuffle(indices)
        n = len(indices)
        train_end = int(0.8 * n)
        val_end   = int(0.9 * n)
        if split == "train":
            chosen = indices[:train_end]
        elif split == "val":
            chosen = indices[train_end:val_end]
        else:
            chosen = indices[val_end:]
        return [data[i] for i in chosen]

    def _split_paths(self, paths, split):
        if not paths:
            return []
        n = len(paths)
        if n == 1:
            # Only one sequence: use it for all splits
            return paths
        train_end = max(1, int(0.8 * n))
        val_end = max(train_end + 1, int(0.9 * n))
        val_end = min(val_end, n)
        if split == "train":
            return paths[:train_end]
        elif split == "val":
            return paths[train_end:val_end]
        return paths[val_end:]

    def _discover_sequences(self, root_dir):
        if self._is_sequence_dir(root_dir):
            return [root_dir]
        sequence_dirs = []
        for entry in sorted(os.listdir(root_dir)):
            entry_path = os.path.join(root_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if self._is_sequence_dir(entry_path):
                sequence_dirs.append(entry_path)
            else:
                for sub in sorted(os.listdir(entry_path)):
                    sub_path = os.path.join(entry_path, sub)
                    if os.path.isdir(sub_path) and self._is_sequence_dir(sub_path):
                        sequence_dirs.append(sub_path)
        return sequence_dirs

    def _is_sequence_dir(self, path):
        return (
            os.path.isdir(os.path.join(path, "images"))
            and os.path.isfile(os.path.join(path, "imu.csv"))
            and os.path.isfile(os.path.join(path, "relative_poses.csv"))
        )

    def _load_npz_sequence(self, file_path):
        archive = np.load(file_path)
        images = archive["images"].astype(np.float32) / 255.0
        imu = archive["imu"].astype(np.float32)
        rel_poses = archive["rel_poses"].astype(np.float32)
        for i in range(images.shape[0] - 1):
            self.data.append(
                {
                    "img0": images[i],
                    "img1": images[i + 1],
                    "imu": imu[i],
                    "pose": rel_poses[i],
                }
            )

    def _load_directory_sequence(self, seq_dir):
        metadata = self._read_metadata(os.path.join(seq_dir, "metadata.txt"))
        image_dir = os.path.join(seq_dir, "images")
        image_paths = self._sorted_image_list(image_dir)
        imu = self._read_imu_csv(os.path.join(seq_dir, "imu.csv"))
        rel_poses = self._read_relative_poses_csv(os.path.join(seq_dir, "relative_poses.csv"))

        num_frames = len(image_paths)
        num_transitions = rel_poses.shape[0]
        expected_frames = num_transitions + 1

        frame_step = self._camera_step(num_frames, expected_frames)
        if frame_step is None:
            raise ValueError(
                f"Cannot infer camera sampling step for {seq_dir}: {num_frames} images, {num_transitions} transitions"
            )

        frame_indices = np.arange(0, num_frames, frame_step, dtype=int)
        frame_indices = frame_indices[: expected_frames]
        if len(frame_indices) != expected_frames:
            raise ValueError(
                f"Sequence {seq_dir} has mismatched image/frame counts: found {len(frame_indices)} usable camera frames, expected {expected_frames}"
            )

        imu_step = self._imu_step(imu.shape[0], num_transitions, metadata)
        if imu_step <= 0:
            raise ValueError(
                f"Cannot infer IMU window length for {seq_dir}: imu samples {imu.shape[0]}, transitions {num_transitions}"
            )
        if imu.shape[0] < num_transitions * imu_step:
            raise ValueError(
                f"Not enough IMU samples for {seq_dir}: need {num_transitions * imu_step}, got {imu.shape[0]}"
            )

        for i in range(num_transitions):
            imu_window = imu[i * imu_step : (i + 1) * imu_step]
            self.data.append(
                {
                    "img0": image_paths[frame_indices[i]],
                    "img1": image_paths[frame_indices[i + 1]],
                    "imu": imu_window,
                    "pose": rel_poses[i],
                }
            )

    def _read_metadata(self, file_path):
        if not os.path.isfile(file_path):
            return {}
        metadata = {}
        with open(file_path, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        return metadata

    def _imu_step(self, imu_count, num_transitions, metadata):
        if "imu_hz" in metadata and "cam_hz" in metadata:
            try:
                imu_hz = int(float(metadata["imu_hz"]))
                cam_hz = int(float(metadata["cam_hz"]))
                if cam_hz > 0:
                    return imu_hz // cam_hz
            except ValueError:
                pass
        if num_transitions > 0:
            step = imu_count // num_transitions
            if step > 0:
                return step
        return 0

    def _sorted_image_list(self, image_dir):
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(os.path.join(image_dir, pattern)))
        if not paths:
            raise FileNotFoundError(f"No image files found in {image_dir}")
        return sorted(paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def _camera_step(self, num_images, expected_frames):
        if expected_frames <= 1:
            return 1
        if num_images == expected_frames:
            return 1
        step = num_images // expected_frames
        if step <= 0:
            return None
        if expected_frames * step <= num_images:
            return step
        return None

    def _clean_csv_text(self, text):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n\.(?=\d)", ".", text)
        return text

    def _read_csv(self, file_path):
        with open(file_path, "r") as f:
            text = f.read()
        text = self._clean_csv_text(text)
        data = np.genfromtxt(StringIO(text), delimiter=",", skip_header=1)
        if data.ndim == 1 and data.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return data.astype(np.float32)

    def _read_imu_csv(self, file_path):
        raw = self._read_csv(file_path)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        return raw[:, 1:]

    def _read_relative_poses_csv(self, file_path):
        raw = self._read_csv(file_path)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        rel = raw[:, 1:]
        if rel.shape[1] != 7:
            raise ValueError(f"relative_poses.csv must have 8 columns, got {rel.shape[1] + 1}")
        rel_poses = np.concatenate([rel[:, :3], rel[:, 6:7], rel[:, 3:6]], axis=1)
        return rel_poses

    def _load_image(self, image_data):
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
            return np.array(image, dtype=np.float32) / 255.0
        return image_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pose = torch.from_numpy(item["pose"])
        sample = {"pose": pose}

        if self.mode in ["vision", "vio"]:
            img0 = self._load_image(item["img0"])
            img1 = self._load_image(item["img1"])
            img0 = torch.from_numpy(img0).permute(2, 0, 1)
            img1 = torch.from_numpy(img1).permute(2, 0, 1)
            if self.transforms is not None:
                img0 = self.transforms(img0)
                img1 = self.transforms(img1)
            sample["img0"] = img0
            sample["img1"] = img1

        if self.mode in ["imu", "vio"]:
            imu = torch.from_numpy(item["imu"])
            sample["imu"] = imu

        return sample
