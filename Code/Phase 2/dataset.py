import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class VIODataset(Dataset):
    """
    Loads synthetic VIO sequences from blender_script.py output.

    Directory layout expected:
        root_dir/
            seq_001/
                images/  00000.png … 00999.png
                imu.csv          (timestamp, gx, gy, gz, ax, ay, az)
                relative_poses.csv (timestamp, dtx, dty, dtz, dqx, dqy, dqz, dqw)

    Each __getitem__ returns one consecutive frame pair:
        img_pair : (6, H, W)   — [frame_t | frame_{t+1}] normalised
        imu_seq  : (T, 6)      — IMU readings between the two frames
        gt_p     : (3,)        — relative translation (body frame)
        gt_q     : (4,)        — relative rotation quaternion [qx qy qz qw]
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, root_dir: str, img_size=(224, 224), augment: bool = False):
        self.root_dir = root_dir
        self.samples = []

        normalize = transforms.Normalize(mean=self._MEAN, std=self._STD)
        resize    = transforms.Resize(img_size, antialias=True)

        if augment:
            self.img_tf = transforms.Compose([
                resize,
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.img_tf = transforms.Compose([
                resize,
                transforms.ToTensor(),
                normalize,
            ])

        seq_dirs = sorted(glob.glob(os.path.join(root_dir, "seq_*")))
        if not seq_dirs:
            raise FileNotFoundError(f"No seq_* directories found in {root_dir}")

        for d in seq_dirs:
            self._load_sequence(d)

        print(f"[VIODataset] {len(seq_dirs)} sequences → {len(self.samples)} samples")

    # ------------------------------------------------------------------
    def _load_sequence(self, seq_dir: str):
        img_dir  = os.path.join(seq_dir, "images")
        imu_path = os.path.join(seq_dir, "imu.csv")
        rel_path = os.path.join(seq_dir, "relative_poses.csv")

        if not all(os.path.exists(p) for p in [img_dir, imu_path, rel_path]):
            return

        imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if len(imgs) < 2:
            return

        imu_data  = np.loadtxt(imu_path,  delimiter=",", skiprows=1, dtype=np.float32)
        rel_poses = np.loadtxt(rel_path,   delimiter=",", skiprows=1, dtype=np.float32)

        # Infer how many IMU steps fall between consecutive camera frames.
        n_imu     = len(imu_data)
        n_cam     = len(imgs)
        cam_step  = max(1, n_imu // n_cam)       # e.g. 10 or 100

        n_pairs = min(n_cam - 1, len(rel_poses))

        for i in range(n_pairs):
            imu_start = i * cam_step
            imu_end   = (i + 1) * cam_step
            if imu_end > n_imu:
                break
            self.samples.append({
                "img0":  imgs[i],
                "img1":  imgs[i + 1],
                "imu":   imu_data[imu_start:imu_end, 1:],  # (T, 6)
                "gt_p":  rel_poses[i, 1:4],                # (3,)
                "gt_q":  rel_poses[i, 4:8],                # (4,)
            })

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img0 = self.img_tf(Image.open(s["img0"]).convert("RGB"))
        img1 = self.img_tf(Image.open(s["img1"]).convert("RGB"))
        img_pair = torch.cat([img0, img1], dim=0)           # (6, H, W)

        imu  = torch.from_numpy(s["imu"].copy())            # (T, 6)
        gt_p = torch.from_numpy(s["gt_p"].copy())           # (3,)
        gt_q = torch.from_numpy(s["gt_q"].copy())           # (4,)

        return img_pair, imu, gt_p, gt_q


# ─────────────────────────────────────────────────────────────────────────────
def make_dataloaders(
    root_dir:    str,
    batch_size:  int   = 32,
    val_split:   float = 0.15,
    num_workers: int   = 4,
    img_size:    tuple = (224, 224),
    seed:        int   = 42,
):
    """
    Create train/val/test dataloaders.

    Supports two directory layouts:

    1. Split-based (preferred — from updated blender_script.py):
        root_dir/
            train/seq_001/ seq_002/ ...
            val/seq_001/   seq_002/ ...
            test/seq_001/  seq_002/ ...

    2. Flat (legacy — original blender_script.py):
        root_dir/
            seq_001/ seq_002/ ...
       In this case, a random train/val split is applied (no test set).
    """
    train_dir = os.path.join(root_dir, "train")
    val_dir   = os.path.join(root_dir, "val")
    test_dir  = os.path.join(root_dir, "test")

    # ── Split-based layout ────────────────────────────────────────────────
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_ds = VIODataset(train_dir, img_size=img_size, augment=True)
        val_ds   = VIODataset(val_dir,   img_size=img_size, augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        test_loader = None
        if os.path.isdir(test_dir):
            try:
                test_ds = VIODataset(test_dir, img_size=img_size, augment=False)
                test_loader = DataLoader(
                    test_ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                )
            except FileNotFoundError:
                pass

        return train_loader, val_loader, test_loader

    # ── Flat layout (legacy) ──────────────────────────────────────────────
    else:
        train_ds = VIODataset(root_dir, img_size=img_size, augment=True)
        val_ds   = VIODataset(root_dir, img_size=img_size, augment=False)

        n_val   = max(1, int(len(train_ds) * val_split))
        n_train = len(train_ds) - n_val
        gen = torch.Generator().manual_seed(seed)
        train_indices, val_indices = random_split(
            range(len(train_ds)), [n_train, n_val], generator=gen
        )

        from torch.utils.data import Subset
        train_loader = DataLoader(
            Subset(train_ds, train_indices.indices),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            Subset(val_ds, val_indices.indices),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        return train_loader, val_loader, None
