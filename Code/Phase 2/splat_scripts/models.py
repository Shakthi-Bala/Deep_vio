import torch
import torch.nn as nn
import torch.nn.functional as F


def _correlation(f0, f1, max_disp: int):
    """
    Local correlation volume between f0 and f1.
    For each position in f1, computes dot products with a (2*max_disp+1)^2
    neighbourhood in f0 — the same operation used in FlowNet / PWC-Net.
    Returns (B, (2D+1)^2, H, W).
    """
    B, C, H, W = f0.shape
    d = max_disp
    f0_pad = F.pad(f0, [d, d, d, d])
    f0_n = F.normalize(f0_pad, dim=1)
    f1_n = F.normalize(f1, dim=1)
    K = (2 * d + 1) ** 2
    out = f0.new_zeros(B, K, H, W)
    k = 0
    for dy in range(2 * d + 1):
        for dx in range(2 * d + 1):
            out[:, k] = (f0_n[:, :, dy:dy + H, dx:dx + W] * f1_n).sum(1)
            k += 1
    return out


class FlowEncoder(nn.Module):
    """
    Siamese backbone + local correlation volume.
    Each frame is encoded independently with shared weights; a correlation
    volume captures inter-frame correspondences (motion signal).  This is
    architecturally equivalent to the front-end of FlowNet/PWC-Net and gives
    the pose head an explicit flow representation rather than raw pixel stacks.
    Output: 256-dim feature vector.
    """
    def __init__(self, max_disp: int = 4, dropout: float = 0.3):
        super().__init__()
        self.max_disp = max_disp
        # Shared Siamese backbone — each frame encoded independently
        self.backbone = nn.Sequential(
            nn.Conv2d(3,  32,  7, stride=2, padding=3), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32, 64,  5, stride=2, padding=2), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        )  # → (B, 128, H/8, W/8)

        corr_ch = (2 * max_disp + 1) ** 2  # 81 channels for max_disp=4
        # Flow head: correlation volume + img1 context features
        self.flow_head = nn.Sequential(
            nn.Conv2d(corr_ch + 128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, img0, img1):
        f0 = self.backbone(img0)                          # (B, 128, H', W')
        f1 = self.backbone(img1)                          # (B, 128, H', W')
        corr = _correlation(f0, f1, self.max_disp)        # (B, 81,  H', W')
        x = torch.cat([corr, f1], dim=1)                  # (B, 209, H', W')
        return self.flow_head(x).view(img0.shape[0], -1)  # (B, 256)


class IMUEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


class PoseRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )

    def forward(self, x):
        out = self.head(x)
        trans = out[:, :3]
        quat = out[:, 3:]
        quat = F.normalize(quat, dim=1)
        return torch.cat([trans, quat], dim=1)


class VisionOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FlowEncoder()
        self.regressor = PoseRegressor(256)

    def forward(self, img0, img1):
        z = self.encoder(img0, img1)
        return self.regressor(z)


class IMUOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IMUEncoder()
        self.regressor = PoseRegressor(128)

    def forward(self, imu):
        z = self.encoder(imu)
        return self.regressor(z)


class VIOFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = FlowEncoder()
        self.imu_encoder = IMUEncoder()
        self.regressor = PoseRegressor(384)  # 256 (flow) + 128 (IMU)

    def forward(self, img0, img1, imu):
        vis = self.vision_encoder(img0, img1)
        imu_f = self.imu_encoder(imu)
        z = torch.cat([vis, imu_f], dim=1)
        return self.regressor(z)
