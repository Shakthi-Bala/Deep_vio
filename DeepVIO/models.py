import torch
import torch.nn as nn


class CombinedVIOLoss(nn.Module):
    """L1 position loss + geodesic quaternion loss."""

    def __init__(self, lambda_p=1.0, lambda_q=1.0):
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
        self.l1 = nn.L1Loss()

    def forward(self, pred_p, pred_q, gt_p, gt_q):
        pos_loss = self.l1(pred_p, gt_p)
        # Geodesic angle between unit quaternions: 2*acos(|q1·q2|)
        dot = torch.clamp(torch.abs((pred_q * gt_q).sum(dim=1)), max=1.0 - 1e-7)
        rot_loss = 2.0 * torch.acos(dot).mean()
        total = self.lambda_p * pos_loss + self.lambda_q * rot_loss
        return total, pos_loss.detach(), rot_loss.detach()


class VisualEncoder(nn.Module):
    """
    Encodes a stacked image pair (B, 6, H, W) → (B, 256).
    3-stage conv pyramid + global average pool.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class IMUEncoder(nn.Module):
    """
    Encodes an IMU sequence (B, T, 6) → (B, 256) via bidirectional LSTM.
    Concatenates final forward + backward hidden states of the last layer.
    """

    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        # hn: (num_layers*2, B, hidden) — last layer: hn[-2] fwd, hn[-1] bwd
        feat = torch.cat((hn[-2], hn[-1]), dim=1)  # (B, 256)
        return self.norm(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: IMU-only odometry
# ─────────────────────────────────────────────────────────────────────────────
class DeepIO(nn.Module):

    def __init__(self, use_attention=False):
        super().__init__()
        self.imu_enc = IMUEncoder()
        self.attn = (
            nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
            if use_attention else None
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
        )
        self.fc_pos = nn.Linear(128, 3)
        self.fc_rot = nn.Linear(128, 4)

    def forward(self, imu_seq):
        feat = self.imu_enc(imu_seq)
        if self.attn is not None:
            feat, _ = self.attn(feat.unsqueeze(1), feat.unsqueeze(1), feat.unsqueeze(1))
            feat = feat.squeeze(1)
        out = self.fc(feat)
        pos = self.fc_pos(out)
        rot = self.fc_rot(out)
        rot = rot / (rot.norm(dim=1, keepdim=True) + 1e-8)
        return pos, rot


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: Vision-only odometry
# ─────────────────────────────────────────────────────────────────────────────
class DeepVO(nn.Module):
    """
    Processes a single image pair (B, 6, H, W).
    No temporal LSTM — the visual encoder already captures inter-frame motion
    via channel concatenation. Extend to a sequence of pairs for LSTM context.
    """

    def __init__(self, use_attention=False):
        super().__init__()
        self.vis_enc = VisualEncoder()
        self.attn = (
            nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
            if use_attention else None
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.PReLU(),
        )
        self.fc_pos = nn.Linear(64, 3)
        self.fc_rot = nn.Linear(64, 4)

    def forward(self, img_pair):
        feat = self.vis_enc(img_pair)
        if self.attn is not None:
            feat, _ = self.attn(feat.unsqueeze(1), feat.unsqueeze(1), feat.unsqueeze(1))
            feat = feat.squeeze(1)
        out = self.fc(feat)
        pos = self.fc_pos(out)
        rot = self.fc_rot(out)
        rot = rot / (rot.norm(dim=1, keepdim=True) + 1e-8)
        return pos, rot


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: Visual-Inertial fusion
# ─────────────────────────────────────────────────────────────────────────────
class DeepVIO(nn.Module):

    def __init__(self, use_attention=False):
        super().__init__()
        self.vis_enc = VisualEncoder()   # → 256
        self.imu_enc = IMUEncoder()      # → 256
        self.attn = (
            nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
            if use_attention else None
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.PReLU(),
        )
        self.fc_pos = nn.Linear(128, 3)
        self.fc_rot = nn.Linear(128, 4)

    def forward(self, img_pair, imu_seq):
        v_feat = self.vis_enc(img_pair)
        i_feat = self.imu_enc(imu_seq)
        fused = torch.cat((v_feat, i_feat), dim=1)   # (B, 512)
        if self.attn is not None:
            fused, _ = self.attn(fused.unsqueeze(1), fused.unsqueeze(1), fused.unsqueeze(1))
            fused = fused.squeeze(1)
        out = self.fc(fused)
        pos = self.fc_pos(out)
        rot = self.fc_rot(out)
        rot = rot / (rot.norm(dim=1, keepdim=True) + 1e-8)
        return pos, rot
