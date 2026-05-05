import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.conv(x).view(x.shape[0], -1)


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
        self.encoder = VisionEncoder(in_channels=6)
        self.regressor = PoseRegressor(256)

    def forward(self, img0, img1):
        x = torch.cat([img0, img1], dim=1)
        z = self.encoder(x)
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
        self.vision_encoder = VisionEncoder(in_channels=6)
        self.imu_encoder = IMUEncoder()
        self.regressor = PoseRegressor(384)

    def forward(self, img0, img1, imu):
        vis = self.vision_encoder(torch.cat([img0, img1], dim=1))
        imu_f = self.imu_encoder(imu)
        z = torch.cat([vis, imu_f], dim=1)
        return self.regressor(z)
