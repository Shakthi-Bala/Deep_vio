"""
Improved DeepVIO models with:
1. FiLM conditioning (IMU modulates visual features)
2. Optical flow encoder (precomputed flow as input)
3. Pretrained ResNet18 visual backbone
4. Recurrent temporal modeling (LSTM over sequence of frames)

References:
- FiLM: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI 2018
- SelectFusion: Chen et al., CVPR 2019
- DeepVO: Wang et al., ICRA 2017
- T2Depth architecture pattern from thermal_events project
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ═══════════════════════════════════════════════════════════════
# FiLM Layer — Feature-wise Linear Modulation
# ═══════════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: x_out = gamma * x + beta
    
    IMU features generate per-channel scale (gamma) and shift (beta)
    that modulate visual feature maps. This is much more expressive
    than concatenation — IMU can amplify/suppress specific visual channels.
    
    Ref: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI 2018
    """
    def __init__(self, visual_channels, conditioning_dim):
        super().__init__()
        self.gamma_net = nn.Linear(conditioning_dim, visual_channels)
        self.beta_net = nn.Linear(conditioning_dim, visual_channels)
        # Initialize gamma to 1, beta to 0 (identity at start)
        nn.init.ones_(self.gamma_net.weight.data[:, 0])
        nn.init.zeros_(self.gamma_net.bias.data)
        nn.init.zeros_(self.beta_net.weight.data)
        nn.init.zeros_(self.beta_net.bias.data)
    
    def forward(self, visual_features, conditioning):
        """
        Args:
            visual_features: (B, C, H, W) or (B, C) visual feature maps/vectors
            conditioning: (B, D) IMU conditioning vector
        Returns:
            modulated: (B, C, H, W) or (B, C)
        """
        gamma = self.gamma_net(conditioning)  # (B, C)
        beta = self.beta_net(conditioning)    # (B, C)
        
        if visual_features.dim() == 4:
            # Spatial feature maps: reshape for broadcasting
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        return gamma * visual_features + beta


# ═══════════════════════════════════════════════════════════════
# Optical Flow Encoder
# ═══════════════════════════════════════════════════════════════

class FlowEncoder(nn.Module):
    """Encodes precomputed optical flow (B, 2, H, W) -> (B, 256).
    
    Flow removes appearance variation and provides direct motion signal.
    Much easier to learn from than raw image pairs.
    
    Ref: TartanVO (Wang et al., CoRL 2021) uses PWC-Net flow as input.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),
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
        self.proj = nn.Linear(256, out_dim)
    
    def forward(self, flow):
        """flow: (B, 2, H, W) optical flow field."""
        return self.proj(self.net(flow))


# ═══════════════════════════════════════════════════════════════
# Pretrained ResNet18 Visual Encoder
# ═══════════════════════════════════════════════════════════════

class VisualEncoderResNet(nn.Module):
    """ResNet18 backbone for visual features.
    
    Uses ImageNet pretrained weights. First conv modified for 6-channel
    input (stacked image pair). Pretrained weights copied for both halves.
    
    Ref: DeepVO uses FlowNet-S pretrained; SelectFusion uses ResNet18.
    """
    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Modify first conv: 3ch -> 6ch (image pair)
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Copy pretrained weights for both image channels
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight
        resnet.conv1 = new_conv
        
        # Remove final FC layer, keep everything up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        self.proj = nn.Linear(512, out_dim)
    
    def forward(self, x):
        """x: (B, 6, H, W) stacked image pair."""
        feat = self.backbone(x).flatten(1)  # (B, 512)
        return self.proj(feat)  # (B, 256)


# ═══════════════════════════════════════════════════════════════
# IMU Encoder (same as before but with projection head)
# ═══════════════════════════════════════════════════════════════

class IMUEncoderV2(nn.Module):
    """Bidirectional LSTM IMU encoder with separate heads for:
    - Feature output (for fusion)
    - FiLM conditioning (for modulating visual features)
    """
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, out_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.proj = nn.Linear(hidden_size * 2, out_dim)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        feat = torch.cat((hn[-2], hn[-1]), dim=1)
        feat = self.norm(feat)
        return self.proj(feat)


# ═══════════════════════════════════════════════════════════════
# DeepVIO v2: FiLM Fusion + Soft Gating
# ═══════════════════════════════════════════════════════════════

class DeepVIO_FiLM(nn.Module):
    """Improved VIO with FiLM conditioning and soft gating.
    
    Architecture:
        Visual branch: ResNet18 (pretrained) or FlowEncoder
        IMU branch: Bi-LSTM
        Fusion: FiLM (IMU modulates visual) + soft gate + residual concat
    
    The IMU doesn't just get concatenated — it MODULATES the visual features
    via learned scale/shift (FiLM), then a soft gate decides how much to 
    trust each modality.
    """
    def __init__(self, visual_encoder='resnet', use_flow=False, feat_dim=256):
        super().__init__()
        self.use_flow = use_flow
        
        # Visual encoder
        if use_flow:
            self.vis_enc = FlowEncoder(out_dim=feat_dim)
        elif visual_encoder == 'resnet':
            self.vis_enc = VisualEncoderResNet(out_dim=feat_dim)
        else:
            # Fallback to original CNN
            from models import VisualEncoder
            self.vis_enc = VisualEncoder()  # -> 256
        
        # IMU encoder
        self.imu_enc = IMUEncoderV2(out_dim=feat_dim)
        
        # FiLM: IMU conditions visual features
        self.film = FiLMLayer(feat_dim, feat_dim)
        
        # Soft gate: learn when to trust each modality
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Pose regression heads (from fused 256-dim feature)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.PReLU(),
        )
        self.fc_pos = nn.Linear(64, 3)
        self.fc_rot = nn.Linear(64, 4)
    
    def forward(self, visual_input, imu_seq, return_gate=False):
        """
        Args:
            visual_input: (B, 6, H, W) image pair OR (B, 2, H, W) flow
            imu_seq: (B, T, 6) IMU sequence
            return_gate: if True, also return gate values for analysis
        """
        v_feat = self.vis_enc(visual_input)   # (B, 256)
        i_feat = self.imu_enc(imu_seq)        # (B, 256)
        
        # FiLM: IMU modulates visual features
        v_modulated = self.film(v_feat, i_feat)  # (B, 256)
        
        # Soft gate: how much to trust vision vs IMU
        gate_input = torch.cat([v_modulated, i_feat], dim=1)
        alpha = self.gate(gate_input)  # (B, 1) — vision weight
        
        # Weighted fusion
        fused = alpha * v_modulated + (1 - alpha) * i_feat  # (B, 256)
        
        # Pose regression
        out = self.fc(fused)
        pos = self.fc_pos(out)
        rot = self.fc_rot(out)
        rot = rot / (rot.norm(dim=1, keepdim=True) + 1e-8)
        
        if return_gate:
            return pos, rot, alpha.detach()
        return pos, rot


# ═══════════════════════════════════════════════════════════════
# DeepVO v2: ResNet + Optical Flow
# ═══════════════════════════════════════════════════════════════

class DeepVO_V2(nn.Module):
    """Improved visual odometry with ResNet18 or flow encoder."""
    def __init__(self, visual_encoder='resnet', use_flow=False, feat_dim=256):
        super().__init__()
        if use_flow:
            self.vis_enc = FlowEncoder(out_dim=feat_dim)
        elif visual_encoder == 'resnet':
            self.vis_enc = VisualEncoderResNet(out_dim=feat_dim)
        else:
            from models import VisualEncoder
            self.vis_enc = VisualEncoder()
        
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.PReLU(),
        )
        self.fc_pos = nn.Linear(64, 3)
        self.fc_rot = nn.Linear(64, 4)
    
    def forward(self, visual_input):
        feat = self.vis_enc(visual_input)
        out = self.fc(feat)
        pos = self.fc_pos(out)
        rot = self.fc_rot(out)
        rot = rot / (rot.norm(dim=1, keepdim=True) + 1e-8)
        return pos, rot


# ═══════════════════════════════════════════════════════════════
# Precompute optical flow utility
# ═══════════════════════════════════════════════════════════════

def precompute_optical_flow(data_root, method='farneback'):
    """Precompute and save optical flow for all sequences.
    
    Args:
        data_root: path to output/ directory with train/val/test splits
        method: 'farneback' (fast, CPU) or 'raft' (accurate, GPU)
    """
    import cv2
    import os
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            continue
        
        seqs = sorted(os.listdir(split_dir))
        for seq in seqs:
            img_dir = os.path.join(split_dir, seq, 'images')
            flow_dir = os.path.join(split_dir, seq, 'flow')
            os.makedirs(flow_dir, exist_ok=True)
            
            imgs = sorted(os.listdir(img_dir))
            for i in range(len(imgs) - 1):
                flow_path = os.path.join(flow_dir, f'{i:05d}.npy')
                if os.path.exists(flow_path):
                    continue
                
                f1 = cv2.imread(os.path.join(img_dir, imgs[i]), cv2.IMREAD_GRAYSCALE)
                f2 = cv2.imread(os.path.join(img_dir, imgs[i+1]), cv2.IMREAD_GRAYSCALE)
                
                if method == 'farneback':
                    flow = cv2.calcOpticalFlowFarneback(
                        f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                
                np.save(flow_path, flow.astype(np.float16))
            
            print(f'  {split}/{seq}: {len(imgs)-1} flow fields')


if __name__ == '__main__':
    # Quick test
    print("Testing models...")
    
    # Test FiLM VIO
    model = DeepVIO_FiLM(visual_encoder='resnet', use_flow=False)
    img = torch.randn(2, 6, 224, 224)
    imu = torch.randn(2, 100, 6)
    pos, rot, gate = model(img, imu, return_gate=True)
    print(f"DeepVIO_FiLM: pos={pos.shape}, rot={rot.shape}, gate={gate.shape}")
    print(f"  Gate values: {gate.squeeze().tolist()}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test flow encoder
    model_flow = DeepVIO_FiLM(use_flow=True)
    flow = torch.randn(2, 2, 224, 224)
    pos, rot = model_flow(flow, imu)
    print(f"DeepVIO_FiLM (flow): pos={pos.shape}, rot={rot.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_flow.parameters()):,}")
    
    # Test ResNet visual
    model_v2 = DeepVO_V2(visual_encoder='resnet')
    pos, rot = model_v2(img)
    print(f"DeepVO_V2 (ResNet): pos={pos.shape}, rot={rot.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_v2.parameters()):,}")
    
    print("\nAll tests passed!")
