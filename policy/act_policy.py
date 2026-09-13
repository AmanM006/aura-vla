import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import numpy as np

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel, temperature=1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = temperature
        
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height),
            np.linspace(-1.0, 1.0, self.width),
            indexing='ij'
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # feature: [B, C, H, W]
        B, C, H, W = feature.shape
        feature = feature.view(B, C, H * W)
        softmax_attention = nn.functional.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], dim=1) # [B, 2]
        return expected_xy

class VisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.spatial_softmax = SpatialSoftmax(8, 8, 256)
        
    def forward(self, x):
        features = self.cnn(x)
        B, C, H, W = features.shape
        # Spatial softmax requires spatial map to 2 coords, so output is B, 2
        # Actually standard spatial softmax outputs B, C, 2. But we flatten it.
        # Wait, the implemented spatial softmax outputs B, 2 by summing across channels?
        # Let's fix spatial softmax to output [B, C*2]
        return features

class VisionBackboneFixed(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(x))

class ACTPolicy(nn.Module):
    def __init__(self, obs_dim=12, action_dim=12, chunk_size=50, n_cameras=3, task_vocab=10):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_cameras = n_cameras
        self.task_vocab = task_vocab
        
        self.vision_backbones = nn.ModuleList([VisionBackboneFixed() for _ in range(n_cameras)])
        
        # Transformer encoder
        self.d_model = 256
        self.proj = nn.Linear(obs_dim + n_cameras * 512 + task_vocab, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * chunk_size)
        )
        
        # Dummy learned queries
        self.queries = nn.Parameter(torch.randn(1, chunk_size, self.d_model))
        
    def forward(self, obs_state, images, task_token) -> torch.Tensor:
        # obs_state: [B, obs_dim]
        # images: [B, n_cameras, 3, 128, 128] or list of [B, 3, 128, 128]
        # task_token: [B, task_vocab]
        
        if isinstance(images, (list, tuple)):
            img_feats = []
            for i, img in enumerate(images):
                img_feats.append(self.vision_backbones[i](img))
        else:
            img_feats = []
            for i in range(self.n_cameras):
                img_feats.append(self.vision_backbones[i](images[:, i]))
                
        img_feats = torch.cat(img_feats, dim=1) # [B, n_cameras * 512]
        
        # concat state, vision, task
        fused = torch.cat([obs_state, img_feats, task_token], dim=1) # [B, dim]
        
        # project to d_model
        x = self.proj(fused).unsqueeze(1) # [B, 1, d_model]
        
        # Transformer
        out = self.transformer(x) # [B, 1, d_model]
        
        # decode
        actions = self.decoder(out.squeeze(1)) # [B, action_dim * chunk_size]
        return actions.view(-1, self.chunk_size, self.action_dim)
        
    def predict_action(self, obs_state, images, task_token) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            obs = torch.FloatTensor(obs_state).unsqueeze(0)
            imgs = [torch.FloatTensor(img).unsqueeze(0).permute(0, 3, 1, 2) for img in images]
            task = torch.zeros(1, self.task_vocab)
            task[0, task_token] = 1.0
            
            actions = self.forward(obs, imgs, task)
            # return first 25 actions
            return actions[0, :25].cpu().numpy()
            
    def save(self, path: Path):
        torch.save(self.state_dict(), path)
        
    def load(self, path: Path):
        self.load_state_dict(torch.load(path))
