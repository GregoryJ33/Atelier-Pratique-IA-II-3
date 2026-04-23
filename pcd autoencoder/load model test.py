import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from torch_geometric.nn import GCNConv, knn_graph


class PointCloudAE(nn.Module):
    def __init__(self, latent_dim=256, num_points=4096):
        super().__init__()
        self.num_points = num_points

        # Encoder point-wise
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder global
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)
        )

    def forward(self, x):
        B, N, _ = x.shape

        x = self.encoder(x)          # (B, N, latent)
        x = torch.max(x, dim=1)[0]   # global feature (B, latent)

        x = self.decoder(x)          # (B, N*3)
        x = x.view(B, self.num_points, 3)

        return x


class GCNPointCloudAE(nn.Module):
    def __init__(self, latent_dim=256, num_points=4096, k=16):
        super().__init__()

        self.num_points = num_points
        self.k = k

        # -------------------
        # ENCODER GCN
        # -------------------
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, latent_dim)

        # -------------------
        # DECODER (MLP global)
        # -------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)
        )

    def encode(self, x, batch):
        # x: (B*N, 3)

        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        return x

    def forward(self, x):
        """
        x: (B, N, 3)
        """

        B, N, _ = x.shape

        x = x.view(B * N, 3)
        batch = torch.arange(B).repeat_interleave(N).to(x.device)

        x = self.encode(x, batch)

        # global pooling (max)
        x = x.view(B, N, -1)
        x = torch.max(x, dim=1)[0]   # (B, latent_dim)

        x = self.decoder(x)
        x = x.view(B, self.num_points, 3)

        return x


class PTBlock(nn.Module):
    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, edge_index):
        # x: (N, C)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        row, col = edge_index

        attn = (q[row] * k[col]).sum(-1, keepdim=True)
        attn = torch.softmax(attn, dim=0)

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, attn * v[col])

        return x + self.mlp(agg)


class PointTransformerAE(nn.Module):
    def __init__(self, latent_dim=256, num_points=4096, k=16):
        super().__init__()

        self.num_points = num_points
        self.k = k

        # -------------------
        # ENCODER
        # -------------------
        self.fc_in = nn.Linear(3, 64)

        self.block1 = PTBlock(64, k)
        self.block2 = PTBlock(64, k)

        self.to_latent = nn.Linear(64, latent_dim)

        # -------------------
        # DECODER (MLP global)
        # -------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)
        )

    def forward(self, x):
        """
        x: (B, N, 3)
        """

        B, N, _ = x.shape

        x = x.view(B * N, 3)
        x = self.fc_in(x)

        edge_index = knn_graph(x, k=self.k)

        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)

        x = self.to_latent(x)

        x = x.view(B, N, -1)
        x = torch.max(x, dim=1)[0]  # global latent

        x = self.decoder(x)
        x = x.view(B, self.num_points, 3)

        return x


GCN = "GCN"
TRANSFORMER = "TRANSFORMER"
model_type = GCN

EPOCHS = 2
NUM_POINTS = 1024
LATENT_DIM = 128


checkpoint = torch.load(f"checkpoint_{model_type}_{NUM_POINTS}points_{LATENT_DIM}latent_{EPOCHS}epochs.pth")

latent_dim = checkpoint['latent_dim']
num_points = checkpoint['num_points']

if model_type == GCN:
    model = GCNPointCloudAE(latent_dim=latent_dim, num_points=num_points)
elif model_type == TRANSFORMER:
    model = PointTransformerAE(latent_dim=latent_dim, num_points=num_points)
else:
    model = PointCloudAE(latent_dim=latent_dim, num_points=num_points)

model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

start_epoch = checkpoint['epoch']
loss = checkpoint['loss']

print("Test didn't crash")
