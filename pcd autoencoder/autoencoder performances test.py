import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import glob
import os
import json
from torch_geometric.nn import GCNConv, knn_graph

# =========================
# MODELES
# =========================

class PointCloudAE(nn.Module):
    def __init__(self, latent_dim=256, num_points=4096):
        super().__init__()
        self.num_points = num_points

        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)
        )

    def forward(self, x, return_latent=False):
        B, N, _ = x.shape

        x = self.encoder(x)
        latent = torch.max(x, dim=1)[0]

        recon = self.decoder(latent)
        recon = recon.view(B, self.num_points, 3)

        if return_latent:
            return recon, latent
        return recon


class GCNPointCloudAE(nn.Module):
    def __init__(self, latent_dim=256, num_points=4096, k=16):
        super().__init__()
        self.num_points = num_points
        self.k = k

        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)
        )

    def encode(self, x, batch):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        return x

    def forward(self, x, return_latent=False):
        B, N, _ = x.shape

        x = x.view(B * N, 3)
        batch = torch.arange(B).repeat_interleave(N).to(x.device)

        x = self.encode(x, batch)

        x = x.view(B, N, -1)
        latent = torch.max(x, dim=1)[0]

        recon = self.decoder(latent)
        recon = recon.view(B, self.num_points, 3)

        if return_latent:
            return recon, latent
        return recon


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

        self.fc_in = nn.Linear(3, 64)
        self.block1 = PTBlock(64, k)
        self.block2 = PTBlock(64, k)
        self.to_latent = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)
        )

    def forward(self, x, return_latent=False):
        B, N, _ = x.shape

        x = x.view(B * N, 3)
        x = self.fc_in(x)

        edge_index = knn_graph(x, k=self.k)

        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)

        x = self.to_latent(x)

        x = x.view(B, N, -1)
        latent = torch.max(x, dim=1)[0]

        recon = self.decoder(latent)
        recon = recon.view(B, self.num_points, 3)

        if return_latent:
            return recon, latent
        return recon


# =========================
# METRIQUES
# =========================

def chamfer_distance(pc1, pc2):
    pc1 = pc1.unsqueeze(1)
    pc2 = pc2.unsqueeze(0)

    dist = torch.norm(pc1 - pc2, dim=2)

    cd = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
    return cd


# =========================
# LOAD PCD
# =========================

def load_pcd(path, num_points):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)

    if len(pts) == 0:
        pts = np.zeros((num_points, 3))

    # seed déterministe basé sur le fichier
    seed = abs(hash(path)) % (2**32)
    rng = np.random.default_rng(seed)

    if len(pts) >= num_points:
        choice = rng.choice(len(pts), num_points, replace=False)
    else:
        choice = rng.choice(len(pts), num_points, replace=True)

    return torch.tensor(pts[choice], dtype=torch.float32)


# =========================
# LOAD MODEL
# =========================

def load_model(path, device, model_type):
    checkpoint = torch.load(path, map_location=device)

    latent_dim = checkpoint['latent_dim']
    num_points = checkpoint['num_points']

    if model_type == "GCN":
        model = GCNPointCloudAE(latent_dim, num_points)
    elif model_type == "TRANSFORMER":
        model = PointTransformerAE(latent_dim, num_points)
    else:
        model = PointCloudAE(latent_dim, num_points)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


# =========================
# EVALUATION
# =========================

def evaluate_model(model, input_folder, device):

    pcd_paths = glob.glob(input_folder + "/**/*.pcd", recursive=True)
    pcd_paths.sort()

    chamfer_scores = []
    mse_scores = []

    for path in pcd_paths:

        pts = load_pcd(path, model.num_points)
        pts = pts.unsqueeze(0).to(device)

        with torch.no_grad():
            recon = model(pts)

        pts = pts.squeeze(0)
        recon = recon.squeeze(0)

        cd = chamfer_distance(pts, recon)
        chamfer_scores.append(cd.item())

        mse = torch.mean((pts - recon) ** 2)
        mse_scores.append(mse.item())

    return {
        "chamfer_mean": float(np.mean(chamfer_scores)),
        "chamfer_std": float(np.std(chamfer_scores)),
        "mse_mean": float(np.mean(mse_scores)),
        "mse_std": float(np.std(mse_scores)),
    }


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    INPUT_FOLDER = r"E:\PAIR360\Traversal2\College_of_Physical_Education"

    checkpoints = {
        "GCN_4096points_256latent": ("checkpoint_GCN_4096points_256latent_100epochs.pth", "GCN"),
        "GCN_4096points_128latent": ("checkpoint_GCN_4096points_128latent_100epochs.pth", "GCN"),
        "GCN_2048points_256latent": ("checkpoint_GCN_2048points_256latent_100epochs.pth", "GCN"),
        "GCN_2048points_128latent": ("checkpoint_GCN_2048points_128latent_100epochs.pth", "GCN"),
        "GCN_1024points_256latent": ("checkpoint_GCN_1024points_256latent_100epochs.pth", "GCN"),
        "GCN_1024points_128latent": ("checkpoint_GCN_1024points_128latent_100epochs.pth", "GCN"),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for name, (path, model_type) in checkpoints.items():

        print(f"\n=== Evaluation {name} ===")

        model = load_model(path, device, model_type)

        stats = evaluate_model(model, INPUT_FOLDER, device)

        results[name] = stats

        print(stats)

    print("\n=== RESULTATS FINAUX ===")
    for k, v in results.items():
        print(k, v)

    # sauvegarde
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)