import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import glob
import os
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
# LOAD PCD
# =========================
def load_pcd(path, num_points):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)

    if len(pts) == 0:
        pts = np.zeros((num_points, 3))

    if len(pts) >= num_points:
        choice = np.random.choice(len(pts), num_points, replace=False)
    else:
        choice = np.random.choice(len(pts), num_points, replace=True)

    return torch.tensor(pts[choice], dtype=torch.float32)


# =========================
# VISUALISATION
# =========================
def visualize(original, reconstructed):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(original)
    pcd1.paint_uniform_color([1, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(reconstructed)
    pcd2.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pcd1, pcd2])


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
# INFERENCE + LATENTS
# =========================
def run_inference(model, input_folder, output_folder, device):

    os.makedirs(output_folder, exist_ok=True)

    pcd_paths = glob.glob(input_folder + "/**/*.pcd", recursive=True)
    pcd_paths.sort()

    print(f"{len(pcd_paths)} fichiers trouvés")

    all_latents = []

    for i, path in enumerate(pcd_paths):

        pts = load_pcd(path, model.num_points)
        pts = pts.unsqueeze(0).to(device)

        with torch.no_grad():
            recon, latent = model(pts, return_latent=True)

        pts_np = pts.squeeze(0).cpu().numpy()
        recon_np = recon.squeeze(0).cpu().numpy()
        latent_np = latent.squeeze(0).cpu().numpy()

        # sauvegarde latent individuel
        filename = os.path.basename(path).replace(".pcd", ".npy")
        np.save(os.path.join(output_folder, filename), latent_np)

        all_latents.append(latent_np)

        # visualisation (1 sur 50)
        # if i % 50 == 0:
        #     print(f"{i}/{len(pcd_paths)} traité")
        #     visualize(pts_np, recon_np)

    # sauvegarde globale
    # all_latents = np.array(all_latents)
    # np.save(os.path.join(output_folder, "all_latents.npy"), all_latents)



GCN = "GCN"
TRANSFORMER = "TRANSFORMER"
# =========================
# MAIN
# =========================
if __name__ == "__main__":

    CHECKPOINT_PATH = "checkpoint_GCN_4096points_256latent_100epochs.pth"
    INPUT_FOLDER = r"E:\PAIR360\Traversal2\College_of_Life_Science\3"
    OUTPUT_FOLDER = r".\latents"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(CHECKPOINT_PATH, device, model_type=GCN)

    run_inference(model, INPUT_FOLDER, OUTPUT_FOLDER, device)