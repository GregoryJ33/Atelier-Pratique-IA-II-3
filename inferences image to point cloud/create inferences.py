import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import open3d as o3d
from torch_geometric.nn import GCNConv, knn_graph

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = (768, 384)
NUM_POINTS = 4096
LATENT_DIM = 256

CNN_CHECKPOINT = "../image to pcd/checkpoints/CNN_checkpoint_best.pt"
AE_CHECKPOINT = "../pcd autoencoder/checkpoint_GCN_4096points_256latent_100epochs.pth"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# MODELES
# =========================

class CNNRegressor(nn.Module):
    def __init__(self, in_channels=7, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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

    def forward(self, x):
        B, N, _ = x.shape

        x = x.view(B * N, 3)
        batch = torch.arange(B).repeat_interleave(N).to(x.device)

        x = self.encode(x, batch)

        x = x.view(B, N, -1)
        x = torch.max(x, dim=1)[0]

        x = self.decoder(x)
        x = x.view(B, self.num_points, 3)

        return x


# =========================
# UTILS
# =========================

def load_models():
    # CNN
    cnn = CNNRegressor(in_channels=7, latent_dim=LATENT_DIM).to(DEVICE)
    ckpt = torch.load(CNN_CHECKPOINT, map_location=DEVICE)
    cnn.load_state_dict(ckpt["model"])
    cnn.eval()

    # AE
    ae = GCNPointCloudAE(latent_dim=LATENT_DIM, num_points=NUM_POINTS).to(DEVICE)
    ckpt = torch.load(AE_CHECKPOINT, map_location=DEVICE)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()

    return cnn, ae


def load_image_triplet(stitch_path, depth_path, seg_path):
    def load_img(path, mode):
        img = Image.open(path).resize(IMG_SIZE, mode)
        img = np.array(img).astype(np.float32) / 255.0

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        return img

    stitch = load_img(stitch_path, Image.BILINEAR)
    depth = load_img(depth_path, Image.BILINEAR)
    seg = load_img(seg_path, Image.NEAREST)

    x = np.concatenate([stitch, depth, seg], axis=2)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    return x


def save_pointcloud(points, filename):
    points = np.asarray(points, dtype=np.float64)
    points = np.nan_to_num(points)

    assert points.ndim == 2 and points.shape[1] == 3, f"Shape invalide: {points.shape}"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(filename, pcd)


# =========================
# PIPELINE INFERENCE
# =========================

def inference(stitch_path, depth_path, seg_path, name="sample"):
    cnn, ae = load_models()

    # Load image
    x = load_image_triplet(stitch_path, depth_path, seg_path)
    x = x.to(DEVICE)

    with torch.no_grad():
        # Image -> latent
        latent = cnn(x)

        recon = ae.decoder(latent)

        recon = recon.view(NUM_POINTS, 3)

    recon = recon.squeeze(0).cpu().numpy()

    # Save
    output_path = os.path.join(OUTPUT_DIR, f"{name}.pcd")
    save_pointcloud(recon, output_path)

    print(f"Point cloud sauvegardé: {output_path}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    for i in range(15):
        depth = rf"E:\PAIR360\Traversal2\College_of_Physical_Education\Sequence0\T2-College_of_Physical_Education-0-depth\depth\{i:04d}.png"
        seg = rf"E:\PAIR360\Traversal2\College_of_Physical_Education\Sequence0\T2-College_of_Physical_Education-0-segmentation\segmentation\{i:04d}.png"
        stitch = rf"E:\PAIR360\Traversal2\College_of_Physical_Education\Sequence0\T2-College_of_Physical_Education-0-stitching\stitching\{i:04d}.png"

        inference(stitch, depth, seg, name=f"{i:04d}")
