import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from torch_geometric.nn import GCNConv, knn_graph


# =========================
# DATASET POINT CLOUD
# =========================
class PointCloudDataset(Dataset):
    def __init__(self, pcd_paths, num_points=4096):
        self.pcd_paths = pcd_paths
        self.num_points = num_points

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.pcd_paths[idx])
        pts = np.asarray(pcd.points)

        if len(pts) == 0:
            pts = np.zeros((self.num_points, 3))

        # sampling / duplication
        if len(pts) >= self.num_points:
            choice = np.random.choice(len(pts), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(pts), self.num_points, replace=True)

        pts = pts[choice]

        return torch.tensor(pts, dtype=torch.float32)


# =========================
# AUTOENCODEUR SIMPLE
# =========================
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


# =========================
# CHAMFER LOSS SIMPLE (sans pytorch3d)
# =========================
def chamfer_distance(p1, p2):
    """
    Version simple (pas optimale mais fonctionne sans dépendance externe)
    p1, p2: (B, N, 3)
    """
    diff1 = torch.cdist(p1, p2).min(dim=2)[0]
    diff2 = torch.cdist(p2, p1).min(dim=2)[0]

    return (diff1.mean() + diff2.mean())


# =========================
# TRAINING LOOP
# =========================
def train(folders, epochs=50, batch_size=8, num_points=4096, val_split=0.2, model_type="", latent_dim=256):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pcd_paths = []

    for f in folders:
        pcd_paths.extend(glob.glob(f + r"\**\*.pcd", recursive=True))

    # split train / val
    split = int(len(pcd_paths) * (1 - val_split))
    train_paths = pcd_paths[:split]
    val_paths = pcd_paths[split:]

    train_ds = PointCloudDataset(train_paths, num_points)
    val_ds = PointCloudDataset(val_paths, num_points)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    if model_type == GCN:
        model = GCNPointCloudAE(num_points=num_points, latent_dim=latent_dim).to(device)
    elif model_type == TRANSFORMER:
        model = PointTransformerAE(num_points=num_points, latent_dim=latent_dim).to(device)
    else:
        model = PointCloudAE(num_points=num_points, latent_dim=latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Training on {device}\n")

    for epoch in range(epochs):

        # =====================
        # TRAIN
        # =====================
        model.train()
        train_loss = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]", leave=False)

        for pts in train_bar:
            pts = pts.to(device)

            optimizer.zero_grad()

            recon = model(pts)
            loss = chamfer_distance(recon, pts)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # =====================
        # VALIDATION
        # =====================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]", leave=False)

            for pts in val_bar:
                pts = pts.to(device)

                recon = model(pts)
                loss = chamfer_distance(recon, pts)

                val_loss += loss.item()

                val_bar.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)

        # =====================
        # PRINT EPOCH SUMMARY
        # =====================
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'latent_dim': latent_dim,
        'num_points': num_points,
    }, f"checkpoint_{model_type}_{num_points}points_{latent_dim}latent_{epochs}epochs.pth")

    return model


GCN = "GCN"
TRANSFORMER = "TRANSFORMER"

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # PCD_FOLDER = r"E:\PAIR360\Traversal2\1st_Dormitory\Sequence0\T2-1st_Dormitory-0-pcd\pcd"

    FOLDERS = [
        r"E:\PAIR360\Traversal2\1st_Dormitory",
        r"E:\PAIR360\Traversal2\2nd_dormitory",
        r"E:\PAIR360\Traversal2\Central_Library",
    ]

    EPOCHS = 35
    NUM_POINTS_ARRAY = [4096]
    LATENT_DIM_ARRAY = [128, 256]

    for num_points in NUM_POINTS_ARRAY:
        for latent_dim in LATENT_DIM_ARRAY:

            model = train(
                folders=FOLDERS,
                epochs=EPOCHS,
                batch_size=8,
                num_points=num_points,
                model_type=GCN,
                latent_dim=latent_dim
            )
