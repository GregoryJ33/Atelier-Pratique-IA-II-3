import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm


# =========================
# DATASET
# =========================

class Pair360Dataset(Dataset):
    def __init__(self, roots, img_size, transform=None, verbose=True):
        """
        roots: liste des dossiers Sequence (ex: .../Sequence0)
        """
        self.samples = []
        self.transform = transform
        self.img_size = img_size

        for root in roots:
            cam_file = os.path.join(root, "CAM.csv")
            if not os.path.exists(cam_file):
                continue

            df = pd.read_csv(cam_file)

            # Trouver les dossiers automatiquement
            stitching_dir = None
            depth_dir = None
            seg_dir = None
            latent_dir = None

            for d in os.listdir(root):
                full = os.path.join(root, d)

                if "stitching" in d:
                    stitching_dir = os.path.join(full, "stitching")
                elif "depth" in d:
                    depth_dir = os.path.join(full, "depth")
                elif "segmentation" in d:
                    seg_dir = os.path.join(full, "segmentation")
                elif "latents" in d:
                    latent_dir = full

            # Vérification
            if not all([stitching_dir, depth_dir, seg_dir, latent_dir]):
                if verbose:
                    print(f"[WARNING] Dossiers manquants dans {root}")
                continue

            # Construire les samples
            for _, row in df.iterrows():
                img_name = row["image_name"]
                pcd_name = row["pcd_name"]

                # Tentative simple
                latent_name = pcd_name.replace(".pcd", ".npy")
                latent_path = os.path.join(latent_dir, latent_name)

                if not os.path.exists(latent_path):
                    continue  # skip si pas de correspondance

                stitch_path = os.path.join(stitching_dir, img_name)
                depth_path = os.path.join(depth_dir, img_name)
                seg_path = os.path.join(seg_dir, img_name)

                if not (os.path.exists(stitch_path) and
                        os.path.exists(depth_path) and
                        os.path.exists(seg_path)):
                    continue

                self.samples.append({
                    "stitch": stitch_path,
                    "depth": depth_path,
                    "seg": seg_path,
                    "latent": latent_path
                })

        if verbose:
            print(f"Dataset chargé: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # # Charger images
        # img_stitch = np.array(Image.open(s["stitch"]))
        # img_depth = np.array(Image.open(s["depth"]))
        # img_seg = np.array(Image.open(s["seg"]))

        # Charger images au format demandé
        img_stitch = Image.open(s["stitch"]).resize(self.img_size, Image.BILINEAR)
        img_depth = Image.open(s["depth"]).resize(self.img_size, Image.BILINEAR)
        img_seg = Image.open(s["seg"]).resize(self.img_size, Image.NEAREST)

        img_stitch = np.array(img_stitch)
        img_depth = np.array(img_depth)
        img_seg = np.array(img_seg)

        # Si depth ou seg sont en 2D → ajouter channel
        if len(img_depth.shape) == 2:
            img_depth = np.expand_dims(img_depth, axis=2)

        if len(img_seg.shape) == 2:
            img_seg = np.expand_dims(img_seg, axis=2)

        # Normalisation
        img_stitch = img_stitch.astype(np.float32) / 255.0
        img_depth = img_depth.astype(np.float32) / 255.0
        img_seg = img_seg.astype(np.float32) / 255.0

        # Stack channels
        x = np.concatenate([img_stitch, img_depth, img_seg], axis=2)
        x = torch.from_numpy(x).permute(2, 0, 1)  # C,H,W

        # Charger latent
        y = np.load(s["latent"])
        y = torch.from_numpy(y).float()

        return x, y

# class Pair360Dataset(Dataset):
#     def __init__(self, roots, img_size, transform=None, verbose=True):
#         self.data = []
#         self.img_size = img_size
#         self.transform = transform
#
#         for root in tqdm(roots, desc="Scanning datasets"):
#             cam_file = os.path.join(root, "CAM.csv")
#             if not os.path.exists(cam_file):
#                 continue
#
#             df = pd.read_csv(cam_file)
#
#             stitching_dir = None
#             depth_dir = None
#             seg_dir = None
#             latent_dir = None
#
#             for d in os.listdir(root):
#                 full = os.path.join(root, d)
#
#                 if "stitching" in d:
#                     stitching_dir = os.path.join(full, "stitching")
#                 elif "depth" in d:
#                     depth_dir = os.path.join(full, "depth")
#                 elif "segmentation" in d:
#                     seg_dir = os.path.join(full, "segmentation")
#                 elif "latents" in d:
#                     latent_dir = full
#
#             if not all([stitching_dir, depth_dir, seg_dir, latent_dir]):
#                 if verbose:
#                     print(f"[WARNING] Dossiers manquants dans {root}")
#                 continue
#
#             for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {os.path.basename(root)}"):
#                 img_name = row["image_name"]
#                 pcd_name = row["pcd_name"]
#
#                 latent_name = pcd_name.replace(".pcd", ".npy")
#                 latent_path = os.path.join(latent_dir, latent_name)
#
#                 if not os.path.exists(latent_path):
#                     continue
#
#                 stitch_path = os.path.join(stitching_dir, img_name)
#                 depth_path = os.path.join(depth_dir, img_name)
#                 seg_path = os.path.join(seg_dir, img_name)
#
#                 if not (os.path.exists(stitch_path) and
#                         os.path.exists(depth_path) and
#                         os.path.exists(seg_path)):
#                     continue
#
#                 # =========================
#                 # CHARGEMENT IMMÉDIAT (clé de l’optimisation)
#                 # =========================
#
#                 try:
#                     img_stitch = Image.open(stitch_path).resize(self.img_size, Image.BILINEAR)
#                     img_depth = Image.open(depth_path).resize(self.img_size, Image.BILINEAR)
#                     img_seg = Image.open(seg_path).resize(self.img_size, Image.NEAREST)
#
#                     img_stitch = np.array(img_stitch, dtype=np.float32) / 255.0
#                     img_depth = np.array(img_depth, dtype=np.float32) / 255.0
#                     img_seg = np.array(img_seg, dtype=np.float32) / 255.0
#
#                     # garantir channel dimension
#                     if img_depth.ndim == 2:
#                         img_depth = img_depth[..., None]
#                     if img_seg.ndim == 2:
#                         img_seg = img_seg[..., None]
#
#                     x = np.concatenate([img_stitch, img_depth, img_seg], axis=2)
#                     x = torch.from_numpy(x).permute(2, 0, 1)
#
#                     y = np.load(latent_path)
#                     y = torch.from_numpy(y).float()
#
#                     self.data.append((x, y))
#
#                 except Exception as e:
#                     if verbose:
#                         print(f"[ERROR] {stitch_path}: {e}")
#
#         if verbose:
#             print(f"Dataset prêt: {len(self.data)} samples chargés en RAM")
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]

# =========================
# FONCTION POUR TROUVER TOUTES LES SEQUENCES
# =========================

def find_all_sequences(root_directories):
    sequences = []

    for root_dir in root_directories:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if "CAM.csv" in filenames:
                sequences.append(dirpath)

    return sequences


class CNNRegressor(nn.Module):
    def __init__(self, in_channels=5, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # /4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),          # /8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),         # /16
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


# =========================
# UTILISATION
# =========================

if __name__ == "__main__":

    root_datasets = [
        r"E:\PAIR360\Traversal2\College_of_Engineering",
        r"E:\PAIR360\Traversal2\College_of_Life_Science"
    ]

    # Trouver toutes les sequences automatiquement
    sequence_dirs = find_all_sequences(root_datasets)

    print(f"{len(sequence_dirs)} sequences trouvées")

    IMG_SIZE = (768, 384)

    dataset = Pair360Dataset(sequence_dirs, IMG_SIZE)

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        # prefetch_factor=4
    )

    # Test rapide
    for x, y in dataloader:
        print("X:", x.shape)
        print("y:", y.shape)
        break

    model = CNNRegressor(in_channels=7).cuda()  # ou ViTRegressor
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()

    EPOCHS = 20

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for x, y in loop:
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} mean loss: {running_loss / len(dataloader):.4f}")
