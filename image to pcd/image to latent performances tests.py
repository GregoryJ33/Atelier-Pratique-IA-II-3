import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 16
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATASET (identique)
# =========================
class ChunkedDataset(torch.utils.data.Dataset):

    def __init__(self, chunk_dir):
        self.files = sorted([
            os.path.join(chunk_dir, f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".pt")
        ])

        self.index = []
        for i, f in enumerate(self.files):
            x, y = torch.load(f, map_location="cpu")
            for j in range(len(x)):
                self.index.append((i, j))

        self.cache = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        chunk_id, sample_id = self.index[idx]

        if chunk_id not in self.cache:
            self.cache[chunk_id] = torch.load(
                self.files[chunk_id],
                map_location="cpu"
            )

        x, y = self.cache[chunk_id]
        return x[sample_id], y[sample_id]


# =========================
# MODELS (identiques)
# =========================
class CNNRegressor(nn.Module):
    def __init__(self, in_channels=7, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
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
        return self.fc(x)


class CNNTransformer(nn.Module):
    def __init__(self, in_channels=7, embed_dim=128, num_heads=4, num_layers=2, latent_dim=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, embed_dim, 3, 4, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, latent_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        b, e, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)

        if x.size(1) > 64:
            x = x[:, :64, :]

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)

        return x


# =========================
# EVALUATION
# =========================
def evaluate(model, loader):
    model.eval()

    smoothl1_fn = nn.SmoothL1Loss()
    mse_fn = nn.MSELoss()
    cos_fn = nn.CosineSimilarity(dim=1)

    total_smoothl1 = 0
    total_mse = 0
    total_l2 = 0
    total_cos = 0

    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating", leave=False)

        for x, y in loop:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            smoothl1 = smoothl1_fn(pred, y)
            mse = mse_fn(pred, y)

            l2 = torch.norm(pred - y, dim=1).mean()
            cos = cos_fn(pred, y).mean()

            total_smoothl1 += smoothl1.item()
            total_mse += mse.item()
            total_l2 += l2.item()
            total_cos += cos.item()

    n = len(loader)

    return {
        "smoothl1": total_smoothl1 / n,
        "mse": total_mse / n,
        "l2": total_l2 / n,
        "cosine": total_cos / n
    }

# =========================
# LOAD MODEL FROM CHECKPOINT
# =========================
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if "CNN_TRANSFORMER" in checkpoint_path:
        model = CNNTransformer().to(DEVICE)
    else:
        model = CNNRegressor().to(DEVICE)

    model.load_state_dict(checkpoint["model"])

    epoch = checkpoint.get("epoch", -1)

    return model, epoch


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
            df = df.head(100)

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


def save_results(results, filepath="results.txt"):
    with open(filepath, "w") as f:
        f.write("=== FINAL RANKING ===\n\n")

        for i, (ckpt, epoch, m) in enumerate(results):
            line = (
                f"{i+1}. [Epoch {epoch}] {ckpt} → "
                f"S1: {m['smoothl1']:.6f} | "
                f"MSE: {m['mse']:.6f} | "
                f"L2: {m['l2']:.6f} | "
                f"COS: {m['cosine']:.6f}\n"
            )
            f.write(line)


# =========================
# MAIN
# =========================
def main():

    # dataset = ChunkedDataset(OUTPUT_DIR)
    #
    # train_size = int(TRAIN_RATIO * len(dataset))
    # val_size = len(dataset) - train_size
    #
    # _, val_ds = random_split(dataset, [train_size, val_size])
    #
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     pin_memory=True
    # )

    root_datasets = [
        r"E:\PAIR360\Traversal2\College_of_Physical_Education\Sequence0"
    ]

    # Trouver toutes les sequences automatiquement
    sequence_dirs = find_all_sequences(root_datasets)

    print(f"{len(sequence_dirs)} sequences trouvées")

    IMG_SIZE = (768, 384)

    dataset = Pair360Dataset(sequence_dirs, IMG_SIZE)

    val_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        # prefetch_factor=4
    )

    results = []

    checkpoints = [
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith(".pt")
    ]

    print(f"\nFound {len(checkpoints)} checkpoints\n")

    for ckpt in checkpoints:
        print(f"Testing: {ckpt}")

        model, epoch = load_model(ckpt)

        metrics = evaluate(model, val_loader)

        results.append((ckpt, epoch, metrics))

        print(
            f"[Epoch {epoch}] → "
            f"→ SmoothL1: {metrics['smoothl1']:.6f} | "
            f"MSE: {metrics['mse']:.6f} | "
            f"L2: {metrics['l2']:.6f} | "
            f"Cosine: {metrics['cosine']:.6f}\n"
        )

    # =========================
    # SORT RESULTS
    # =========================
    results.sort(key=lambda x: -x[2]["cosine"])

    print("\n=== FINAL RANKING ===")
    for i, (ckpt, epoch, m) in enumerate(results):
        print(
            f"{i+1}. [Epoch {epoch}] {ckpt} → "
            f"{i + 1}. {ckpt} → "
            f"S1: {m['smoothl1']:.6f} | "
            f"MSE: {m['mse']:.6f} | "
            f"L2: {m['l2']:.6f} | "
            f"COS: {m['cosine']:.6f}"
        )

    # save_results(results, "evaluation_results.txt")

    # print("\nRésultats sauvegardés dans evaluation_results.txt")


if __name__ == "__main__":
    main()
