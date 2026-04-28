import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# =========================
# CONFIG
# =========================
OUTPUT_DIR = "dataset_chunks"
BATCH_SIZE = 16
EPOCHS = 1000
LR = 1e-4
TRAIN_RATIO = 0.8
CNN = "CNN"
CNN_TRANSFORMER = "CNN_TRANSFORMER"
MODEL_TYPES = [CNN, CNN_TRANSFORMER]


# =========================
# DATASET
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

        worker = get_worker_info()
        _ = worker.id if worker else 0

        chunk_id, sample_id = self.index[idx]

        if chunk_id not in self.cache:
            self.cache[chunk_id] = torch.load(
                self.files[chunk_id],
                map_location="cpu"
            )

        x, y = self.cache[chunk_id]
        return x[sample_id], y[sample_id]


# =========================
# MODEL
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


# =========================
# CNN + TRANSFORMER HYBRID
# =========================
class CNNTransformer(nn.Module):

    def __init__(
            self,
            in_channels=7,
            embed_dim=128,  # ↓ réduit pour accélérer
            num_heads=4,  # ↓ moins de coût attention
            num_layers=2,  # ↓ moins profond
            latent_dim=256
    ):
        super().__init__()

        # =========================
        # CNN (réduction + downsampling fort)
        # =========================
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # stride plus agressif → moins de tokens
            nn.Conv2d(64, embed_dim, 3, 4, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

        # =========================
        # TRANSFORMER (léger et rapide)
        # =========================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,  # ↓ plus léger
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # =========================
        # HEAD
        # =========================
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, latent_dim)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        # CNN feature extraction
        x = self.cnn(x)  # (B, E, H', W')

        b, e, h, w = x.shape

        # Flatten spatial → tokens
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)

        # ⚠️ sécurité perf : limite nombre de tokens
        if x.size(1) > 64:
            x = x[:, :64, :]

        # Transformer
        x = self.transformer(x)

        # pooling global
        x = x.mean(dim=1)

        # projection finale
        x = self.fc(x)

        return x

# =========================
# VALIDATION LOOP
# =========================
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        loop = tqdm(loader, desc="Validation", leave=False)

        for x, y in loop:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            loop.set_postfix(val_loss=loss.item())

    return total_loss / len(loader)


# =========================
# TRAIN LOOP
# =========================
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChunkedDataset(OUTPUT_DIR)

    # Split train / val
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=4,
        pin_memory=True,
        # persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # num_workers=4,
        pin_memory=True,
        # persistent_workers=True
    )

    for model_type in MODEL_TYPES:

        if model_type == CNN_TRANSFORMER:
            model = CNNTransformer(in_channels=7).to(device)
        if model_type == CNN:
            model = CNNRegressor(in_channels=7).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.SmoothL1Loss()

        writer = SummaryWriter(log_dir="runs/exp1")

        best_val_loss = float("inf")

        for epoch in range(EPOCHS):

            # =========================
            # TRAIN
            # =========================
            model.train()
            train_loss = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

            for x, y in loop:

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pred = model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                loop.set_postfix(train_loss=loss.item())

            avg_train_loss = train_loss / len(train_loader)

            # =========================
            # VALIDATION
            # =========================
            avg_val_loss = evaluate(model, val_loader, loss_fn, device)

            # =========================
            # LOGS (TensorBoard)
            # =========================
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)

            print(f"\nEpoch {epoch+1}: train={avg_train_loss:.4f} | val={avg_val_loss:.4f}")

            # =========================
            # SAVE BEST MODEL
            # =========================
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss
                }, f"checkpoints/{model_type}_checkpoint_best.pt")

            if (epoch + 1) % 100 == 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss
                }, f"checkpoints/{model_type}_checkpoint_{epoch + 1}.pt")

    writer.close()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()
