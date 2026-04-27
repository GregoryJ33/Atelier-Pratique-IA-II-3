import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import open3d as o3d
import glob
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch_geometric.nn import EdgeConv, global_max_pool

# ==========================================
# --- CONSTANTES ET CONFIGURATION ---
# ==========================================

IMG_H, IMG_W = 512, 1024
DEPTH_MAX = 77.0
N_PCD_POINTS = 1024
EMBED_DIM = 256
N_SEG_CLASSES = 20
BATCH_SIZE = 4
EPOCHS = 10

# ==========================================
# --- UTILITAIRES LIDAR (PCD) ---
# ==========================================

def read_pcd_xyz(path, n=N_PCD_POINTS):
    try:
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points).astype(np.float32)
    except Exception:
        return np.zeros((n, 3), dtype=np.float32)
    
    if len(pts) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    
    if len(pts) >= n:
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx]
    
    reps = (n // len(pts)) + 1
    return np.tile(pts, (reps, 1))[:n]


def normalize_pcd(pts):
    p5 = np.percentile(pts, 5, axis=0)
    p95 = np.percentile(pts, 95, axis=0)
    rng = (p95 - p5).clip(min=1e-6)
    return (pts - p5) / rng * 2 - 1 


# ==========================================
# --- GESTION DES DONNÉES (DATASET) ---
# ==========================================

class StitchingDepthDataset(Dataset):
    def __init__(self, csv_path, stitch_dir, depth_dir, seg_dir, pcd_dir):
        super().__init__()
        df = pd.read_csv(csv_path)
        
        self.stitch_dir = Path(stitch_dir)
        self.depth_dir = Path(depth_dir)
        self.seg_dir = Path(seg_dir)
        self.pcd_dir = Path(pcd_dir)
        
        self.samples = []
        for _, row in df.iterrows():
            num = row['image_name']
            pcd_n = row['pcd_name']
            
            s_p = self.stitch_dir / num
            d_p = self.depth_dir / num
            sg_p = self.seg_dir / num
            p_p = self.pcd_dir / pcd_n
            
            if all(path.exists() for path in [s_p, d_p, sg_p, p_p]):
                self.samples.append({
                    'stitch': s_p, 
                    'depth': d_p, 
                    'seg': sg_p, 
                    'pcd': p_p
                })
        
        self.img_tf = T.Compose([
            T.Resize((IMG_H, IMG_W)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Image stitching
        img = Image.open(s['stitch']).convert('RGB')
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
        
        # Profondeur réelle (normalisée)
        depth_img = Image.open(s['depth']).convert('L')
        depth_img = depth_img.resize((IMG_W, IMG_H), Image.NEAREST)
        depth_raw = np.array(depth_img, dtype=np.float32) / DEPTH_MAX
        
        # Segmentation
        seg_img = Image.open(s['seg']).resize((IMG_W, IMG_H), Image.NEAREST)
        seg_raw = np.array(seg_img)
        if seg_raw.ndim == 3:
            seg_raw = seg_raw[:, :, 0]
        
        # Nuage de points LiDAR
        pts_raw = read_pcd_xyz(s['pcd'])
        pts_norm = normalize_pcd(pts_raw)
        
        return (
            self.img_tf(img),
            torch.from_numpy(depth_raw).unsqueeze(0).float(),
            torch.from_numpy(seg_raw.astype(np.int64)),
            torch.from_numpy(pts_norm).float()
        )


# ==========================================
# --- ARCHITECTURES DES RÉSEAUX ---
# ==========================================

class GCNEncoder(nn.Module):
    """Encodeur pour le nuage de points basé sur les graphes (EdgeConv)."""
    def __init__(self, out_dim=EMBED_DIM, k=16):
        super().__init__()
        self.k = k
        self.mlp_edge = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.conv = EdgeConv(self.mlp_edge, aggr='max')
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, pts):
        batch_size, n_pts, _ = pts.shape
        pts_flat = pts.view(-1, 3)
        
        # Création des index de batch pour PyTorch Geometric
        batch_idx = torch.arange(batch_size, device=pts.device)
        batch_idx = batch_idx.repeat_interleave(n_pts)
        
        from torch_cluster import knn_graph
        edge_index = knn_graph(pts_flat, k=self.k, batch=batch_idx)
        
        x = self.conv(pts_flat, edge_index)
        x = global_max_pool(x, batch_idx)
        return self.fc(x)


class UpBlock(nn.Module):
    """Bloc de décodage pour l'U-Net."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class StitchingToDepth(nn.Module):
    """Modèle principal d'estimation de profondeur à partir d'images stitching."""
    def __init__(self):
        super().__init__()
        res = models.resnet18(weights='DEFAULT')
        
        # Encodeur (ResNet18 partagé)
        self.enc0 = nn.Sequential(res.conv1, res.bn1, res.relu)
        self.pool = res.maxpool
        self.enc1 = res.layer1
        self.enc2 = res.layer2
        self.enc3 = res.layer3
        self.enc4 = res.layer4
        
        # Bottleneck / Latent space
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.ReLU()
        )
        
        # Décodeur U-Net
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        
        # Têtes de sortie
        self.depth_head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, N_SEG_CLASSES, 1)
        )

    def forward(self, x):
        # Encodage
        s0 = self.enc0(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        
        img_emb = self.bottleneck(s4)
        
        # Décodage
        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        
        # Upsampling final pour match de résolution
        d_out = F.interpolate(self.depth_head(d), (IMG_H, IMG_W), mode='bilinear')
        s_out = F.interpolate(self.seg_head(d), (IMG_H, IMG_W), mode='bilinear')
        
        return d_out, s_out, img_emb


# ==========================================
# --- VISUALISATION ET MÉTRIQUES ---
# ==========================================

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))
    
    # Courbes des pertes (Train)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Total Loss', color='black', lw=2)
    plt.plot(epochs, history['loss_depth'], label='Depth Loss', alpha=0.6)
    plt.plot(epochs, history['loss_seg'], label='Seg Loss', alpha=0.6)
    plt.plot(epochs, history['loss_emb'], label='GCN Alignment', alpha=0.6)
    plt.title("Evolution des Losses (Train)")
    plt.xlabel("Epoch")
    plt.legend()
    
    # Métrique de Validation
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_mae'], color='red', lw=2, label='Val MAE')
    plt.title("Erreur Moyenne (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Mètres")
    plt.legend()
    
    plt.savefig("training_summary.png")
    plt.show()


def visualize_results(model, dataset, device, epoch):
    Path("results").mkdir(exist_ok=True)
    model.eval()
    
    # 1. Sélection d'un échantillon aléatoire
    sample_idx = np.random.randint(len(dataset))
    img_t, depth_t, seg_t, _ = dataset[sample_idx]
    
    with torch.no_grad():
        d_p, s_p, _ = model(img_t.unsqueeze(0).to(device))
    
    # 2. Préparation des données (mètres et numpy)
    d_p_np = d_p[0, 0].cpu().numpy() * DEPTH_MAX
    d_t_np = depth_t[0].numpy() * DEPTH_MAX
    error_map = np.abs(d_p_np - d_t_np)
    
    # Dénormalisation de l'image stitching (RGB)
    img_np = img_t.permute(1, 2, 0).numpy()
    img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img_np = np.clip(img_np, 0, 1)

    # 3. Création de la figure
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.patch.set_facecolor('white')
    plt.suptitle(f"ÉVALUATION MODÈLE - ÉPOQUE {epoch+1}\n(Entrée: Image 360° | Sorties: Profondeur & Segmentation)", 
                 fontsize=18, fontweight='bold', y=0.98)

    # --- LIGNE 1 : PROFONDEUR ---
    
    # Image Stitching
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("1. Image Stitching (Entrée)", fontsize=14, pad=10)
    axes[0, 0].axis('off')

    # Profondeur Réelle (GT)
    im1 = axes[0, 1].imshow(d_t_np, cmap='plasma', vmin=0, vmax=DEPTH_MAX)
    axes[0, 1].set_title(f"2. Profondeur Réelle (LiDAR)\nEchelle: 0m - {DEPTH_MAX}m", fontsize=14, pad=10)
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label('Distance (mètres)', rotation=270, labelpad=15)
    axes[0, 1].axis('off')

    # Profondeur Prédite
    im2 = axes[0, 2].imshow(d_p_np, cmap='plasma', vmin=0, vmax=DEPTH_MAX)
    axes[0, 2].set_title(f"3. Profondeur Prédite (Réseau)\nEchelle: 0m - {DEPTH_MAX}m", fontsize=14, pad=10)
    cbar2 = fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar2.set_label('Distance (mètres)', rotation=270, labelpad=15)
    axes[0, 2].axis('off')

    # --- LIGNE 2 : ERREUR ET SEGMENTATION ---

    # Erreur Absolue
    im3 = axes[1, 0].imshow(error_map, cmap='hot', vmin=0, vmax=10) # Max 10m pour le contraste
    axes[1, 0].set_title("4. Erreur Absolue |abs(Prédit - Réel)|\nEchelle: 0m (noir) à 10m+ (blanc)", fontsize=14, pad=10)
    cbar3 = fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar3.set_label('Erreur (mètres)', rotation=270, labelpad=15)
    axes[1, 0].axis('off')

    # Segmentation Réelle
    axes[1, 1].imshow(seg_t, cmap='tab20', vmin=0, vmax=N_SEG_CLASSES)
    axes[1, 1].set_title("5. Segmentation Sémantique (Réelle)", fontsize=14, pad=10)
    axes[1, 1].axis('off')

    # Segmentation Prédite
    pred_seg = s_p[0].argmax(0).cpu().numpy()
    axes[1, 2].imshow(pred_seg, cmap='tab20', vmin=0, vmax=N_SEG_CLASSES)
    axes[1, 2].set_title("6. Segmentation Sémantique (Prédite)", fontsize=14, pad=10)
    axes[1, 2].axis('off')

    # 4. Finalisation et sauvegarde
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(f"results/epoch_{epoch+1}.png", dpi=150, bbox_inches='tight')
    plt.close()


# ==========================================
# --- BOUCLE D'ENTRAÎNEMENT ---
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StitchingToDepth().to(device)
lidar_enc = GCNEncoder().to(device)

optimizer = optim.AdamW(
    list(model.parameters()) + list(lidar_enc.parameters()), 
    lr=1e-4
)
cosine_crit = nn.CosineEmbeddingLoss()

if glob.glob("CAM.csv"):
    full_ds = StitchingDepthDataset(
        "CAM.csv", "dataset/stitching", "dataset/depth", "dataset/segmentation", "dataset/pcd"
    )
    
    # Séparation Train (90%) / Val (10%)
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 3e-4, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )
    
    history = {
        'train_loss': [], 'loss_depth': [], 
        'loss_seg': [], 'loss_emb': [], 'val_mae': []
    }

    print(f"Lancement de l'entraînement sur {device}...")

    for epoch in range(EPOCHS):
        model.train()
        lidar_enc.train()
        
        t_loss, t_depth, t_seg, t_emb = 0, 0, 0, 0
        
        for img, depth, seg, pts in train_loader:
            img, depth, seg, pts = img.to(device), depth.to(device), seg.to(device), pts.to(device)
            
            optimizer.zero_grad()
            
            # Prédictions
            d_p, s_p, i_e = model(img)
            l_e = lidar_enc(pts)
            
            # Calcul des pertes unitaires
            ld = (d_p - depth).abs()[depth > 0].mean()
            ls = F.cross_entropy(s_p, seg, ignore_index=255)
            le = cosine_crit(i_e, l_e, torch.ones(img.size(0), device=device))
            
            # Somme pondérée
            loss = ld + 0.3 * ls + 0.2 * le
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            t_loss += loss.item()
            t_depth += ld.item()
            t_seg += ls.item()
            t_emb += le.item()

        # Phase d'évaluation (Validation)
        model.eval()
        v_mae = 0
        with torch.no_grad():
            for img, depth, _, _ in val_loader:
                d_p, _, _ = model(img.to(device))
                error = torch.abs(d_p.cpu() - depth) * DEPTH_MAX
                v_mae += error[depth > 0].mean().item()
        
        # Enregistrement historique
        n = len(train_loader)
        history['train_loss'].append(t_loss / n)
        history['loss_depth'].append(t_depth / n)
        history['loss_seg'].append(t_seg / n)
        history['loss_emb'].append(t_emb / n)
        history['val_mae'].append(v_mae / len(val_loader))
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {t_loss/n:.4f} | Val MAE: {v_mae/len(val_loader):.3f}m")
        
        # Génération d'une image de validation
        visualize_results(model, val_ds, device, epoch)

    # Fin d'entraînement
    plot_training_history(history)
    torch.save(model.state_dict(), "model_final.pt")
    print("Modèle sauvegardé sous model_final.pt")