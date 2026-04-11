import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import open3d as o3d
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from torch_geometric.nn import GCNConv, knn_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

class ImageToLidarDataset(Dataset):
    def __init__(self, csv_path, img_dir, pcd_dir):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.pcd_dir = pcd_dir
        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_name']
        pcd_name = self.df.iloc[idx]['pcd_name']
        
        # Image
        img = Image.open(f"{self.img_dir}/{img_name}").convert('RGB')
        img_tensor = self.img_transform(img)

        # LiDAR (Normalisation par 50m pour rester entre 0 et 1)
        pcd = o3d.io.read_point_cloud(f"{self.pcd_dir}/{pcd_name}")
        target_pts = np.asarray(pcd.points)
        
        if len(target_pts) == 0:
            target_pts = np.zeros((1000, 3))
        elif len(target_pts) > 1000:
            target_pts = target_pts[:1000]
            
        y = torch.from_numpy(target_pts).float() / 50.0 
        return img_tensor, y

class GlobalToLidar(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        resnet = models.resnet18(weights='DEFAULT')
        # On garde les couches jusqu'au bloc 256 channels
        self.backbone = nn.Sequential(*list(resnet.children())[:-3]) 
        self.adapter = nn.Linear(256, embed_dim) 

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.initial_depth = nn.Linear(embed_dim, 1)
        self.graph_conv = GCNConv(in_channels=1, out_channels=32)
        self.final_depth = nn.Linear(32, 1)

    def forward(self, img):
        feat = self.backbone(img) 
        # Forcer une grille de 16x16 (soit 256 points)
        feat = torch.nn.functional.interpolate(feat, size=(16, 16), mode='bilinear', align_corners=False)
        
        feat = feat.flatten(2).permute(0, 2, 1) 
        feat = torch.relu(self.adapter(feat))   
        context = self.transformer(feat)        

        z_init = torch.sigmoid(self.initial_depth(context)) 
        
        # GCN pour la cohérence locale
        z_init_flatten = z_init.view(-1, 1)
        features_flatten = context.view(-1, 128)
        edge_index = knn_graph(features_flatten, k=6)
        
        z_refined = torch.relu(self.graph_conv(z_init_flatten, edge_index))
        return torch.sigmoid(self.final_depth(z_refined))

def visualize_results(model, dataset, device, idx=0):
    model.eval()
    with torch.no_grad():
        img_tensor, target_pcd = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        
        # Prédiction (256 pts) -> Retour à l'échelle 0-50m
        pred_z = model(img_input).cpu().numpy().flatten() * 50.0 
        target_pcd_m = target_pcd.cpu().numpy() * 50.0

        # Projection inverse simplifiée (Caméra)
        u, v = np.meshgrid(np.linspace(-0.8, 0.8, 16), np.linspace(-0.6, 0.6, 16))
        u, v = u.flatten(), v.flatten()
        pred_x = u * pred_z
        pred_y = v * pred_z

        fig = plt.figure(figsize=(18, 6))
        
        # Image
        ax1 = fig.add_subplot(1, 3, 1)
        img_unnorm = img_tensor.permute(1, 2, 0).numpy() * 0.22 + 0.45
        ax1.imshow(np.clip(img_unnorm, 0, 1))
        ax1.set_title("Image d'entrée")
        ax1.axis('off')

        # Vrai LiDAR
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.scatter(target_pcd_m[:,0], target_pcd_m[:,1], target_pcd_m[:,2], s=1, alpha=0.5)
        ax2.set_xlim(-40, 40); ax2.set_ylim(-40, 40); ax2.set_zlim(0, 50)
        ax2.set_title("Vérité Terrain (Vrai LiDAR)")

        # Prédit
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.scatter(pred_x, pred_y, pred_z, c=pred_z, s=15, cmap='magma')
        ax3.set_xlim(-40, 40); ax3.set_ylim(-40, 40); ax3.set_zlim(0, 50)
        ax3.set_title("Pseudo-LiDAR (256 points)")
        
        plt.tight_layout()
        plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GlobalToLidar().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
criterion = nn.L1Loss()

CSV_PATH = "CAM.csv"
IMG_DIR = "T1-College_of_Engineering-2-fisheye/origin1"
PCD_DIR = "T1-College_of_Engineering-2-pcd/pcd"

if glob.glob(CSV_PATH):
    dataset = ImageToLidarDataset(CSV_PATH, IMG_DIR, PCD_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Entraînement lancé sur {device} pour 30 époques...")
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        for img, target_pcd in dataloader:
            img, target_pcd = img.to(device), target_pcd.to(device)
            optimizer.zero_grad()
            
            output_z = model(img)
            
            # Loss par tri (Histogram Matching)
            pred_sorted, _ = torch.sort(output_z.flatten())
            target_sorted, _ = torch.sort(target_pcd[0, :, 2]) # On prend le Z du target
            
            # On compare les 256 points les plus représentatifs
            loss = criterion(pred_sorted, target_sorted[:256])
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/30 | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Résultat final
    visualize_results(model, dataset, device, idx=np.random.randint(len(dataset)))
else:
    print("Erreur : Vérifiez les chemins des dossiers et du CSV.")