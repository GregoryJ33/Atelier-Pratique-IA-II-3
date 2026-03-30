import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_cluster import knn_graph
import glob
import numpy as np

class LidarGraphDataset(Dataset):
    def __init__(self, file_paths, target_paths):
        super().__init__()
        self.file_paths = file_paths
        self.target_paths = target_paths

    def len(self):
        return len(self.file_paths)
    
    def get(self, idx):
        data_np = np.load(self.file_paths[idx])
        pos = torch.from_numpy(data_np).float()

        edge_index = knn_graph(pos, k=16)

        target_np = np.load(self.target_paths[idx])
        y = torch.from_numpy(target_np).float()

        return Data(x=pos, edge_index=edge_index, y=y)


class GlobalToLidar(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        self.input_layer = nn.Linear(3, 128)

        #Encoder --> Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        #Projecteur pour avec une distance initiale
        self.initial_depth = nn.Linear(embed_dim,1)

        #Correction de la distance en fonction des voisins
        self.graph_conv = GCNConv(in_channels=1, out_channels=32)
        self.final_depth = nn.Linear(32,1)

    def forward(self, x, edge_index):
        # x est (N, 3)
        x = torch.relu(self.input_layer(x)) # (N, 128)

        # On ajoute une dimension pour le Transformer : (N, 1, 128)
        x = x.unsqueeze(1) 

        features = self.encoder(x)

        # On retire la dimension : (N, 128)
        features = features.squeeze(1)

        z_initial = torch.relu(self.initial_depth(features))

        z_refined = torch.relu(self.graph_conv(z_initial, edge_index))

        return self.final_depth(z_refined)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GlobalToLidar(embed_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()
epochs = 50

list_x = glob.glob("/chemin/dataset_x.*npy") #glob.glob --> pour transformer le chemin de dossier en une liste de fichiers utilisable
list_y = glob.glob("/chemin/dataset_y.*npy")

if len(list_x) > 0:
    dataset = LidarGraphDataset(list_x, list_y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)

        #Remise à 0 des gradients
        optimizer.zero_grad()

        #Forward
        output = model(batch.x, batch.edge_index)

        #Loss
        loss = criterion(output, batch.y)

        #Backpropagation
        loss.backward()

        #Mise à jour des poids
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")