import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from graph_utils import smiles_to_graph

# Physical Constants
R_GAS = 8.314 
WALDEN_DS = 56.5 

# DES Mapping
DES_MAPPING = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, 'IL': 5}
UNKNOWN_DES_IDX = 6
NUM_DES_TYPES = 7

class GraphMixtureDataset(Dataset):
    def __init__(self, dataframe, target_scale=1000.0):
        self.df = dataframe.reset_index(drop=True)
        self.target_scale = target_scale
        
        if 'Type of DES' in self.df.columns:
            self.df['Type of DES'] = self.df['Type of DES'].astype(str)
            
        print("Featurizing graphs (this may take a moment)...")
        self.graphs1 = [smiles_to_graph(s) for s in self.df['Smiles#1']]
        self.graphs2 = [smiles_to_graph(s) for s in self.df['Smiles#2']]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x1, adj1, mask1 = self.graphs1[idx]
        x2, adj2, mask2 = self.graphs2[idx]
        
        t1, t2 = float(row['T#1']), float(row['T#2'])
        frac1, frac2 = float(row['X#1 (molar fraction)']), float(row['X#2 (molar fraction)'])
        
        # Target: Scaled 1000/T
        target_scaled = 0.0
        if 'Tmelt, K' in row:
            target_scaled = (1.0 / float(row['Tmelt, K'])) * self.target_scale
            
        # Baselines (Thermal Knowledge)
        # 1. Linear
        t_lin = frac1*t1 + frac2*t2
        base_lin = (1.0 / t_lin) * self.target_scale
        
        # 2. Eutectic (Walden)
        sx1, sx2 = max(frac1, 1e-6), max(frac2, 1e-6)
        t_id1 = t1 / (1 - (R_GAS/WALDEN_DS)*np.log(sx1))
        t_id2 = t2 / (1 - (R_GAS/WALDEN_DS)*np.log(sx2))
        base_eut = (1.0 / max(t_id1, t_id2)) * self.target_scale
        
        # DES Type
        des_str = str(row.get('Type of DES', 'Unknown')).strip()
        des_idx = DES_MAPPING.get(des_str, UNKNOWN_DES_IDX)
        
        return {
            'x1': x1, 'adj1': adj1, 'mask1': mask1,
            'x2': x2, 'adj2': adj2, 'mask2': mask2,
            'frac1': torch.tensor(frac1, dtype=torch.float32), 
            'frac2': torch.tensor(frac2, dtype=torch.float32),
            'base_lin': torch.tensor(base_lin, dtype=torch.float32),
            'base_eut': torch.tensor(base_eut, dtype=torch.float32),
            'des_type': torch.tensor(des_idx, dtype=torch.long),
            'target': torch.tensor(target_scaled, dtype=torch.float32)
        }

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        # x: [batch, nodes, in_dim]
        # adj: [batch, nodes, nodes]
        out = torch.bmm(adj, x) # Graph Conv
        out = self.linear(out)
        return torch.relu(out)

class ThermoGraphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        node_dim = config['model']['node_dim']
        hidden_dim = config['model']['hidden_dim']
        out_dim = config['model']['output_dim']
        des_dim = config['model']['des_embed_dim']
        
        # 1. Graph Encoder
        self.gcn1 = GCNLayer(node_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, out_dim)
        
        # 2. DES Embedding
        self.des_emb = nn.Embedding(NUM_DES_TYPES, des_dim)
        
        # 3. Bilinear Interaction (The "Physics" Layer)
        # Input: GraphVec1 + GraphVec2
        self.bilinear = nn.Bilinear(out_dim, out_dim, 64)
        
        # 4. Interaction MLP
        # Input: BilinearOutput(64) + DES_Emb(32)
        self.inter_mlp = nn.Sequential(
            nn.Linear(64 + des_dim, 64),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(64, 1) # Outputs Interaction W
        )
        
        # 5. Baseline Gate
        # Input: Graph1 + Graph2 + DES_Emb
        gate_dim = out_dim * 2 + des_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(self, x, adj, mask):
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        h = self.gcn3(h, adj) # [batch, nodes, out_dim]
        
        # Masked Mean Pooling
        mask_exp = mask.unsqueeze(-1)
        sum_h = (h * mask_exp).sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return sum_h / count

    def forward(self, x1, adj1, mask1, x2, adj2, mask2, frac1, frac2, base_lin, base_eut, des_type):
        # Encode Molecules
        v1 = self.encode(x1, adj1, mask1)
        v2 = self.encode(x2, adj2, mask2)
        
        # Encode DES Type
        d_emb = self.des_emb(des_type)
        
        # Interaction
        inter_vec = torch.relu(self.bilinear(v1, v2))
        inter_in = torch.cat([inter_vec, d_emb], dim=1)
        W = self.inter_mlp(inter_in).squeeze(1)
        
        # Deviation
        deviation = frac1 * frac2 * W
        
        # Gate
        gate_in = torch.cat([v1, v2, d_emb], dim=1)
        alpha = self.gate_net(gate_in).squeeze(1)
        
        return (alpha * base_eut + (1 - alpha) * base_lin) + deviation
