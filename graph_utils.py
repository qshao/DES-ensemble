import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

# Fixed size (Small organics + Virtual Node)
MAX_ATOMS = 50 

def smiles_to_graph(smiles):
    """
    Featurizes a SMILES string into a graph with a VIRTUAL NODE.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(MAX_ATOMS, 10), torch.zeros(MAX_ATOMS, MAX_ATOMS), torch.zeros(MAX_ATOMS)

    # 1. Basic Adjacency (Covalent Bonds)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    num_real_atoms = adj.shape[0]
    
    if num_real_atoms > MAX_ATOMS - 1: # Reserve 1 spot for Virtual Node
        num_real_atoms = MAX_ATOMS - 1
        adj = adj[:num_real_atoms, :num_real_atoms]

    # 2. Node Features for Real Atoms
    # New Dim = 10 (Added 'is_virtual' flag)
    atom_feats = []
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= num_real_atoms: break
        
        symbol = atom.GetSymbol()
        is_donor = 1 if symbol in ['N', 'O', 'F'] and atom.GetTotalNumHs() > 0 else 0
        is_acceptor = 1 if symbol in ['N', 'O', 'F'] else 0
        
        feats = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),     # <--- Critical for Ions
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetMass() * 0.01,
            is_donor,
            is_acceptor,
            atom.GetTotalValence(),
            0.0 # is_virtual = 0
        ]
        atom_feats.append(feats)
    
    # 3. Create Virtual Node Feature
    # [0, 0, 0, ..., 1]
    virtual_feat = [0.0] * 9 + [1.0] 
    
    # 4. Construct Tensors
    # Node Features
    x_padded = torch.zeros(MAX_ATOMS, 10)
    x_padded[:num_real_atoms, :] = torch.tensor(atom_feats, dtype=torch.float32)
    x_padded[num_real_atoms, :] = torch.tensor(virtual_feat, dtype=torch.float32) # Add Virtual Node
    
    # Adjacency Matrix
    adj_padded = torch.zeros(MAX_ATOMS, MAX_ATOMS)
    # Copy real bonds
    adj_padded[:num_real_atoms, :num_real_atoms] = torch.tensor(adj, dtype=torch.float32)
    
    # Connect Virtual Node (Index: num_real_atoms) to ALL real atoms
    # Bidirectional connection
    adj_padded[num_real_atoms, :num_real_atoms] = 1.0 
    adj_padded[:num_real_atoms, num_real_atoms] = 1.0
    
    # Add Self-Loops (Real + Virtual)
    idx = torch.arange(num_real_atoms + 1)
    adj_padded[idx, idx] = 1.0
    
    # Mask (Real + Virtual are valid)
    mask = torch.zeros(MAX_ATOMS)
    mask[:num_real_atoms + 1] = 1.0
    
    return x_padded, adj_padded, mask
