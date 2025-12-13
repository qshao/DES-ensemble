import torch
import pandas as pd
import numpy as np
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import yaml
from torch.utils.data import DataLoader
from gnn_model import ThermoGraphModel, GraphMixtureDataset, DES_MAPPING, UNKNOWN_DES_IDX

# Configuration
CONFIG_PATH = "config.yaml"
N_FOLDS = 5

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_phys_features(smiles):
    """Calculates tabular descriptors for the Booster."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            return [
                float(Descriptors.NumHDonors(mol)),
                float(Descriptors.NumHAcceptors(mol)),
                float(Descriptors.MolWt(mol) / 100.0),
                float(Descriptors.TPSA(mol) / 100.0),
                float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
                float(Descriptors.MolLogP(mol))
            ]
    except:
        pass
    return [0.0] * 6

def predict_ensemble(smiles1, t1, smiles2, t2, des_type='3'):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale = config['data']['target_scale']
    
    print(f"\nPredicting System: {smiles1} + {smiles2}")
    print(f"Loading {N_FOLDS}-Fold Ensemble on {device}...")

    # 1. Prepare Data Grid (0.0 to 1.0)
    x_range = np.linspace(0.0, 1.0, 11)
    data_records = []
    for x1 in x_range:
        data_records.append({
            'Smiles#1': smiles1, 'T#1': t1,
            'Smiles#2': smiles2, 'T#2': t2,
            'X#1 (molar fraction)': x1,
            'X#2 (molar fraction)': 1.0 - x1,
            'Type of DES': des_type,
            'Tmelt, K': 300.0 
        })
    df = pd.DataFrame(data_records)
    
    # Pre-calculate Tabular Features (LightGBM)
    lgb_feats = []
    for idx, row in df.iterrows():
        p1 = get_phys_features(row['Smiles#1'])
        p2 = get_phys_features(row['Smiles#2'])
        x1_val = float(row['X#1 (molar fraction)'])
        dtype_str = str(row.get('Type of DES', 'Unknown')).strip()
        des_int = DES_MAPPING.get(dtype_str, UNKNOWN_DES_IDX)
        lgb_feats.append(p1 + p2 + [x1_val, des_int])
    lgb_feats = np.array(lgb_feats)
    
    # Pre-calculate Graph Dataset
    dataset = GraphMixtureDataset(df, scale)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Iterate Through Folds
    all_fold_preds = [] # Shape: [n_folds, 11_points]
    
    for fold in range(N_FOLDS):
        # Load GNN
        gnn = ThermoGraphModel(config).to(device)
        gnn_path = f"ensemble_gnn_fold_{fold}.pth"
        try:
            gnn.load_state_dict(torch.load(gnn_path, map_location=device))
            gnn.eval()
        except FileNotFoundError:
            print(f"Skipping Fold {fold} (Model not found)")
            continue
            
        # Get GNN Preds
        gnn_preds = []
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                out = gnn(
                    inputs['x1'], inputs['adj1'], inputs['mask1'],
                    inputs['x2'], inputs['adj2'], inputs['mask2'],
                    inputs['frac1'], inputs['frac2'],
                    inputs['base_lin'], inputs['base_eut'], inputs['des_type']
                )
                out = torch.clamp(out, min=0.5, max=50.0)
                gnn_preds.extend((scale / out).cpu().numpy())
        
        # Load Booster
        lgb_path = f"ensemble_lgb_fold_{fold}.txt"
        try:
            gbm = lgb.Booster(model_file=lgb_path)
            resid_preds = gbm.predict(lgb_feats)
        except Exception:
            resid_preds = np.zeros_like(gnn_preds)
            
        # Stack Prediction
        total_pred = np.array(gnn_preds) + resid_preds
        all_fold_preds.append(total_pred)

    # 3. Aggregate Results
    if not all_fold_preds:
        print("Error: No models loaded.")
        return

    all_fold_preds = np.array(all_fold_preds) # [5, 11]
    mean_preds = np.mean(all_fold_preds, axis=0)
    std_preds = np.std(all_fold_preds, axis=0)
    
    # Calculate Linear Baseline for reference
    t_lin = x_range*t1 + (1-x_range)*t2

    # 4. Print Table
    print("-" * 105)
    print(f"{'x1':<6} | {'Pred Tm (K)':<12} | {'Uncertainty':<12} | {'Linear (K)':<10} | {'Is Deep?':<10}")
    print("-" * 105)
    
    for i, x1 in enumerate(x_range):
        tm = mean_preds[i]
        unc = std_preds[i]
        lin = t_lin[i]
        depression = lin - tm
        is_deep = "YES" if depression > 20 else "No"
        
        print(f"{x1:<6.2f} | {tm:<12.2f} | ± {unc:<10.2f} | {lin:<10.2f} | {is_deep:<10}")
        
    print("-" * 105)
    print(f"Interpretation: 'Uncertainty' is the disagreement between the {N_FOLDS} models.")
    print("If Uncertainty > 10 K, the prediction might be unreliable (out of domain).")

if __name__ == "__main__":
    # Test 1: Relin + Urea (Common DES)
    # Urea T=406K, Choline Chloride T=575K (approx)
    predict_ensemble(
        smiles1="NC(=O)N", t1=406.0,
        smiles2="C[N+](C)(C)CCO.[Cl-]", t2=575.0,
        des_type='3'
    )
    
    # Test 2: Menthol + Decanoic Acid (Hydrophobic DES)
    # Menthol=315K, Decanoic=304K
    predict_ensemble(
        smiles1="CC(C)C1CCC(C)CC1O", t1=315.0,
        smiles2="CCCCCCCCCC(=O)O", t2=304.0,
        des_type='5'
    )
