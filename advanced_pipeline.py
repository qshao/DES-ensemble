import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import yaml
import os
import copy
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from gnn_model import ThermoGraphModel, DES_MAPPING, UNKNOWN_DES_IDX, R_GAS, WALDEN_DS
from graph_utils import smiles_to_graph

# ==========================================
# 1. Advanced Configuration
# ==========================================
CONFIG = {
    'batch_size': 32,
    'epochs': 80,
    'lr': 0.0005,           # Conservative LR
    'n_folds': 5,
    'target_scale': 1000.0,
    'aug_prob': 0.5,
    'clean_threshold_low': 150.0, 
    'clean_threshold_std': 10.0   
}

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# ==========================================
# 2. Augmented Dataset (Safe Version)
# ==========================================
class AugmentedGraphDataset(Dataset):
    def __init__(self, dataframe, scale, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.scale = scale
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
        
    def randomize_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol:
                return Chem.MolToSmiles(mol, doRandom=True)
        except:
            pass
        return str(smiles)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s1 = str(row['Smiles#1'])
        s2 = str(row['Smiles#2'])
        
        # Safe Augmentation
        if self.augment and np.random.rand() < CONFIG['aug_prob']:
            s1_aug = self.randomize_smiles(s1)
            # Verify validity
            if Chem.MolFromSmiles(s1_aug): s1 = s1_aug
            
        if self.augment and np.random.rand() < CONFIG['aug_prob']:
            s2_aug = self.randomize_smiles(s2)
            if Chem.MolFromSmiles(s2_aug): s2 = s2_aug
            
        x1, adj1, mask1 = smiles_to_graph(s1)
        x2, adj2, mask2 = smiles_to_graph(s2)
        
        # Physics
        t1, t2 = float(row['T#1']), float(row['T#2'])
        f1, f2 = float(row['X#1 (molar fraction)']), float(row['X#2 (molar fraction)'])
        
        target = 0.0
        if 'Tmelt, K' in row:
            target = (1.0 / float(row['Tmelt, K'])) * self.scale
            
        t_lin = f1*t1 + f2*t2
        base_lin = (1.0 / t_lin) * self.scale
        
        sx1, sx2 = max(f1, 1e-6), max(f2, 1e-6)
        t_id1 = t1 / (1 - (R_GAS/WALDEN_DS)*np.log(sx1))
        t_id2 = t2 / (1 - (R_GAS/WALDEN_DS)*np.log(sx2))
        base_eut = (1.0 / max(t_id1, t_id2)) * self.scale
        
        dtype = str(row.get('Type of DES', 'Unknown')).strip()
        des_idx = DES_MAPPING.get(dtype, UNKNOWN_DES_IDX)
        
        return {
            'x1': x1, 'adj1': adj1, 'mask1': mask1,
            'x2': x2, 'adj2': adj2, 'mask2': mask2,
            'frac1': torch.tensor(f1, dtype=torch.float32), 
            'frac2': torch.tensor(f2, dtype=torch.float32),
            'base_lin': torch.tensor(base_lin, dtype=torch.float32),
            'base_eut': torch.tensor(base_eut, dtype=torch.float32),
            'des_type': torch.tensor(des_idx, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.float32)
        }

# ==========================================
# 3. Ensemble Training Loop (Fault Tolerant)
# ==========================================
def train_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = load_config()
    
    # Load and Clean
    raw_df = pd.read_csv(base_config['data']['raw_csv'])
    # Safe copy
    clean_df = raw_df[raw_df['Tmelt, K'] > CONFIG['clean_threshold_low']].copy()
    
    clean_df['sys_id'] = clean_df.apply(lambda x: f"{sorted([str(x['Smiles#1']), str(x['Smiles#2'])])}_{round(x['X#1 (molar fraction)'], 2)}", axis=1)
    dup_stats = clean_df.groupby('sys_id')['Tmelt, K'].std()
    bad_ids = dup_stats[dup_stats > CONFIG['clean_threshold_std']].index
    clean_df = clean_df[~clean_df['sys_id'].isin(bad_ids)]
    
    kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
    
    print("\n" + "="*60)
    print(f"STARTING {CONFIG['n_folds']}-FOLD ROBUST ENSEMBLE")
    print("="*60)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(clean_df)):
        print(f"\n--- FOLD {fold+1}/{CONFIG['n_folds']} ---")
        
        train_sub = clean_df.iloc[train_idx]
        val_sub = clean_df.iloc[val_idx]
        
        train_ds = AugmentedGraphDataset(train_sub, CONFIG['target_scale'], augment=True)
        val_ds = AugmentedGraphDataset(val_sub, CONFIG['target_scale'], augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])
        
        model = ThermoGraphModel(base_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
        criterion = nn.HuberLoss(delta=0.5)
        
        best_mae = float('inf')
        model_saved = False
        
        # --- GNN Training Phase ---
        for epoch in range(CONFIG['epochs']):
            model.train()
            train_losses = []
            
            for batch in train_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                pred = model(
                    inputs['x1'], inputs['adj1'], inputs['mask1'],
                    inputs['x2'], inputs['adj2'], inputs['mask2'],
                    inputs['frac1'], inputs['frac2'],
                    inputs['base_lin'], inputs['base_eut'], inputs['des_type']
                )
                loss = criterion(pred, inputs['target'])
                
                if torch.isnan(loss): continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    out = model(
                        inputs['x1'], inputs['adj1'], inputs['mask1'],
                        inputs['x2'], inputs['adj2'], inputs['mask2'],
                        inputs['frac1'], inputs['frac2'],
                        inputs['base_lin'], inputs['base_eut'], inputs['des_type']
                    )
                    # Safe Clamp
                    out = torch.clamp(out, min=0.5, max=50.0) 
                    
                    y_pred.extend((CONFIG['target_scale']/out).cpu().numpy())
                    y_true.extend((CONFIG['target_scale']/inputs['target']).cpu().numpy())
            
            # Clean NaNs in predictions
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            
            # Replace NaNs/Infs with 0 or Mean (Robustness)
            y_pred = np.nan_to_num(y_pred, nan=300.0, posinf=300.0, neginf=300.0)
            
            mae = mean_absolute_error(y_true, y_pred)
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), f"ensemble_gnn_fold_{fold}.pth")
                model_saved = True
        
        if not model_saved or best_mae == float('inf'):
            print(f"!!! Fold {fold+1} FAILED to converge (Best MAE: {best_mae}). Skipping this fold.")
            continue
            
        print(f"Best GNN MAE (Fold {fold+1}): {best_mae:.2f} K")
        
        # --- LightGBM Boosting Phase ---
        try:
            model.load_state_dict(torch.load(f"ensemble_gnn_fold_{fold}.pth", map_location=device))
            model.eval()
            
            # Get GNN Train Preds
            gnn_preds_train = []
            train_targets = []
            train_eval_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
            
            with torch.no_grad():
                for batch in train_eval_loader:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    out = model(
                        inputs['x1'], inputs['adj1'], inputs['mask1'],
                        inputs['x2'], inputs['adj2'], inputs['mask2'],
                        inputs['frac1'], inputs['frac2'],
                        inputs['base_lin'], inputs['base_eut'], inputs['des_type']
                    )
                    out = torch.clamp(out, min=0.5, max=50.0)
                    gnn_preds_train.extend((CONFIG['target_scale']/out).cpu().numpy())
                    train_targets.extend((CONFIG['target_scale']/inputs['target']).cpu().numpy())
            
            gnn_preds_train = np.nan_to_num(np.array(gnn_preds_train), nan=300.0)
            train_targets = np.array(train_targets)

            # Extract Features
            def get_tab_feats(df_in):
                feats = []
                from rdkit.Chem import Descriptors
                for _, r in df_in.iterrows():
                    try:
                        m1 = Chem.MolFromSmiles(str(r['Smiles#1']))
                        m2 = Chem.MolFromSmiles(str(r['Smiles#2']))
                        if m1 and m2:
                            f = [
                                Descriptors.MolWt(m1), Descriptors.TPSA(m1), Descriptors.MolLogP(m1),
                                Descriptors.MolWt(m2), Descriptors.TPSA(m2), Descriptors.MolLogP(m2),
                                r['X#1 (molar fraction)'], 
                                DES_MAPPING.get(str(r.get('Type of DES', 'Unknown')).strip(), UNKNOWN_DES_IDX)
                            ]
                            feats.append(f)
                        else: feats.append([0]*8)
                    except: feats.append([0]*8)
                return np.array(feats)

            X_train_tab = get_tab_feats(train_sub)
            residuals = train_targets - gnn_preds_train
            
            # Train Booster
            lgb_train = lgb.Dataset(X_train_tab, residuals, categorical_feature=[7])
            # Reduce verbosity
            params = {'objective': 'regression', 'metric': 'l1', 'verbose': -1, 'learning_rate': 0.05}
            gbm = lgb.train(params, lgb_train, num_boost_round=500)
            gbm.save_model(f"ensemble_lgb_fold_{fold}.txt")
            
            # Eval
            X_val_tab = get_tab_feats(val_sub)
            
            # Re-predict Val with GNN
            gnn_preds_val = []
            val_targets = []
            val_eval_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
            with torch.no_grad():
                for batch in val_eval_loader:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    out = model(
                        inputs['x1'], inputs['adj1'], inputs['mask1'],
                        inputs['x2'], inputs['adj2'], inputs['mask2'],
                        inputs['frac1'], inputs['frac2'],
                        inputs['base_lin'], inputs['base_eut'], inputs['des_type']
                    )
                    out = torch.clamp(out, min=0.5, max=50.0)
                    gnn_preds_val.extend((CONFIG['target_scale']/out).cpu().numpy())
                    val_targets.extend((CONFIG['target_scale']/inputs['target']).cpu().numpy())

            gnn_preds_val = np.nan_to_num(np.array(gnn_preds_val), nan=300.0)
            
            resid_preds = gbm.predict(X_val_tab)
            final_preds = gnn_preds_val + resid_preds
            
            fold_mae = mean_absolute_error(val_targets, final_preds)
            print(f"Stack MAE (Fold {fold+1}): {fold_mae:.2f} K")
            fold_results.append(fold_mae)
            
        except Exception as e:
            print(f"Error in Boosting stage for Fold {fold+1}: {e}")
            continue

    if fold_results:
        print("\n" + "="*60)
        print(f"ENSEMBLE COMPLETE")
        print(f"Average MAE across successful folds: {np.mean(fold_results):.2f} K")
        print("="*60)
    else:
        print("All folds failed.")

if __name__ == "__main__":
    train_ensemble()
