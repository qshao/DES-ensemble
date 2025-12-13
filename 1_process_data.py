import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def process_data():
    config = load_config()
    input_file = config['data']['raw_csv']
    
    print(f"Processing {input_file}...")
    df = pd.read_csv(input_file)
    
    # 1. Clean Missing Values
    # Drop rows where critical info is missing
    subset = ['T#1', 'T#2', 'Smiles#1', 'Smiles#2', 'Tmelt, K']
    initial_len = len(df)
    df = df.dropna(subset=subset)
    print(f"Dropped {initial_len - len(df)} rows with missing values.")
    
    # 2. Standardize DES Type
    if 'Type of DES' in df.columns:
        df['Type of DES'] = df['Type of DES'].astype(str).str.strip()
        df['Type of DES'] = df['Type of DES'].replace(['nan', 'NaN', ''], 'Unknown')
    else:
        df['Type of DES'] = 'Unknown'
        
    print(f"Found DES Types: {df['Type of DES'].unique()}")

    # 3. Create System ID for splitting
    # Ensures we don't leak chemical pairs from train to val
    df['system_id'] = df.apply(lambda x: str(sorted([x['Smiles#1'], x['Smiles#2']])), axis=1)
    
    # 4. Split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df['system_id']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_df.to_csv(config['data']['train_file'], index=False)
    val_df.to_csv(config['data']['val_file'], index=False)
    
    print(f"Saved {len(train_df)} training samples and {len(val_df)} validation samples.")

if __name__ == "__main__":
    process_data()
