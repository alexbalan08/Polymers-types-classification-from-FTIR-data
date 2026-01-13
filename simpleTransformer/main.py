import os
import tensorflow as tf
import numpy as np
import pandas as pd
from src.data.ftir_dataset import FTIRDataset
from src.data.monomer_mapping import MonomerMapping
from src.data.smiles_tokenizer import RDKitSMILESTokenizer
from src.data.data_module import FTIRToSMILESDataModule
from src.models.predictor import FTIRMonomerPredictor
from src.training.train_helper import train_cross_validation
from datetime import datetime
import csv

# --------------------------
# CONFIG
# --------------------------
FTIR_CSV = "data/merged_postprocessed_FTIR.csv"
PLASTIC_MONOMER_CSV = "data/monomers - plastics to monomers.csv"
MONOMERS_PUBCHEM_CSV = "data/monomers - Monomers PubChem.csv"

D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
DROP_RATE = 0.1
MAX_LEN = 48
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
PRETRAIN_EPOCHS = 5

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = os.path.join("checkpoints", timestamp)
os.makedirs(checkpoint_dir, exist_ok=True)

# Update paths to point inside this folder
config_csv_path = os.path.join(checkpoint_dir, "config_log.csv")
history_csv_path = os.path.join(checkpoint_dir, "training_history.csv")

# --------------------------
# Log configuration
# --------------------------
config = {
    "FTIR_CSV": FTIR_CSV,
    "PLASTIC_MONOMER_CSV": PLASTIC_MONOMER_CSV,
    "MONOMERS_PUBCHEM_CSV": MONOMERS_PUBCHEM_CSV,
    "D_MODEL": D_MODEL,
    "NUM_HEADS": NUM_HEADS,
    "NUM_LAYERS": NUM_LAYERS,
    "DROP_RATE": DROP_RATE,
    "MAX_LEN": MAX_LEN,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "TIMESTAMP": timestamp
}

file_exists = os.path.isfile(config_csv_path)

with open(config_csv_path, mode='a', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=config.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(config)

print(f"Configuration logged to {config_csv_path}")
# --------------------------
# 1. Load dataset
# --------------------------
ftir_ds = FTIRDataset(FTIR_CSV)
ftir_ds.load()

mapping = MonomerMapping(PLASTIC_MONOMER_CSV, MONOMERS_PUBCHEM_CSV)
tokenizer = RDKitSMILESTokenizer()

data_module = FTIRToSMILESDataModule(
    ftir_ds=ftir_ds,
    monomer_map=mapping,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

X, Y = data_module.build()

# --------------------------
# 1a. Build fingerprint -> SMILES dataset
# --------------------------
print("Building fingerprint dataset for pretraining...")

monomer_df = pd.read_csv(MONOMERS_PUBCHEM_CSV)

fingerprint_column = 'cactvs_fingerprint'
smiles_column = 'canonical_smiles'

X_fingerprint = []
Y_fingerprint = []

# Loop over each row in the FTIR CSV dataframe
for idx, row in monomer_df.iterrows():
    fp_str = row[fingerprint_column]
    smiles_list = row[smiles_column]

    # Some entries might have multiple SMILES separated by a delimiter (e.g., ';' or ',')
    if pd.isna(fp_str) or pd.isna(smiles_list):
        continue

    # Make sure smiles_list is iterable
    if isinstance(smiles_list, str):
        # Split by comma or semicolon if multiple monomers in the row
        smiles_items = [s.strip() for s in smiles_list.replace(';', ',').split(',')]
    else:
        smiles_items = list(smiles_list)

    # Convert fingerprint string to numeric array (0/1)
    fp_vec = np.array([int(c) for c in fp_str], dtype=float)

    # One entry per canonical SMILES
    for smiles in smiles_items:
        if smiles == '':
            continue
        X_fingerprint.append(fp_vec)
        Y_fingerprint.append(smiles)

# Convert to NumPy arrays
X_fingerprint = np.array(X_fingerprint, dtype=float)
Y_fingerprint = np.array(Y_fingerprint)

print(f"Fingerprint dataset shape: X={X_fingerprint.shape}, Y={Y_fingerprint.shape}")

# --------------------------
# 2. Train model using helper
# --------------------------
model, (scaler_path, pca_path), training_history, (X_val, Y_val) = train_cross_validation(
    X, Y, tokenizer, data_module, checkpoint_dir,
    n_splits=3,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    drop_rate=DROP_RATE,
    max_len=MAX_LEN,
    learning_rate=LEARNING_RATE,
    do_pretraining=True,
    X_fp=X_fingerprint,
    Y_fp=Y_fingerprint,
    pretrain_epochs=PRETRAIN_EPOCHS
)

# --------------------------
# 3. Use predictor
# --------------------------
predictor = FTIRMonomerPredictor(
    model=model,
    tokenizer=tokenizer,
    scaler_path=scaler_path,
    pca_path=pca_path,
    max_len=MAX_LEN
)

example_ftir = X_val[0]
print("Reduced form spectra:", X_val[0])
predicted_smiles = predictor.predict(example_ftir, debug=True)
print(f"Predicted SMILES: {predicted_smiles}")
print(f"True SMILES was:  {Y_val[0]}")
