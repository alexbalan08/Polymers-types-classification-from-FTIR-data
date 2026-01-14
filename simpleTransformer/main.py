import os
import tensorflow as tf
import numpy as np
import pandas as pd
from src.data.ftir_dataset import FTIRDataset
from src.data.monomer_mapping import MonomerMapping
from src.data.smiles_tokenizer import RDKitSMILESTokenizer
from src.data.selfies_tokenizer import SELFIESTokenizer
from src.data.data_module import FTIRToSequenceDataModule
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
FINGERPRINT_CSV = "data/monomers - Monomers PubChem.csv"

D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
DROP_RATE = 0.1
MAX_LEN = 48
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
PRETRAIN_EPOCHS = 5
PRED_THRESHOLD = 0.10
POST_PRETRAIN_FREEZE_DECODER = True
USE_SELFIES = False
TARGET_COL = "selfies" if USE_SELFIES else "canonical_smiles"

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
    "FINGERPRINT_CSV": FINGERPRINT_CSV,
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
# Depending on datatype used
tokenizer = SELFIESTokenizer() if USE_SELFIES else RDKitSMILESTokenizer()

data_module = FTIRToSequenceDataModule(
    ftir_ds=ftir_ds,
    monomer_map=mapping,
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    use_selfies=USE_SELFIES
)

X, Y = data_module.build()

# --------------------------
# 1a. Build fingerprint -> SMILES dataset
# --------------------------
print("Building fingerprint dataset for pretraining...")

fingerprint_df = pd.read_csv(FINGERPRINT_CSV)
#fingerprint_df = fingerprint_df[fingerprint_df["max_tanimoto"] >= 0.60]

fingerprint_df["cactvs_fingerprint"] = fingerprint_df["cactvs_fingerprint"].apply(lambda s: np.array(list(s), dtype=int))
X_fp = np.stack(fingerprint_df["cactvs_fingerprint"].to_numpy())  # np.array(fingerprint_df['cactvs_fingerprint'], dtype=float)
Y_fp = np.array(fingerprint_df[TARGET_COL])

print(f"Fingerprint dataset shape: X={X_fp.shape}, Y={Y_fp.shape}")

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
    X_fp=X_fp,
    Y_fp=Y_fp,
    pretrain_epochs=PRETRAIN_EPOCHS,
    freeze_decoder_after_pretrain=POST_PRETRAIN_FREEZE_DECODER,
    use_selfies=USE_SELFIES,
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
predicted_instances, probs = predictor.predict(example_ftir, PRED_THRESHOLD, debug=True)
for target_instance, p in zip(predicted_instances, probs):
    print(f"Predicted Instances (conf={p:5.3f}): {target_instance}")
print(f"True Instance/-s was/were:  {Y_val[0]}")
