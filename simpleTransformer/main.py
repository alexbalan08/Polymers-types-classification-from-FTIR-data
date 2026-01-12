import os
import tensorflow as tf
import numpy as np
from src.data.ftir_dataset import FTIRDataset
from src.data.monomer_mapping import MonomerMapping
from src.data.smiles_tokenizer import RDKitSMILESTokenizer
from src.data.data_module import FTIRToSMILESDataModule
from src.models.predictor import FTIRMonomerPredictor
from src.training.train_helper import train_cross_validation
from datetime import datetime
import csv
import pickle

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

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = os.path.join("checkpoints", timestamp)
os.makedirs(checkpoint_dir, exist_ok=True)

# Update paths to point inside this folder
MODEL_SAVE_PATH = os.path.join(checkpoint_dir, "ftir_transformer.weights.h5")
SCALER_PATH = os.path.join(checkpoint_dir, "ftir_scaler.save")
PCA_PATH = os.path.join(checkpoint_dir, "ftir_pca.save")
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
    "MODEL_SAVE_PATH": MODEL_SAVE_PATH,
    "SCALER_PATH": SCALER_PATH,
    "PCA_PATH": PCA_PATH,
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

# Save scaler
with open(SCALER_PATH, "wb") as f:
    pickle.dump(data_module.scaler, f)

# Save PCA
with open(PCA_PATH, "wb") as f:
    pickle.dump(data_module.pca, f)

print(f"Scaler saved to {SCALER_PATH}")
print(f"PCA saved to {PCA_PATH}")


print("X shape before flattening:", X.shape)
print("Max number of tokens:     ", (Y == 0).argmax(axis=1).max())

# Stratification labels
plastic_labels = np.array([mapping.get_plastic_id(p) for p in data_module.plastic_names_used])
assert len(plastic_labels) == X.shape[0]

# --------------------------
# 2. Train model using helper
# --------------------------
model, training_history = train_cross_validation(
    X, Y, plastic_labels,
    tokenizer, mapping,
    n_splits=3,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    model_save_path=MODEL_SAVE_PATH
)

# --------------------------
# 3. Use predictor
# --------------------------
predictor = FTIRMonomerPredictor(
    model=model,
    tokenizer=tokenizer,
    scaler_path=SCALER_PATH,
    pca_path=PCA_PATH,
    max_len=MAX_LEN
)

example_ftir = X[0]
predicted_smiles = predictor.predict(example_ftir, debug=True)
print(f"Predicted SMILES: {predicted_smiles}")
