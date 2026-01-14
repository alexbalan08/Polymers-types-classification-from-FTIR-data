import os
import pandas as pd
import tensorflow as tf
import numpy as np
from src.data.ftir_dataset import FTIRDataset
from src.data.monomer_mapping import MonomerMapping
from src.data.smiles_tokenizer import RDKitSMILESTokenizer
from src.data.data_module import FTIRToSequenceDataModule
from src.training.train_helper import train_cross_validation
from datetime import datetime
import csv

# --------------------------
# CONFIG
# --------------------------
FTIR_CSV = "data/merged_postprocessed_FTIR.csv"
PLASTIC_MONOMER_CSV = "data/monomers - plastics to monomers.csv"
MONOMERS_PUBCHEM_CSV = "data/monomers - Monomers PubChem.csv"
FINGERPRINT_CSV = "data/filtered_data_40_50_1272421.csv"

# d_model, num_heads, num_layers
CONFIGS = [
    (8,  2, 1),
    (16, 2, 1),
    (32, 2, 1),
    (8,  4, 1),
    (16, 4, 1),
    (32, 4, 1),
    (8,  8, 1),
    (16, 8, 1),
    (32, 8, 1),

    (8,  4, 2),
    (16, 4, 2),
    (32, 4, 2),
    (8,  8, 2),
    (16, 8, 2),
    (32, 8, 2),
    (16, 16, 2),
    (32, 16, 2),

    (16, 8, 4),
    (32, 8, 4),
    (48, 8, 4),
]
EXP_NAME = "Basic_model_with_clust"
PRE_TRAIN = False

DROP_RATE = 0.1
MAX_LEN = 48
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PRETRAIN_EPOCHS = 25

def run():
    for d_model, num_heads, num_layers in CONFIGS:
        print("START with:", d_model, num_heads, num_layers)
        function_train_parameters(d_model, num_heads, num_layers, PRE_TRAIN)


def function_train_parameters(d_model, num_heads, num_layers, do_pretraining):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------------------------
    # 1. Load dataset
    # --------------------------
    ftir_ds = FTIRDataset(FTIR_CSV)
    ftir_ds.load()

    mapping = MonomerMapping(PLASTIC_MONOMER_CSV, MONOMERS_PUBCHEM_CSV)
    tokenizer = RDKitSMILESTokenizer()

    data_module = FTIRToSequenceDataModule(
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

    fingerprint_df = pd.read_csv(FINGERPRINT_CSV)
    fingerprint_df = fingerprint_df[fingerprint_df["max_tanimoto"] >= 0.60]

    fingerprint_df["cactvs_fingerprint"] = fingerprint_df["cactvs_fingerprint"].apply(
        lambda s: np.array(list(s), dtype=int))
    X_fp = np.stack(
        fingerprint_df["cactvs_fingerprint"].to_numpy())  # np.array(fingerprint_df['cactvs_fingerprint'], dtype=float)
    Y_fp = np.array(fingerprint_df['canonical_smiles'])

    print(f"Fingerprint dataset shape: X={X_fp.shape}, Y={Y_fp.shape}")

    # --------------------------
    # 2. Train model using helper
    # --------------------------
    _, (_, _), _, (_, _) = train_cross_validation(
        X, Y, tokenizer, data_module, checkpoint_dir,
        n_splits=3,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        drop_rate=DROP_RATE,
        max_len=MAX_LEN,
        learning_rate=LEARNING_RATE,
        do_pretraining=do_pretraining,
        X_fp=X_fp,
        Y_fp=Y_fp,
        pretrain_epochs=PRETRAIN_EPOCHS
    )
    end_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --------------------------
    # Log configuration
    # --------------------------
    config = {
        "EXPERIMENT_NAME": "First round basic model",
        "FTIR_CSV": FTIR_CSV,
        "PLASTIC_MONOMER_CSV": PLASTIC_MONOMER_CSV,
        "MONOMERS_PUBCHEM_CSV": MONOMERS_PUBCHEM_CSV,
        "FINGERPRINT_CSV": FINGERPRINT_CSV,
        "D_MODEL": d_model,
        "NUM_HEADS": num_heads,
        "NUM_LAYERS": num_layers,
        "DROP_RATE": DROP_RATE,
        "MAX_LEN": MAX_LEN,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
        "TIMESTAMP": timestamp,
        "START_TIME": timestamp,
        "END_TIME": end_timestamp
    }

    config_csv_path = os.path.join(checkpoint_dir, "config_log.csv")
    file_exists = os.path.isfile(config_csv_path)
    with open(config_csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=config.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(config)
    print(f"Configuration logged to {config_csv_path}")


if __name__ == "__main__":
    run()
