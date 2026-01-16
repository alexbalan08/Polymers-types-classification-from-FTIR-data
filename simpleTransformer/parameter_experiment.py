import gc
import os
import time

import pandas as pd
import numpy as np
from src.data.ftir_dataset import FTIRDataset
from src.data.monomer_mapping import MonomerMapping
from src.data.smiles_tokenizer import SMILESTokenizer
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
FINGERPRINT_CSV = "data/filtered_data_40_50_1272421.csv"

# d_model, num_heads, num_layers
CONFIGS = [
    # (8,  2, 1),
    # (16, 2, 1),
    # (32, 2, 1),
    # (48, 2, 1),  # NEW
    # (8,  4, 1),
    # (16, 4, 1),
    # (32, 4, 1),
    # (48, 4, 1),  # NEW
    # (8,  8, 1),
    # (16, 8, 1),
    # (32, 8, 1),
    # (48, 8, 1),  # NEW
    # (16, 16, 1),  # NEW
    # (32, 16, 1),  # NEW
    # (48, 16, 1),  # NEW

    # (8,  4, 2),
    # (16, 4, 2),
    # (32, 4, 2),
    # (48, 4, 2),  # NEW
    # (8,  8, 2),
    # (16, 8, 2),
    # (32, 8, 2),
    # (48, 8, 2),  # NEW
    # (16, 16, 2),
    # (32, 16, 2),
    # (48, 16, 2),  # NEW
    # (48, 16, 2, True),  # NEW
    (48, 16, 2, False, True, True),  # NEW
    (48, 16, 2, False, True, False),  # NEW

    # (8, 8, 4),  # NEW
    # (16, 8, 4),
    # (32, 8, 4),
    # (48, 8, 4),
    # (16, 16, 4),  # New
    # (32, 16, 4),
    # (48, 16, 4),  # New
]
EXP_NAME = "Basic_model_with_clust"
USE_SELFIES = False
PRE_TRAIN = False
POST_PRETRAIN_FREEZE_DECODER = False
DO_PREDICTION = True

DROP_RATE = 0.1
MAX_LEN = 48
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PRETRAIN_EPOCHS = 25
PRED_THRESHOLD = 0.10
TARGET_COL = "selfies" if USE_SELFIES else "canonical_smiles"

def run():
    for config in CONFIGS:
        if len(config) == 3:
            (d_model, num_heads, num_layers), use_selfies, pre_train, do_freeze = config, USE_SELFIES, PRE_TRAIN, POST_PRETRAIN_FREEZE_DECODER
        elif len(config) == 4:
            (d_model, num_heads, num_layers, use_selfies), pre_train, do_freeze = config, PRE_TRAIN, POST_PRETRAIN_FREEZE_DECODER
        elif len(config) == 6:
            (d_model, num_heads, num_layers, use_selfies, pre_train, do_freeze) = config

        print("START with:", d_model, num_heads, num_layers)
        function_train_parameters(d_model, num_heads, num_layers, pre_train, use_selfies, do_freeze)


def function_train_parameters(d_model, num_heads, num_layers, do_pretraining, use_selfies, do_freeze):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------------------------
    # 1. Load dataset
    # --------------------------
    ftir_ds = FTIRDataset(FTIR_CSV)
    ftir_ds.load()

    mapping = MonomerMapping(PLASTIC_MONOMER_CSV, MONOMERS_PUBCHEM_CSV)
    tokenizer = SELFIESTokenizer() if use_selfies else SMILESTokenizer()

    data_module = FTIRToSequenceDataModule(
        ftir_ds=ftir_ds,
        monomer_map=mapping,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        use_selfies=use_selfies
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
    Y_fp = np.array(fingerprint_df[TARGET_COL])

    print(f"Fingerprint dataset shape: X={X_fp.shape}, Y={Y_fp.shape}")

    # --------------------------
    # 2. Train model using helper
    # --------------------------
    model, (scaler_path, pca_path), training_history, val_data = train_cross_validation(
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
        pretrain_epochs=PRETRAIN_EPOCHS,
        freeze_decoder_after_pretrain = do_freeze,
        use_selfies=use_selfies,
    )
    end_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    del X_fp, Y_fp

    # --------------------------
    # Log configuration
    # --------------------------
    config = {
        "EXPERIMENT_NAME": EXP_NAME,
        "FTIR_CSV": FTIR_CSV,
        "PLASTIC_MONOMER_CSV": PLASTIC_MONOMER_CSV,
        "MONOMERS_PUBCHEM_CSV": MONOMERS_PUBCHEM_CSV,
        "FINGERPRINT_CSV": FINGERPRINT_CSV,
        "D_MODEL": d_model,
        "NUM_HEADS": num_heads,
        "NUM_LAYERS": num_layers,
        "USE_SELFIES": use_selfies,
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

    if DO_PREDICTION:
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

        store_x = []
        store_y = []
        store_pred = []
        store_probs = []

        start_time = time.time()
        for X_val, Y_val in val_data:
            for i, (xv, yv) in enumerate(zip(X_val, Y_val)):
                if (i % 100) == 0:
                    print(i)
                gc.collect()

                predicted_molecules, probs = predictor.predict(xv, PRED_THRESHOLD, debug=False)

                store_x.append(xv)
                store_y.append(yv)
                store_pred.append(predicted_molecules)
                store_probs.append(probs)

        print(f"PREDICTION took {time.time() - start_time} seconds")

        prediction_df = pd.DataFrame({
            "X_val": store_x,
            "Y_val": store_y,
            "prediction": store_pred,
            "probabilities": store_probs,
        })
        prediction_df.to_csv(os.path.join(checkpoint_dir, "prediction.csv"), index=False)


if __name__ == "__main__":
    run()
