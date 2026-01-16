import os

import pandas as pd

from simpleTransformer.src.data.data_module import FTIRToSequenceDataModule
from simpleTransformer.src.data.ftir_dataset import FTIRDataset
from simpleTransformer.src.data.monomer_mapping import MonomerMapping
from simpleTransformer.src.data.selfies_tokenizer import SELFIESTokenizer
from simpleTransformer.src.data.smiles_tokenizer import SMILESTokenizer
from simpleTransformer.src.training.val_helper import test_cross_validation

# MODEL_PATH = r"basic_2026-01-15_02-38-40"
# MODEL_PATH = r"frozen_2026-01-15_23-21-11"
MODEL_PATH = r"selfies_2026-01-15_20-28-13"
START_FOLD = 1

# --------------------------
# CONFIG
# --------------------------
FTIR_CSV = "data/merged_postprocessed_FTIR.csv"
PLASTIC_MONOMER_CSV = "data/monomers - plastics to monomers.csv"
MONOMERS_PUBCHEM_CSV = "data/monomers - Monomers PubChem.csv"
FINGERPRINT_CSV = "data/filtered_data_40_50_1272421.csv"

MAX_LEN = 48
PRED_THRESHOLD = 0.10

checkpoint_dir = os.path.join("checkpoints", MODEL_PATH)

# Update paths to point inside this folder
config_csv_path = os.path.join(checkpoint_dir, "config_log.csv")
history_csv_path = os.path.join(checkpoint_dir, "training_history.csv")

df = pd.read_csv(config_csv_path)
config = df.iloc[0].to_dict()
USE_SELFIES = config["USE_SELFIES"]
TARGET_COL = "selfies" if USE_SELFIES else "canonical_smiles"

# --------------------------
# 1. Load dataset
# --------------------------
ftir_ds = FTIRDataset(FTIR_CSV)
ftir_ds.load()

mapping = MonomerMapping(PLASTIC_MONOMER_CSV, MONOMERS_PUBCHEM_CSV)
# Depending on datatype used
tokenizer = SELFIESTokenizer() if USE_SELFIES else SMILESTokenizer()

data_module = FTIRToSequenceDataModule(
    ftir_ds=ftir_ds,
    monomer_map=mapping,
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    use_selfies=USE_SELFIES
)

X, Y = data_module.build()

# --------------------------
# 2. Train model using helper
# --------------------------
test_cross_validation(
    X, Y, tokenizer, data_module, checkpoint_dir,
    n_splits=3,
    batch_size=config["BATCH_SIZE"],
    d_model=config["D_MODEL"],
    num_heads=config["NUM_HEADS"],
    num_layers=config["NUM_LAYERS"],
    drop_rate=config["DROP_RATE"],
    pred_threshold=PRED_THRESHOLD,
    max_len=MAX_LEN,
    start_fold=START_FOLD
)
