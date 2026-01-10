import os
import tensorflow as tf
import numpy as np
from src.data.ftir_dataset import FTIRDataset
from src.data.monomer_mapping import MonomerMapping
from src.data.smiles_tokenizer import SMILESTokenizer, RDKitSMILESTokenizer
from src.data.data_module import FTIRToSMILESDataModule
from src.models.encoder import FTIREncoder
from src.models.decoder import SMILESDecoder
from src.models.transformer import FTIRToSMILESTransformer
from src.models.predictor import FTIRMonomerPredictor
from sklearn.model_selection import train_test_split

# --------------------------
# CONFIGURATION
# --------------------------
FTIR_CSV = "data/merged_postprocessed_FTIR.csv"
PLASTIC_MONOMER_CSV = "data/monomers - plastics to monomers.csv"
MONOMERS_PUBCHEM_CSV = "data/monomers - Monomers PubChem.csv"

MAX_LEN = 64
BATCH_SIZE = 100
EPOCHS = 100

MODEL_SAVE_PATH = "checkpoints/ftir_transformer.weights.h5"
SCALER_PATH = "checkpoints/ftir_scaler.save"
PCA_PATH = "checkpoints/ftir_pca.save"

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
print("X shape before flattening:", X.shape)  # (samples, features, 1)

# Convert to NumPy if it's a TF tensor
if isinstance(X, tf.Tensor):
    X = X.numpy()
if isinstance(Y, tf.Tensor):
    Y = Y.numpy()

# TODO: Mitigate class imbalances?

# --------------------------
# 2. Train/test split
# --------------------------

# IMPORTANT:
# X and Y already correspond to filtered, valid samples
# We must build labels from the SAME order

plastic_names_used = data_module.plastic_names_used  # must exist
plastic_labels = np.array(
    [mapping.get_plastic_id(p) for p in plastic_names_used]
)

assert len(plastic_labels) == X.shape[0]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=plastic_labels
)

#from sklearn.model_selection import StratifiedKFold

#cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

#folds = []
#for fold_id, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
#    folds.append((train_idx, val_idx))
#    print(f"Fold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# --------------------------
# 3. Build model
# --------------------------
encoder = FTIREncoder()
decoder = SMILESDecoder(vocab_size=tokenizer.vocab_size)
model = FTIRToSMILESTransformer(encoder, decoder)

# --------------------------
# 4. Prepare dataset for training
# --------------------------
# Use train/test splits instead of full dataset
train_dataset = tf.data.Dataset.from_tensor_slices(((X_train, Y_train[:, :-1]), Y_train[:, 1:]))
test_dataset = tf.data.Dataset.from_tensor_slices(((X_test, Y_test[:, :-1]), Y_test[:, 1:]))

# Shuffle and batch training dataset; batch test dataset
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# --------------------------
# 5. Compile model
# --------------------------
# TODO: Fix padding - Needs to be the same everywhere. Cannot clash with other values.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
optimizer = tf.keras.optimizers.Adam(1e-3)
model.compile(optimizer=optimizer, loss=loss_fn)

# --------------------------
# 6. Train model
# --------------------------
print(f"Training on {X_train.shape[0]} samples, validating on {X_test.shape[0]} samples for {EPOCHS} epoch(s)...")
model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, verbose=2)

# --------------------------
# 7. Save model weights
# --------------------------
model.save_weights(MODEL_SAVE_PATH)
print(f"Model weights saved to {MODEL_SAVE_PATH}")

# --------------------------
# 8. Use predictor
# --------------------------
predictor = FTIRMonomerPredictor(
    model=model,
    tokenizer=tokenizer,
    scaler_path=SCALER_PATH,
    pca_path=PCA_PATH,
    max_len=MAX_LEN
)

# Example prediction
example_ftir = X[0]  # raw spectrum
predicted_smiles = predictor.predict(example_ftir, debug=True)
print(f"Predicted SMILES: {predicted_smiles}")

