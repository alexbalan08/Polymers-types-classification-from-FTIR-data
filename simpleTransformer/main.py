import os
import tensorflow as tf
from src.data.ftir_dataset import FTIRDataset
from src.data.monomer_mapping import MonomerMapping
from src.data.smiles_tokenizer import SMILESTokenizer
from src.data.data_module import FTIRToSMILESDataModule
from src.models.encoder import FTIREncoder
from src.models.decoder import SMILESDecoder
from src.models.transformer import FTIRToSMILESTransformer
from src.models.predictor import FTIRMonomerPredictor

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
tokenizer = SMILESTokenizer()

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

# TODO: Add train/test splitting

# --------------------------
# 3. Build model
# --------------------------
encoder = FTIREncoder()
decoder = SMILESDecoder(vocab_size=tokenizer.vocab_size)
model = FTIRToSMILESTransformer(encoder, decoder)

# --------------------------
# 4. Prepare dataset for training
# --------------------------
dataset = tf.data.Dataset.from_tensor_slices(((X, Y[:, :-1]), Y[:, 1:]))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE)

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
print(f"Training on {BATCH_SIZE} batch(es) for {EPOCHS} epoch(s)...")
model.fit(dataset, epochs=EPOCHS, verbose=2)

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

