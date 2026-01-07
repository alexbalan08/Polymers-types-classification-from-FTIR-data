# main.py
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

MAX_LEN = 64           # max SMILES length
BATCH_SIZE = 8         # small batch for demo
EPOCHS = 20            # small number of epochs for demo
MODEL_SAVE_PATH = "checkpoints/ftir_transformer_weights.weights.h5"

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

# --------------------------
# 2. Build model
# --------------------------
encoder = FTIREncoder()
decoder = SMILESDecoder(vocab_size=tokenizer.vocab_size)
model = FTIRToSMILESTransformer(encoder, decoder)

# --------------------------
# 3. Prepare dataset for training
# --------------------------
# Teacher forcing: decoder input is Y[:, :-1], target is Y[:, 1:]
dataset = tf.data.Dataset.from_tensor_slices(((X, Y[:, :-1]), Y[:, 1:]))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE)

# --------------------------
# 4. Compile model
# --------------------------
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
optimizer = tf.keras.optimizers.Adam(1e-3)
model.compile(optimizer=optimizer, loss=loss_fn)

# --------------------------
# 5. Train model (configurable)
# --------------------------
print(f"Training on {BATCH_SIZE} batch(es) for {EPOCHS} epoch(s)...")
model.fit(dataset.take(5), epochs=EPOCHS, verbose=2)  # .take(5) limits training to 5 batches

# --------------------------
# 6. Save model weights
# --------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save_weights(MODEL_SAVE_PATH)
print(f"Model weights saved to {MODEL_SAVE_PATH}")

# --------------------------
# 7. Load model weights (demonstrate reload)
# --------------------------
encoder2 = FTIREncoder()
decoder2 = SMILESDecoder(vocab_size=tokenizer.vocab_size)
model2 = FTIRToSMILESTransformer(encoder2, decoder2)
model2.load_weights(MODEL_SAVE_PATH)
print("Model weights loaded successfully")

# --------------------------
# 8. Predict SMILES
# --------------------------
predictor = FTIRMonomerPredictor(model2, tokenizer)

# Single prediction
single_smiles = predictor.predict(X[0])
print("\nSingle spectrum prediction:", single_smiles)

# Batch prediction
batch_smiles = predictor.predict_multiple(X[:5])
print("\nBatch predictions:", batch_smiles)
