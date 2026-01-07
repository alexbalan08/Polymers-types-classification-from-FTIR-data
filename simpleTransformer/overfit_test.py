import tensorflow as tf

from src.data.ftir_dataset import FTIRDataset
from src.data.monomer_mapping import MonomerMapping
from src.data.smiles_tokenizer import SMILESTokenizer
from src.data.data_module import FTIRToSMILESDataModule

from src.models.encoder import FTIREncoder
from src.models.decoder import SMILESDecoder
from src.models.transformer import FTIRToSMILESTransformer

# Load data
ftir = FTIRDataset("data/merged_postprocessed_FTIR.csv")
ftir.load()

mapping = MonomerMapping(
    "data/monomers - plastics to monomers.csv",
    "data/monomers - Monomers PubChem.csv"
)

tokenizer = SMILESTokenizer()

dm = FTIRToSMILESDataModule(
    ftir_ds=ftir,
    monomer_map=mapping,
    tokenizer=tokenizer,
    max_len=64
)

X, Y = dm.build()

# Tiny subset
X = X[:5]
Y = Y[:5]

encoder = FTIREncoder()
decoder = SMILESDecoder(vocab_size=tokenizer.vocab_size)

model = FTIRToSMILESTransformer(encoder, decoder)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, ignore_class=0
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=loss_fn
)

model.fit(
    x=(X, Y[:, :-1]),
    y=Y[:, 1:],
    epochs=200,
    verbose=2
)
