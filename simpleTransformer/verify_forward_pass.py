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

# Take a small batch
Xb = X[:4]
Yb = Y[:4, :-1]  # teacher forcing input

encoder = FTIREncoder()
decoder = SMILESDecoder(vocab_size=tokenizer.vocab_size)

model = FTIRToSMILESTransformer(encoder, decoder)

logits = model(Xb, Yb)

print("Input FTIR batch:", Xb.shape)
print("Input target batch:", Yb.shape)
print("Output logits:", logits.shape)
