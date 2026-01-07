from src.data.ftir_dataset import FTIRDataset
from src.data.monomer_mapping import MonomerMapping
from src.data.smiles_tokenizer import SMILESTokenizer
from src.data.data_module import FTIRToSMILESDataModule

ftir = FTIRDataset("data/merged_postprocessed_FTIR.csv")
ftir.load()

mapping = MonomerMapping(
    "data/monomers - plastics to monomers.csv",
    "data/monomers - Monomers PubChem.csv"
)

tokenizer = SMILESTokenizer()

data_module = FTIRToSMILESDataModule(
    ftir_ds=ftir,
    monomer_map=mapping,
    tokenizer=tokenizer,
    max_len=128
)

X, Y = data_module.build()

print("FTIR tensor shape:", X.shape)
print("Target tensor shape:", Y.shape)

# Inspect one example
idx = 0
print("\nExample plastic:", ftir.get_plastics()[idx])
print("First 10 FTIR values:", X[idx, :10, 0].numpy())

decoded = tokenizer.decode(Y[idx].numpy())
print("Decoded target SMILES:", decoded)
