import pandas as pd
from rdkit import Chem
import selfies as sf

# ---- configuration ----
input_csv = "data/monomers - Monomers PubChem.csv"
output_csv = "data/monomers - Monomers PubChem_incl.csv"
smiles_column = "smiles"
# -----------------------

# Load CSV
df = pd.read_csv(input_csv)

def to_canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except Exception:
        return None

# Generate new columns
canonical_smiles = df[smiles_column].apply(to_canonical_smiles)
selfies = df[smiles_column].apply(to_selfies)

# Find index of the smiles column
smiles_idx = df.columns.get_loc(smiles_column)

# Insert columns right after "smiles"
df.insert(smiles_idx + 1, "canonical_smiles", canonical_smiles)
df.insert(smiles_idx + 2, "selfies", selfies)

# Save output
df.to_csv(output_csv, index=False)

print(f"Done. Output written to: {output_csv}")
