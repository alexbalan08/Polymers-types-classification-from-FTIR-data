from src.data.monomer_mapping import MonomerMapping

mapping = MonomerMapping(
    "data/monomers - plastics to monomers.csv",
    "data/monomers - Monomers PubChem.csv"
)

# Test a few plastics manually
test_plastics = [
    "1_2_polybutadiene",
    "polyethylene",
    "polystyrene"
]

for plastic in test_plastics:
    monomers = mapping.get_monomers_for_plastic(plastic)
    smiles = mapping.get_smiles_for_plastic(plastic)

    print("\nPlastic:", plastic)
    print("  Monomers:", monomers)
    print("  SMILES:", smiles)
