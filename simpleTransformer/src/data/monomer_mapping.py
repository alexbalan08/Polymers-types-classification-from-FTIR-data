# Maps plastics → monomers
import pandas as pd

class MonomerMapping:
    def __init__(self, plastic_to_monomers_csv, monomers_pubchem_csv):
        self.plastic_df = pd.read_csv(plastic_to_monomers_csv)
        self.monomer_df = pd.read_csv(monomers_pubchem_csv)

        # Normalize mapping table
        self.plastic_df["Plastic"] = (
            self.plastic_df["Plastic"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        self.monomer_df["monomer"] = (
            self.monomer_df["monomer"]
            .astype(str)
            .str.strip()
        )

        self.smiles_lookup = dict(
            zip(self.monomer_df["monomer"], self.monomer_df["smiles"])
        )

    def _normalize_plastic_name(self, plastic: str) -> str:
        """
        Convert FTIR plastic name to mapping-compatible format
        Example: 1_2_polybutadiene -> 1 2 polybutadiene
        """
        return plastic.replace("_", " ").strip().lower()

    def get_monomers_for_plastic(self, plastic):
        plastic_norm = self._normalize_plastic_name(plastic)

        row = self.plastic_df[self.plastic_df["Plastic"] == plastic_norm]
        if row.empty:
            return []

        monomers = row.iloc[0]["Monomers"]
        return [m.strip() for m in monomers.split(";") if m.strip()]

    def get_smiles_for_plastic(self, plastic):
        monomers = self.get_monomers_for_plastic(plastic)
        smiles = []

        # TODO: Add conversion to canonical smiles


        for m in monomers:
            if m in self.smiles_lookup:
                smiles.append(self.smiles_lookup[m])

        return smiles

    def get_selfies_for_plastic(self, plastic):
        monomers = self.get_monomers_for_plastic(plastic)
        smiles = []

        # TODO: Add conversion to selfies
        for m in monomers:
            if m in self.smiles_lookup:
                smiles.append(self.smiles_lookup[m])

        return smiles
