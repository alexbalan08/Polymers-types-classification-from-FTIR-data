# Maps plastics → monomers
import pandas as pd

class MonomerMapping:
    def __init__(self, plastic_to_monomers_csv, monomers_pubchem_csv):
        self.plastic_df = pd.read_csv(plastic_to_monomers_csv)
        self.monomer_df = pd.read_csv(monomers_pubchem_csv)
        self.plastic_to_monomers = self._load_mapping(plastic_to_monomers_csv)

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
            zip(self.monomer_df["monomer"], self.monomer_df["canonical_smiles"])
        )

        self.selfies_lookup = dict(
            zip(self.monomer_df["monomer"], self.monomer_df["selfies"])
        )

        # -------------------------
        # Add plastic → ID mapping
        # -------------------------
        unique_plastics = sorted(self.plastic_df["Plastic"].unique())
        self.plastic_to_id = {p: i for i, p in enumerate(unique_plastics)}
        self.id_to_plastic = {i: p for p, i in self.plastic_to_id.items()}

    def _load_mapping(self, csv_path):
        """
        Build dict {plastic_name -> [monomer1, monomer2, ...]}
        """
        df = pd.read_csv(csv_path)
        mapping = {}
        for _, row in df.iterrows():
            plastic = str(row["Plastic"]).strip().lower()
            monomers = str(row["Monomers"]).split(";")
            monomers = [m.strip() for m in monomers if m.strip()]
            mapping[plastic] = monomers
        return mapping

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

        for m in monomers:
            if m in self.smiles_lookup:
                smiles.append(self.smiles_lookup[m])

        return smiles

    def get_selfies_for_plastic(self, plastic):
        monomers = self.get_monomers_for_plastic(plastic)
        selfies = []

        for m in monomers:
            if m in self.smiles_lookup:
                selfies.append(self.selfies_lookup[m])

        return selfies

        # -------------------------
        # NEW HELPER METHODS FOR MAIN.PY
        # -------------------------

    def get_plastic_id(self, plastic_name: str) -> int:
        """
        Convert a plastic name to its integer ID.
        Normalizes the name first.
        Returns -1 if the plastic is unknown.
        """
        plastic_norm = self._normalize_plastic_name(plastic_name)
        return self.plastic_to_id.get(plastic_norm, -1)

    def get_plastic_name(self, plastic_id: int) -> str:
        """
        Return plastic name from an integer ID.
        """
        return self.id_to_plastic.get(plastic_id, "unknown")

    def get_all_plastics(self):
        """
        Returns a list of all plastics that have monomer mappings
        """
        return list(self.plastic_to_monomers.keys())

