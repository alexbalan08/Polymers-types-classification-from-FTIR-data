import os
import time

import numpy as np
import pandas as pd
import pubchempy as pcp
from matplotlib import pyplot as plt

from fingerprint_enc_dec.missing_cids import NON_EXISTING_CID


BATCH_PATH = r'batches'
TAN_TH = 0.0
LEN_TH = 40

MINI_BATCH_SIZE = 1000
BATCH_SIZE = 100_000
NUM_BATCHES = 120_000_000 // BATCH_SIZE
STARTING_BATCH = 0

def run():
    load_all()
    filter_all()

def load_all():
    for batch_id in range(STARTING_BATCH, NUM_BATCHES):
        mini_batches = []
        start_indx = batch_id*BATCH_SIZE +1
        for end_indx in range(start_indx+MINI_BATCH_SIZE, start_indx+BATCH_SIZE+2, MINI_BATCH_SIZE):
            s_time = time.time()

            batch_df = get_batch(start_indx, end_indx)
            mini_batches.append(batch_df)

            print(start_indx, end_indx-1, time.time()-s_time)
            start_indx = end_indx

        batch_df = pd.concat(mini_batches, ignore_index=True)
        store_batch(batch_df, f'batch_plus_features_{BATCH_SIZE}_{batch_id}',
                    max_tanimoto=False, additional_features=True)


def filter_all():
    mono_df = get_monomers()
    mono_df["SMILES_length"] = mono_df["connectivity_smiles"].str.len()
    mono_df["bits_cactvs_fingerprint"] = mono_df["cactvs_fingerprint"].apply(bin_fp_to_bits)
    # mono_df["max_tanimoto"] = calc_max_tanimoto_similarity(mono_df["bits_cactvs_fingerprint"], mono_df["bits_cactvs_fingerprint"])
    with pd.option_context('display.max_columns', None, 'display.max_rows', None, 'display.width', 1000, 'display.expand_frame_repr', False):
        print(mono_df)

    filtered_batches = []
    length_lim_sum = 0
    for batch_id in range(120):
        s_time = time.time()

        batch_df = pd.read_csv(os.path.join(r"batches", f"batch_plus_features_{BATCH_SIZE}_{batch_id}.csv"))

        batch_df["SMILES_length"] = batch_df["connectivity_smiles"].str.len()
        batch_df["bits_cactvs_fingerprint"] = batch_df["cactvs_fingerprint"].apply(bin_fp_to_bits)
        batch_df["max_tanimoto"] = calc_max_tanimoto_similarity(batch_df["bits_cactvs_fingerprint"], mono_df["bits_cactvs_fingerprint"])

        smiles_len_filter = batch_df["SMILES_length"] <= LEN_TH
        tanimoto_filter = batch_df["max_tanimoto"] >= TAN_TH
        filtered_batches.append(batch_df[smiles_len_filter & tanimoto_filter])
        print(batch_id, time.time() - s_time)

        print(smiles_len_filter.sum())
        length_lim_sum += smiles_len_filter.sum()
        print(tanimoto_filter.sum())
        print((smiles_len_filter & tanimoto_filter).sum())
        print()

        """import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 16})
        plt.figure(figsize=(8, 5))
        # plt.hist(batch_df["max_tanimoto"], bins=25)
        # plt.hist(batch_df["SMILES_length"], bins=200)
        plt.hist(filtered_batches[-1]["SMILES_length"], bins=LEN_TH-1)
        plt.title("Max tanimoto for 100,000 molecules")
        plt.xlabel("Max Tanimoto score")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()"""

    filtered_batches = pd.concat(filtered_batches, ignore_index=True)
    print(f"\nThe total would be {length_lim_sum} if length would be the only limit")
    store_batch(filtered_batches, f'filtered_data_{LEN_TH}_{int(TAN_TH*100)}')


def get_monomers():
    df = pd.read_csv(os.path.join(r"batches", f"monomers.csv"))
    return df


def get_batch(start_id, end_id):
    cids = np.arange(start_id, end_id)
    try:
        compounds = pcp.get_compounds(cids[~np.isin(cids, NON_EXISTING_CID)].tolist(), 'cid')
        if len(compounds) == 0:
            compounds = _get_smaller_batches(start_id, end_id)
    except Exception as e:
        print(f"Exception caught: {e}")
        compounds = _get_smaller_batches(start_id, end_id)

    df = pd.DataFrame([c.to_dict() for c in compounds])
    return df


def _get_smaller_batches(start_id, end_id):
    """
    When a single CID does not exist the whole batch is empty.
    This function uses binary search to identify the missing CID and retrieve the remaining CIDs.
    """
    mid_id = (start_id + end_id) //2

    all_compounds = []
    for start_indx, end_indx in [(start_id, mid_id), (mid_id, end_id)]:
        compounds = pcp.get_compounds(np.arange(start_indx, end_indx).tolist(), 'cid')
        if len(compounds) == 0:
            if end_indx - start_indx > 1:
                all_compounds += _get_smaller_batches_old(start_indx, end_indx)
            else:
                print(f"COMPOUND {start_indx} DOES NOT EXIST!")
        else:
            all_compounds += compounds

    return all_compounds


def _get_smaller_batches_old(start_id, end_id):
    old_batch_size = end_id - start_id
    new_batch_size = 1 if old_batch_size < 10 else old_batch_size // 10

    all_compounds = []
    start_indx = start_id
    for end_indx in range(start_indx + new_batch_size, end_id + 1, new_batch_size):
        compounds = pcp.get_compounds(np.arange(start_indx, end_indx).tolist(), 'cid')
        if len(compounds) == 0:
            if new_batch_size > 1:
                all_compounds += _get_smaller_batches_old(start_indx, end_indx)
            else:
                print(f"COMPOUND {start_indx} DOES NOT EXIST!")
        else:
            all_compounds += compounds

        start_indx = end_indx
    return all_compounds


def bin_fp_to_bits(binstr):
    return np.frombuffer(binstr.encode(), dtype="S1").astype(int)


def calc_max_tanimoto_similarity(s1, s2):
    matrix1 = np.stack(s1.values)
    matrix2 = np.stack(s2.values)

    intersection = matrix1 @ matrix2.T  # shape: (n2, n1)

    sum1 = matrix1.sum(axis=1).reshape(-1, 1)  # shape: (n2, 1)
    sum2 = matrix2.sum(axis=1).reshape(1, -1)  # shape: (1, n1)

    tanimoto = intersection / (sum1 + sum2 - intersection)
    return tanimoto.max(axis=1)


def store_batch(df, name, max_tanimoto=True, additional_features=False):
    columns = ["cid", "iupac_name", "connectivity_smiles", "smiles", "fingerprint", "cactvs_fingerprint"]
    if additional_features:
        columns += ["complexity", "exact_mass", "h_bond_acceptor_count", "h_bond_donor_count", "heavy_atom_count",
                    "molecular_weight", "rotatable_bond_count", "tpsa", "xlogp"]
    if max_tanimoto:
        columns.append("max_tanimoto")
    df = df[columns]
    df.to_csv(os.path.join(BATCH_PATH, f"{name}.csv"), index=False)


if __name__ == "__main__":
    run()
