import os
import time

import numpy as np
import pandas as pd
import pubchempy as pcp
from matplotlib import pyplot as plt

from fingerprint_enc_dec.monomer_list import MONO


"""
estimates percentage max similarity > th over 10,000 instances
th   pct
0.0  1.0
0.05 0.9975
0.1  0.9969
0.15 0.9923
0.2  0.9672
0.25 0.9322
0.3  0.8757
0.35 0.7907
0.4  0.6651
0.45 0.532
0.5  0.4149
0.55 0.3125
0.6  0.2263
0.65 0.159
0.7  0.1017
0.75 0.0584
0.8  0.0328
0.85 0.0168
0.9  0.0092
0.95 0.0035
1.0  0.0019
"""

BATCH_PATH = r'batches'
TH = 0.65  # ~16%

MINI_BATCH_SIZE = 1000  # Needs to be a multiple of 10
BATCH_SIZE = 2_000_000
NUM_BATCHES = 120_000_000 // BATCH_SIZE
STARTING_BATCH = 0

def run():
    mono_df = get_monomers()
    # with pd.option_context('display.max_columns', None, 'display.max_rows', None, 'display.width', 1000, 'display.expand_frame_repr', False):
    #     print(mono_df)
    mono_df["bits_cactvs_fingerprint"] = mono_df["cactvs_fingerprint"].apply(bin_fp_to_bits)
    mono_df["max_tanimoto"] = calc_max_tanimoto_similarity(mono_df["bits_cactvs_fingerprint"], mono_df["bits_cactvs_fingerprint"])
    store_batch(mono_df, 'monomers')

    for batch_id in range(NUM_BATCHES):
        mini_batches = []
        start_indx = batch_id*BATCH_SIZE +1
        for end_indx in range(start_indx+MINI_BATCH_SIZE, BATCH_SIZE+2, MINI_BATCH_SIZE):
            s_time = time.time()
            batch_df = get_batch(start_indx, end_indx)
            print(start_indx, end_indx-1, time.time()-s_time)

            batch_df["bits_cactvs_fingerprint"] = batch_df["cactvs_fingerprint"].apply(bin_fp_to_bits)
            batch_df["max_tanimoto"] = calc_max_tanimoto_similarity(batch_df["bits_cactvs_fingerprint"], mono_df["bits_cactvs_fingerprint"])
            mini_batches.append(batch_df[batch_df["max_tanimoto"] >= TH])

            start_indx = end_indx

        batch_df = pd.concat(mini_batches, ignore_index=True)
        store_batch(batch_df, f'batch_{BATCH_SIZE}_{batch_id}')


def get_monomers():
    compounds = [pcp.get_compounds(monomer, 'name')[0].to_dict() for monomer in MONO]
    return pd.DataFrame(compounds)


def get_batch(start_id, end_id):
    compounds = pcp.get_compounds(np.arange(start_id, end_id).tolist(), 'cid')
    if len(compounds) == 0:
        compounds = _get_smaller_batches(start_id, end_id)

    df = pd.DataFrame([{
        "cid": c.cid,
        "iupac_name": c.iupac_name,
        "connectivity_smiles": c.connectivity_smiles,
        "smiles": c.smiles,
        "fingerprint": c.fingerprint,
        "cactvs_fingerprint": c.cactvs_fingerprint
    } for c in compounds])
    return df


def _get_smaller_batches(start_id, end_id):
    old_batch_size = end_id - start_id
    new_batch_size = 1 if old_batch_size < 10 else old_batch_size // 10

    all_compounds = []
    start_indx = start_id
    for end_indx in range(start_indx + new_batch_size, end_id + 1, new_batch_size):
        compounds = pcp.get_compounds(np.arange(start_indx, end_indx).tolist(), 'cid')
        if len(compounds) == 0:
            if new_batch_size > 1:
                all_compounds += _get_smaller_batches(start_indx, end_indx)
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


def store_batch(df, name):
    df = df[["cid", "iupac_name", "connectivity_smiles", "smiles", "fingerprint", "cactvs_fingerprint", "max_tanimoto"]]
    df.to_csv(os.path.join(BATCH_PATH, f"{name}.csv"), index=False)


if __name__ == "__main__":
    run()
