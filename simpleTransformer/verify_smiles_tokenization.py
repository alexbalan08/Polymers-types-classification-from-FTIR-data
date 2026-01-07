from src.data.smiles_tokenizer import SMILESTokenizer

smiles_examples = [
    "C=CC=C",
    "C=C<SEP>C=CC=C",
    "CC(C)COC(=O)C(=C)C"
]

tokenizer = SMILESTokenizer()
tokenizer.fit(smiles_examples)

for s in smiles_examples:
    encoded = tokenizer.encode(s)
    decoded = tokenizer.decode(encoded)

    print("\nOriginal:", s)
    print("Encoded:", encoded)
    print("Decoded:", decoded)

print("\nVocab size:", tokenizer.vocab_size)
