# src/data/smiles_tokenizer.py
import re


class SMILESTokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<SEP>"]
        self.char2idx = {}
        self.idx2char = {}

    def fit(self, smiles_list):
        chars = set()
        for s in smiles_list:
            chars.update(list(s))

        vocab = self.special_tokens + sorted(chars)
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def encode(self, smiles):
        return (
            [self.char2idx["<SOS>"]]
            + [self.char2idx[c] for c in smiles]
            + [self.char2idx["<EOS>"]]
        )

    def decode(self, tokens):
        chars = []
        for t in tokens:
            c = self.idx2char.get(t, "")
            if c in ("<SOS>", "<EOS>", "<PAD>"):
                continue
            chars.append(c)
        return "".join(chars)

    @property
    def vocab_size(self):
        return len(self.char2idx)


SMILES_TOKEN_PATTERN = re.compile(
    r"(\%\d{2}|\[[^\]]+\]|Br|Cl|Si|Na|Ca|Li|Mg|Al|"
    r"[BCNOSPFI]|b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|@|\?|>|\*|\$|"
    r"\d)"
)


def tokenize_smiles(smiles):
    tokens = SMILES_TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"Failed to tokenize SMILES: {smiles}")
    return tokens


class RDKitSMILESTokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.token2idx = {}
        self.idx2token = {}

    def fit(self, smiles_list):
        vocab = set()
        for s in smiles_list:
            vocab.update(tokenize_smiles(s))

        vocab = self.special_tokens + sorted(vocab)
        self.token2idx = {t: i for i, t in enumerate(vocab)}
        self.idx2token = {i: t for t, i in self.token2idx.items()}

    def encode(self, smiles):
        tokens = tokenize_smiles(smiles)
        return (
            [self.token2idx["<SOS>"]]
            + [self.token2idx.get(t, self.token2idx["<UNK>"]) for t in tokens]
            + [self.token2idx["<EOS>"]]
        )

    def decode(self, indices):
        tokens = []
        for i in indices:
            t = self.idx2token.get(i, "")
            if t in self.special_tokens:
                continue
            tokens.append(t)
        return "".join(tokens)

    @property
    def vocab_size(self):
        return len(self.token2idx)
