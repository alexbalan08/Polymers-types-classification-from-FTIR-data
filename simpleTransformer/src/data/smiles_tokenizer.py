import tensorflow as tf
import re


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


class SMILESTokenizer:
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

    def token_to_id(self, token):
        return self.token2idx.get(token, self.token2idx["<UNK>"])

    def id_to_token(self, idx):
        return self.idx2token.get(idx, "<UNK>")
