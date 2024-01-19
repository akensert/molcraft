
class SpecialTokens:

    BOS = '[BOS]'       # beginning of sequence
    EOS = '[EOS]'       # ending of sequence
    UNK = '[UNK]'       # unknown token
    MASK = '[MASK]'     # mask AND padding token
   

SMILES_REGEX_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|="
    r"|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)
