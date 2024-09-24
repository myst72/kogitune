from typing import List
from .commons import adhoc

EOS_TOKEN = "</s>"
EOS_ID = 0
UNK_TOKEN = "<unk>"
UNK_ID = 1
MASK_TOKEN = "…"
MASK_ID = 2

SPECIAL_TOKENS = [EOS_TOKEN, UNK_TOKEN, MASK_TOKEN]

VOCAB = [
    (EOS_TOKEN, 0.0),  # EOD
    (UNK_TOKEN, 0.0),  # UNK
    (MASK_TOKEN, 0.0),  # MASK
    ("\n", -0.01),  # 改行
]

@adhoc.cli
def train_bpe(files:List[str], save_path:str, **kwargs):
    tokenizers = adhoc.safe_import('tokenizers')
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    tokenizer = Tokenizer(models.BPE())
    byte_level = adhoc.get("bytelevel|byte_level|=True")
    if byte_level:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.UnicodeScripts(), pre_tokenizers.Whitespace()]
        )
        # デコーダーの設定
        tokenizer.decoder = decoders.WordPiece(prefix="##")

    # トレーナーの設定
    trainer = trainers.BpeTrainer(
        vocab_size=adhoc.get(kwargs, "vocab_size|=32000"),
        min_frequency=adhoc.get(kwargs, "min_frequency|=2"),
        special_tokens=SPECIAL_TOKENS,
    )
    with adhoc.start_timer() as timer:
        adhoc.notice("BPEモデルの訓練を始めます", options=kwargs)
        # トークナイザーのトレーニング
        tokenizer.train(files, trainer)
        timer.notice("お疲れ様です。")

    if save_path:
        save_tokenizer(tokenizer, save_path=save_path)
        display_token_fraction(files, save_path)


def save_tokenizer(tokenizer, save_path, text="🦊 token化、大丈夫？"):
    from transformers import PreTrainedTokenizerFast
    # FastTokenizerの作成
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        eos_token=EOS_TOKEN,
        mask_token=MASK_TOKEN,
    )
    print(fast_tokenizer.get_vocab())
    fast_tokenizer.save_pretrained(save_path)
    print(text)
    encoded = fast_tokenizer.encode(text)
    print(len(encoded), encoded)
    decoded = fast_tokenizer.decode(encoded)
    print(decoded)



def display_token_fraction(files, save_path):
    from transformers import AutoTokenizer
    import pandas as pd
    PERCENTILES = [0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 0.95, 0.99]

    tokenizer = AutoTokenizer.from_pretrained(save_path)
    text_lengths = []
    token_lengths = []
    fractions = []
    with open(files[0], "r", encoding="utf-8") as fin:
        for line in fin:
            text_length = len(line)
            if text_length > 40:
                tokens = tokenizer.encode(line)
                token_length = len(tokens)
                text_lengths.append(text_length)
                token_lengths.append(token_length)
                fractions.append(token_length / text_length)
    df = pd.DataFrame(
        {"token": token_lengths, "text": text_lengths, "token/text": fractions}
    )
    print(df.describe(percentiles=PERCENTILES))
    df.to_csv(f"{save_path}/token_fraction.csv", index=False)
    with open(f"{save_path}/token_fraction_describe.txt", "w") as w:
        print(df.describe(percentiles=PERCENTILES), file=w)
