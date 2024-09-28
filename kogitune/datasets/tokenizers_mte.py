
from typing import List
from transformers import PreTrainedTokenizerFast
from ..loads.commons import adhoc, VerboseCounter

class MTEEncoding(object):
    """
    FastTokenizerのEncodingをアドホックにラップする
    FIXME:足らないプロパティは順次追加してください
    """
    def __init__(self, enc, orig_size:int, bases:List[int], start:int, end:int):
        self.enc = enc
        self.orig_size = orig_size
        self.bases = bases
        self.start = start
        self.end = end
        self.length = end - start
        self.cached_ids = None

    def encode(self, ids:List[int]):
        new_ids = []
        for token_id in ids:
            if token_id < self.orig_size:
                new_ids.append(token_id)
            else:
                extra_id = token_id-self.orig_size
                index = extra_id // self.length
                offset = extra_id % self.length
                new_ids.append(self.bases[index])
                new_ids.append(offset+self.start)
        return new_ids

    @property
    def ids(self):
        if self.cached_ids is None:
            self.cached_ids = self.encode(self.enc.ids)
        return self.cached_ids

    @property
    def type_ids(self):
        return self.enc.type_ids[:1] * len(self.ids)

    @property
    def attention_mask(self):
        return self.enc.attention_mask[:1] * len(self.ids)

    @property
    def special_tokens_mask(self):
        return self.enc.special_tokens_mask[:1] * len(self.ids)

    # Add the missing n_sequences attribute
    @property
    def n_sequences(self):
#        print('@@', len(self.ids), self.enc.n_sequences)
        return self.enc.n_sequences

def decode_mte(ids:List[int], orig_size:int, bases:List[int], start:int, end:int):
    if isinstance(ids[0], list):
        return [decode_mte(a, orig_size, bases, start, end) for a in ids]
    new_ids=[]
    offset = None
    for token_id in ids:
        if token_id in bases:
            index = bases.index(token_id)
            offset = orig_size + index * (end - start)
            continue
        if offset is None:
            new_ids.append(token_id)
        else:
            new_ids.append(token_id + offset - start)
            offset = None
    return new_ids

def wrap_encode(original_method):
    def mte_encode(self, *args, **kwargs):
        encoding = original_method(self, *args, **kwargs)
        if hasattr(self, "mte_bases"):
            bases = self.mte_bases
            orig_size = self.mte_orig_size
            start = self.mte_start
            end = self.mte_end
            return MTEEncoding(encoding, orig_size, bases, start, end)
        return encoding
    return mte_encode

def wrap_encode_batch(original_method):
    def mte_encode(self, *args, **kwargs):
        encoding = original_method(self, *args, **kwargs)
        if hasattr(self, "mte_bases"):
            bases = self.mte_bases
            orig_size = self.mte_orig_size
            start = self.mte_start
            end = self.mte_end
            return [MTEEncoding(enc, orig_size, bases, start, end) for enc in encoding]
        return encoding
    return mte_encode


def wrap_decode(original_method):
    def mte_decode(self, *args, **kwargs):
        if hasattr(self, "mte_bases"):
            orig_size = self.mte_orig_size
            bases = self.mte_bases
            start = self.mte_start
            end = self.mte_end
            args = [decode_mte(a, orig_size, bases, start, end) for a in args]
        return original_method(self, *args, **kwargs)
    return mte_decode

ALREADY_WRAPPED = False

def load_mte_config(tokenizer):
    global ALREADY_WRAPPED
    if 'mte_bases' in tokenizer.init_kwargs and hasattr(tokenizer, '_tokenizer'):
        tokenizer._tokenizer.mte_bases = tokenizer.init_kwargs['mte_bases']
        tokenizer._tokenizer.mte_start = tokenizer.init_kwargs['mte_start']
        tokenizer._tokenizer.mte_end = tokenizer.init_kwargs['mte_end']
        tokenizer._tokenizer.mte_orig_size = tokenizer.init_kwargs['mte_orig_size']
        if ALREADY_WRAPPED == False:
            Tokenizer = tokenizer._tokenizer.__class__
            Tokenizer.encode = wrap_encode(Tokenizer.encode)
            Tokenizer.encode_batch = wrap_encode_batch(Tokenizer.encode_batch)
            Tokenizer.decode = wrap_decode(Tokenizer.decode)
            Tokenizer.decode_batch = wrap_decode(Tokenizer.decode_batch)
            ALREADY_WRAPPED = True

def make_mte(tokenizer, new_tokens:List[str], bases:List[int], start=1000, end=None, /, **kwargs):
    import numpy as np
    #print('is_PreTrainedTokenizerFast', isinstance(tokenizer, PreTrainedTokenizerFast))
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    vocab_size = len(tokenizer)
    adhoc.print('Vocab. size//オリジナルのサブワード辞書の大きさ', vocab_size)
    before_tokens_len = []
    before_tokens = []
    good_tokens = []
    verbose = VerboseCounter(**kwargs)
    for token in new_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) > 2:
            good_tokens.append(token)
            before_tokens_len.append(len(token_ids))
            before_tokens.append(token_ids)
            verbose.print(token, len(token_ids), token_ids, face='  ')
    adhoc.print('Additional Vocab. size//追加される語彙の大きさ', len(new_tokens), '=>', len(good_tokens))

    tokenizer.add_tokens(good_tokens)
    if isinstance(bases[0], str):
        adhoc.print('特殊トークン', bases, end=' ')
        bases = tokenizer.convert_tokens_to_ids(bases)
        adhoc.print(bases)
    
    tokenizer.init_kwargs["mte_orig_size"]=vocab_size
    tokenizer.init_kwargs["mte_bases"]=bases
    tokenizer.init_kwargs["mte_start"]=start
    tokenizer.init_kwargs["mte_end"]=end or vocab_size
    load_mte_config(tokenizer)

    after_tokens_len = []
    after_tokens = []
    verbose = VerboseCounter(**kwargs)
    for token in good_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        after_tokens_len.append(len(token_ids))
        after_tokens.append(token_ids)
        verbose.print(token, len(token_ids), token_ids, tokenizer.decode(token_ids), face='  ')

    tables = {'token': good_tokens, 
             'before_len': before_tokens_len, 
             'before': before_tokens,
             'after_len': after_tokens_len,
             'after': after_tokens}
    adhoc.print(f'平均トークン数 {np.mean(before_tokens_len)} => {np.mean(after_tokens_len)}')
    return tokenizer, tables
