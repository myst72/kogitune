import math
# import torch
from collections import Counter

from ..loads.commons import *
from .metrics import Metric


def levenshtein_similarity(candidate, reference):
    distance = adhoc.edit_distance(candidate, reference)
    max_length = max(len(candidate), len(reference))
    return 1 - (distance / max_length)


class EditSim(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "editsim"

    def calc_s(self, candidate: str, reference: str) -> float:
        return levenshtein_similarity(candidate, reference)

EditSim.register("editsim|levenshtein")

##
# 字句ベース


def jaccard_similarity(candidate, reference, tokenize):
    # テキストを単語に分割してセットに変換
    set1 = set(tokenize(candidate))
    set2 = set(tokenize(reference))

    # 積集合と和集合を計算
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Jaccard係数を計算
    return len(intersection) / len(union)


class Jaccard(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "jaccard"
        tokenizer_path = self.get(kwargs, "_subpath|tokenizer_path|tokenizer|=simple")
        self.tokenize = adhoc.load("tokenizer", tokenizer_path)

    def calc_s(self, candidate: str, reference: str) -> float:
        return jaccard_similarity(candidate, reference, tokenize=self.tokenize)


Jaccard.register("jaccard")


def cosine_similarity(candidate, reference, tokenize):
    # テキストを単語に分割
    words1 = tokenize(candidate)
    words2 = tokenize(reference)

    # 単語の出現回数をカウント
    word_count1 = Counter(words1)
    word_count2 = Counter(words2)

    # 共通の単語を抽出
    common_words = set(word_count1.keys()) & set(word_count2.keys())

    # 内積を計算
    numerator = sum(word_count1[word] * word_count2[word] for word in common_words)

    # ベクトルの大きさを計算
    sum1 = sum(word_count1[word] ** 2 for word in word_count1.keys())
    sum2 = sum(word_count2[word] ** 2 for word in word_count2.keys())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return numerator / denominator


class CosineSim(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "cosine"
        tokenizer_path = self.get(kwargs, "_subpath|tokenizer_path|tokenizer|=simple")
        self.tokenize = adhoc.load("tokenizer", tokenizer_path)

    def calc_s(self, candidate: str, reference: str) -> float:
        return cosine_similarity(candidate, reference, tokenize=self.tokenize)


CosineSim.register("cosine")

## BLEU


def ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def count_ngrams(tokens, n):
    return Counter(ngrams(tokens, n))


def clip_count(candidate_ngrams, reference_ngrams):
    return {
        ngram: min(count, reference_ngrams[ngram])
        for ngram, count in candidate_ngrams.items()
    }


def modified_precision(candidate, reference, n):
    candidate_ngrams = count_ngrams(candidate, n)
    reference_ngrams = count_ngrams(reference, n)
    clipped_counts = clip_count(candidate_ngrams, reference_ngrams)

    total_clipped_count = sum(clipped_counts.values())
    total_candidate_count = sum(candidate_ngrams.values())

    if total_candidate_count == 0:
        return 0

    return total_clipped_count / total_candidate_count


def brevity_penalty(candidate, reference):
    c = len(candidate)
    r = len(reference)

    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)


def bleu_score(candidate: str, reference: str, n=4, tokenize=None):
    candidate = tokenize(candidate)
    reference = tokenize(reference)

    precisions = [modified_precision(candidate, reference, i) for i in range(1, n + 1)]

    if any(p == 0 for p in precisions):
        return 0

    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / n)

    bp = brevity_penalty(candidate, reference)

    return bp * geometric_mean


class BLEU(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = kwargs.get("n", 4)
        self.name = f"bleu-{self.n}"
        tokenizer_path = self.get(kwargs, "_subpath|tokenizer_path|tokenizer|=simple")
        self.tokenize = adhoc.load("tokenizer", tokenizer_path)

    def calc_s(self, candidate: str, reference: str) -> float:
        return bleu_score(candidate, reference, n=self.n, tokenize=self.tokenize)


BLEU.register("bleu|blue")


class SacreBLEU(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f"(sacre)bleu"
        self.sacrebleu = adhoc.safe_import('sacrebleu')
        tokenizer_path = self.get(kwargs, "_subpath|tokenizer_path|tokenizer")
        if tokenizer_path:
            self.tokenize = adhoc.load("tokenizer", tokenizer_path)
        else:
            self.tokenize = None

    def calc_m(self, candidates: List[str], references: List[str], n=1, suffix='') -> float:
        if self.tokenize:
            candidates = [' '.join(self.tokenize(text)) for text in candidates]
            references = [[' '.join(self.tokenize(text))] for text in references]
            bleu = self.sacrebleu.corpus_bleu(candidates, references, tokenize='none')
        else:
            candidates = [text for text in candidates]
            references = [[text] for text in references]
            bleu = self.sacrebleu.corpus_bleu(candidates, references)
        return {f'{self.name}{suffix}': [bleu.score] * (len(references) // n)}

SacreBLEU.register("sacrebleu")

def load_cosine_similarity():
    from sklearn.metrics.pairwise import cosine_similarity
    # コサイン類似度の計算
    return cosine_similarity

class CosineSim(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f"cosine"
        self.torch = adhoc.safe_import('torch')
        self.model_path = self.get(kwargs, "_subpath|model_path|model|=sentence-transformers/all-MiniLM-L6-v2")
        self.model = None
        self.cosine_sim = load_cosine_similarity()

    def lazy_load(self):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)

    def get_embedding(self, text):
        if self.model is None:
            self.lazy_load()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        # 最後の隠れ層の平均を取りembeddingベクトルを得る
        embedding = self.torch.mean(outputs.last_hidden_state, dim=1)
        return embedding

    def calc_s(self, candidate: str, reference: str) -> float:
        candidate = self.get_embedding(candidate)
        reference = self.get_embedding(reference)
        return self.cosine_sim(candidate, reference)[0][0]

CosineSim.register("embsim|vecsim")

# ROUGE-L


def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


def rouge_l(candidate, reference, tokenize):
    # 単語単位でトークン化
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)

    lcs_length = lcs(candidate_tokens, reference_tokens)

    # Precision, Recall, F1-scoreの計算
    precision = lcs_length / len(candidate_tokens) if len(candidate_tokens) > 0 else 0
    recall = lcs_length / len(reference_tokens) if len(reference_tokens) > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    # return {
    #     "precision": precision,
    #     "recall": recall,
    #     "f1_score": f1_score
    # }
    return f1_score


class ROUGE_L(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ROUGE-L"
        tokenizer_path = self.get(kwargs, "_subpath|tokenizer_path|tokenizer|=simple")
        self.tokenize = adhoc.load("tokenizer", tokenizer_path)

    def calc_s(self, candidate: str, reference: str) -> float:
        return rouge_l(candidate, reference, tokenize=self.tokenize)


ROUGE_L.register("rouge_l")

# BERTScore

class BERTScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BERTScore"
        bert_score = adhoc.safe_import('bert_score')
        self.bert_score = bert_score.score
        self.model_path = adhoc.get(kwargs, "_subpath|model_type|model_path|model")
        self.lang = kwargs.get("lang", "en")

    def calc_s(self, candidate: str, reference: str) -> float:
        P, R, F1 = self.bert_score(
            [candidate], [reference], lang=self.lang, model_type=self.model_path
        )
        return F1.mean().item()

    def calc_m(self, candidates, references, n=1, suffix=''):
        P, R, F1 = self.bert_score(
            candidates, references, 
            model_type=self.model_path,
            lang=self.lang, 
        )
        return {
            f"{self.nametag}_F1{suffix}": self.flatten_mean(F1.numpy().tolist(), n),
            f"{self.nametag}_Precision{suffix}": self.flatten_mean(P.numpy().tolist(),n),
            f"{self.nametag}_Recall{suffix}": self.flatten_mean(R.numpy().tolist(),n),
        }


BERTScore.register("bertscore|bert_score")
