from typing import Union, List
import torch
import numpy as np
from ..loads.commons import *
from .chaineval import list_testdata

def calc_mia(model, sample_list: Union[List[dict], dict], **kwargs):
    import torch.nn.functional as F

    for sample in list_tqdm(sample_list, desc=f"{model}"):
        input_text = sample["input"]
        input_ids = torch.tensor(model.tokenizer.encode(input_text)).unsqueeze(0)
        input_ids = input_ids.to(model.model.device)
        with torch.no_grad():
            outputs = model.model(input_ids, labels=input_ids)
            # outputs = self.model(**inputs, labels=labels)
            loss, logits = outputs[:2]
        sample["loss"] = loss.item()  # log-likelihood
        # mink and mink++
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        ## mink
#        scores = {}
        for k in range(10, 101, 10):
            k_length = int(len(token_log_probs) * k // 100)
            topk = np.sort(token_log_probs.cpu())[:k_length]
            sample[f'min{k}_prob'] = -np.mean(topk).item()
#        sample["mink_prob"] = scores
        ## mink++
 #       scores = {}
        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        for k in range(10, 101, 10):
            k_length = int(len(mink_plus) * k // 100)
            topk = np.sort(mink_plus.cpu())[:k_length]
            sample[f'min{k}_plus'] = -np.mean(topk).item()
        model.verbose_sample(sample)


def eval_mia(model_list: List[str], /, **kwargs):
    board = adhoc.load('from_kwargs', 'leaderboard', **kwargs)
    for model_path in model_list:
        model = adhoc.load("model", model_path, extract_prefix="model", **kwargs)
        for testdata in list_testdata(model.modeltag, "mia", **kwargs):
            calc_mia(model, testdata.samples(), **kwargs)
            testdata.save()

    board.show()
