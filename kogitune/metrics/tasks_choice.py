import math

from ..loads.commons import *
from .loads import Task, Metric
from .tasks_textgen import guess_template

class QAChoice(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'choice'
        self.template = None

    def guess_template(self, sample):
        return guess_template(sample)

    def apply_template(self, sample:dict, template:dict):
        sample["_choice"] = template['choice']
        sample["_reference"] = self.format(template, "reference", sample)
        for c in template["choice"]:
            sample[f"_choice_{c}"] = self.format(template, f"choice_{c}", sample)

    def eval(self, model, samples: List[dict]):
        for sample in samples:
            sample['_model'] = model.modeltag
            sample['_task'] = self.tasktag
            choices = [sample[f"_choice_{c}"] for c in sample["_choice"]]
            sample["_losses"] = scores = model.compute_loss(choices)
            predicted_idx = scores.index(min(scores))
            sample["_output"] = sample["_choice"][predicted_idx]
            if self.progress_bar:
                self.progress_bar.update(1)

    @property
    def default_metrics(self):
        return ['exact_match']

QAChoice.register('choice')


class MIA(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'mia'
        self.input_key = adhoc.get(kwargs, 'input_key|text_key')

    def guess_template(self, sample):
        if self.input_key and self.input_key in sample:
            return {
                "text": f"{{{self.input_key}}}"
            }
        return guess_template(sample)
        
    def apply_template(self, sample:dict, template:dict):
        if "prompt" in template and "reference" in template:
            sample["_input"] = self.format(template, "prompt", sample) + self.format(template, "reference", sample)
        else:
            sample["_input"] = self.format(template, "text", sample)

    def eval(self, kmodel, samples: List[dict]):
        import numpy as np
        import torch
        import torch.nn.functional as F
        import zlib

        tokenizer = kmodel.tokenizer
        model = kmodel.model
        for sample in samples:
            text = sample["_input"]
            input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
            input_ids = input_ids.to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            ll = -loss.item() # log-likelihood

            # assuming the score is larger for training data
            # and smaller for non-training data
            # this is why sometimes there is a negative sign in front of the score

            # loss and zlib
            sample['_model'] = kmodel.modeltag
            sample['_task'] = self.tasktag
            sample['_loss'] = ll
            sample['_zlib'] = (ll / len(zlib.compress(bytes(text, 'utf-8'))))

            # mink and mink++
            input_ids = input_ids[0][1:].unsqueeze(-1)
            probs = F.softmax(logits[0, :-1], dim=-1)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

            ## mink
            scores = {}
            for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                k_length = int(len(token_log_probs) * ratio)
                topk = np.sort(token_log_probs.cpu())[:k_length]
                scores[f'{int(ratio*100)}'] = np.mean(topk).item()
            sample['_mink_prob'] = scores

            ## mink++
            scores = {}
            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                k_length = int(len(mink_plus) * ratio)
                topk = np.sort(mink_plus.cpu())[:k_length]
                scores[f'{int(ratio*100)}'] = np.mean(topk).item()
            sample['_mink++'] = scores
            if self.progress_bar:
                self.progress_bar.update(1)

    def find_auc_label(self, samples):
        label_keys = []
        if len(samples) > 0:
            sample = samples[0]
            for key in samples[0]:
                if sample[key] in (0, 1):
                    labels = self.extract_values(samples, key)
                    if labels.count(0) + labels.count(1) == len(labels):
                        label_keys.append(key)
        if len(label_keys) > 0:
            adhoc.verbose_print('AUCラベルの候補', label_keys)
            return label_keys[0]
        return None

    def calc(self, metric, samples: List[dict]):
        scores = []
        try:
            for sample in samples:
                scores.append(metric.recalc(sample[metric.score_key]))
        except BaseException as e:
            adhoc.debug_print(f"うまく計算できません。{metric.score_key}")
            adhoc.debug_print(repr(e))
            adhoc.debug_print(adhoc.dump(sample))
            return {}
        results = { metric.name: ('mean', scores) }
        self.update_values(samples, results)        
        auc_label = adhoc.get(self.init_kwargs, "auc_label|roc_label")
        if not auc_label:
            auc_label = self.find_auc_label(samples)
        if auc_label:
            return {metric.name: (f"AUC:{auc_label}", scores)}
        adhoc.verbose_print("AUCを計算するには、auc_label='label'", color="yellow", once=True)
        return results

MIA.register('mia|loss|perplexity')

class Loss(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'loss'
        self.score_key = '_loss'

    def check(self, samples:dict):
        if len(samples) > 0 and self.score_key in samples[0]:
            return True
        adhoc.print(f'{self.score_key} is not found///{self.score_key}が見つかりません', color='magenta')
        adhoc.print(f'評価タスク(mia)が正しく指定されていないことが原因です', color='magenta', face='')
        return False
    
    def recalc(self, result):
        return float(result)

Loss.register('loss')

class Perplexity(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'perplexity'
        self.score_key = '_loss'

    def recalc(self, result):
        return math.exp(result)

Perplexity.register('perplexity|ppl')

class MinKProb(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = str(kwargs.get("k", 10))
        self.name =f"min{self.k}_prob"
        self.score_key = '_mink_prob'

    def recalc(self, result:dict):
        return result[self.k]


MinKProb.register("mink_prob")

class MinKPlusPlus(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = str(kwargs.get("k", 10))
        self.name =f"min{self.k}++"
        self.score_key = '_mink++'

    def recalc(self, result:dict):
        return result[self.k]

MinKPlusPlus.register("mink++")
