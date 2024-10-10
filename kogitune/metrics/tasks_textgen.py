from ..loads import Model
from ..loads.commons import *
from .loads import Task

class TextGeneration(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = '0-shot'
        self.n = 1
        self.few_shots = adhoc.get(kwargs, 'few_shots')

    def guess_template(self, sample):
        return guess_template(sample)

    def apply_template(self, sample:dict, template:dict):
        sample["_input"] = self.format(template, "prompt", sample)
        if "reference" in template:
            sample["_reference"] = self.format(template, "reference", sample)
        # if 'test' in template:
        #     sample["_test"] = self.format(template, "test", sample)


    def eval(self, model:Model, samples: List[dict]):
        input_texts = self.extract_values(samples, "_input")
        gen_args = model.safe_gen_args(**self.init_kwargs)
        output_texts = model.generate(input_texts, self.n, self.progress_bar, **gen_args)
        self.update_kwargs(samples, _model=model.modeltag, _task=self.name)
        self.update_values(samples, {"_output": output_texts})
    
    def calc(self, metric, samples: List[dict]):
        candidates = self.extract_values(samples, "_output")
        references = self.extract_values(samples, "_reference")
        results = metric.calc(candidates, references)
        self.update_values(samples, results)
        return results

TextGeneration.register('|generation|0-shot|few-shots')

class CodeEval(TextGeneration):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = adhoc.get(kwargs, 'num_return_sequences|n|=1')

    def apply_template(self, sample:dict, template:dict):
        sample["_input"] = self.format(template, "prompt", sample)
        sample["_reference"] = self.format(template, "reference", sample)
        sample["_test"] = self.format(template, "test", sample)

    def merge(self, heads: List[str], tails: List[List[str]], merge_fn):
        merged_list = []
        for head, tail in zip(heads, tails):
            merged_list.append([merge_fn(head, x) for x in listfy(tail)])
        return merged_list

    @property
    def default_metrics():
        return ['pass@1']

    def calc(self, metric, samples: List[dict]):
        from .metrics_python import openai_extract_code
        inputs = self.extract_values(samples, "_input")
        outputs = self.extract_values(samples, "_output")
        candidates = self.merge(inputs, outputs, openai_extract_code)
        testcases = self.extract_values(samples, "_test")
        results = metric.calc(candidates, testcases)
        self.update_values(samples, results)
        return results

CodeEval.register("code_eval|pass@1|pass@k")

class SelfCheckGPT(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = adhoc.get(kwargs, 'num_return_sequences|n|=1')

    def apply_template(self, sample:dict, template:dict):
        sample["_input"] = self.format(template, "prompt", sample)

    def eval(self, model, samples: List[dict]):
        input_texts = self.extract_values(samples, "_input")
        output_texts = model.generate(input_texts, 1)
        self.update_values(samples, {"_reference": output_texts})

        # n = adhoc.get(kwargs, "num_return_sequences|n|N|=1")
        # if n == 1:
        #     adhoc.notice("num_return_sequencesを設定してね. num_return_sequences=6")
        #     kwargs["num_return_sequences"] = 6
        # if "temperature" not in kwargs:
        #     adhoc.notice("temperatureを設定してね. temperature=0.8")
        #     kwargs["temperature"] = 1.0
        # if "do_sample" not in kwargs:
        #     kwargs["do_sample"] = True

        output_texts = model.generate(input_texts, self.n)
        self.update_kwargs(samples, _model=model.modeltag, _task=self.name)
        self.update_values(samples, {"_output": output_texts})
    

SelfCheckGPT.register('selfcheck')



def has_schema(data: dict, keys:str):
    for key in keys.split('|'):
        if key not in data:
            return False
    return True

def contains_japanese(text: str) -> bool:
    for char in text:
        if '\u3040' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF' or '\uFF66' <= char <= '\uFF9D':
            return True
    return False

def guess_template(sample: dict):
    if has_schema(sample, 'instruction|input|output'):
        # Alpaca形式
        return {
            "prompt": "{instruction}\n{input}",
            "reference": "{output}",
        }
    if has_schema(sample, 'question|answer|answer_number|equation_solution'):
        # MSGM形式
        if contains_japanese(sample['question']):
            return {
                "prompt": "{question}\n(答)",
                "reference": "{answer_number}",
                "prompt_n": "{question}\n(答)",
            }
        else:
            return {
                "prompt": "{question}\n(Answer) ",
                "reference": "{answer_number}",
                "prompt_n": "{question}\n(Answer) ",
            }

    if has_schema(sample, 'prompt|test|entry_point|canonical_solution'):
        # HumanEval
        return {
            "prompt": "{prompt}",
            "reference": "{canonical_solution}\n",
            "test": "\n{test}\n\ncheck({entry_point})\n",
            "back_translation": "{instruction}\n\n{prompt}{canonical_solution}\n",
        }
    if has_schema(sample, 'question|choice0|choice1|choice2|choice3|choice4|label'):
        # JCommonSenseQA
        return {
            "prompt": "{question}\n選択肢(Choice): [0] {choice0} [1] {choice1} [2] {choice2} [3] {choice3} [4] {choice4}\n",
            "reference": "{label}",
            "choice": ["0", "1", "2", "3", "4"],
            "prompt_0": "{question}\n{choice0}",
            "prompt_1": "{question}\n{choice1}",
            "prompt_2": "{question}\n{choice2}",
            "prompt_3": "{question}\n{choice3}",
            "prompt_4": "{question}\n{choice4}",
        }
    if has_schema(sample, 'question|A|B|C|D|answer'):
        # JMMLU
        if contains_japanese(sample['question']):
            return {
                "prompt": "{question}\n選択肢(Choice): [A] {A} [B] {B} [C] {C} [D] {D}\n",
                "reference": "{answer}",
                "choice": ["A", "B", "C", "D"],
                "choice_A": "{question}\n{A}",
                "choice_B": "{question}\n{B}",
                "choice_C": "{question}\n{C}",
                "choice_D": "{question}\n{D}",
            }
        else:
            return {
                "prompt": "{question}\n(Choice): [A] {A} [B] {B} [C] {C} [D] {D}\n",
                "reference": "{answer}",
                "choice": ["A", "B", "C", "D"],
                "choice_A": "{question}\n{A}",
                "choice_B": "{question}\n{B}",
                "choice_C": "{question}\n{C}",
                "choice_D": "{question}\n{D}",
            }
    if has_schema(sample, 'text'):
        # Kogitune 事前学習形式
        return {
            "prompt": "{text}",
            "reference": "",
        }
    if has_schema(sample, 'prompt'):
        # Kogitune 標準形式
        return {
            "prompt": "{prompt}",
        }
    return None
