from ..loads.commons import *
from ..loads import Model
from .tasks import Task
from .templates import guess_template

class TextGeneration(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_mode = True
        self.name = f'{self.shots}-shot'

    def apply_template(self, sample:dict, template:dict):
        sample["_input"] = self.format(template, "prompt", sample)
        if "reference" in template:
            sample["_reference"] = self.format(template, "reference", sample)
        # if 'test' in template:
        #     sample["_test"] = self.format(template, "test", sample)

    def eval(self, model:Model, samples: List[dict]):
        prompts = self.extract_values(samples, "_input")
        if self.extra_prompt:
            prompts = [f"{prompt}{self.extra_prompt}" for prompt in prompts]
        if self.chat_mode:
            prompts = model.transform_messages(prompts, heading=self.heading_messages)
            adhoc.verbose_print('[Chat Message]', dump=prompts, once="chat_message")
        gen_args = model.filter_gen_args(**self.init_kwargs)
        output_texts = model.generate(prompts, self.progress_bar, **gen_args)
        self.update_kwargs(samples, _model=model.modeltag, _task=self.name)
        self.update_values(samples, {"_prompt": prompts, "_output": output_texts})
    
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
        self.chat_mode = False

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
        from ..loads.metrics_python import openai_extract_code

        inputs = self.extract_values(samples, "_input")
        outputs = self.extract_values(samples, "_output")
        candidates = self.merge(inputs, outputs, openai_extract_code)
        testcases = self.extract_values(samples, "_test")
        results = metric.calc(candidates, testcases)
        self.update_values(samples, results)
        return results

CodeEval.register("code_eval|pass@1|pass@k")

class SelfCheckGPT(TextGeneration):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'selfcheck'
        self.n = adhoc.get(kwargs, 'num_return_sequences|n|=1')

    def apply_template(self, sample:dict, template:dict):
        sample["_input"] = self.format(template, "prompt", sample)

    def eval(self, model, samples: List[dict]):
        input_texts = self.extract_values(samples, "_input")
        gen_args = model.filter_gen_args(**self.init_kwargs)

        output_texts = model.generate(input_texts, _n=1, _do_sample=False, **gen_args)
        self.update_values(samples, {"_reference": output_texts})

        n = adhoc.get(gen_args, "num_return_sequences|n|=6")
        if "temperature" not in gen_args:
            adhoc.notice("temperatureを設定してね. temperature=1.0")
            gen_args["temperature"] = 1.0
        output_texts = model.generate(input_texts, self.progress_bar, _n=n, _do_sample=True, **gen_args)
        self.update_kwargs(samples, _model=model.modeltag, _task=self.name)
        self.update_values(samples, {"_output": output_texts})

SelfCheckGPT.register('selfcheck')

