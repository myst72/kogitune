from ..loads.commons import *
from ..loads import Model
from .tasks import Task

@adhoc.reg('|generation|0-shot|few-shots')
class TextGeneration(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_mode = True
        self.name = f'{self.shots}-shot'

    def apply_template(self, sample:dict, template:dict):
        sample["_input"] = self.format(template, "prompt", sample)
        if "reference" in template:
            sample["_reference"] = self.format(template, "reference", sample)

    def eval(self, model:Model, samples: List[dict]):
        prompts = self.column_values(samples, "_input")
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
        candidates = self.column_values(samples, "_output")
        if self.extractor:
            candidates = [self.extractor.extract(c)[-1] for c in candidates]
            self.update_values(samples, {"_extracted": candidates})
        references = self.column_values(samples, "_reference")
        results = metric.calc(candidates, references)
        self.update_values(samples, results)
        return results


@adhoc.reg('selfcheck')
class SelfCheckGPT(TextGeneration):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'selfcheck'
        self.n = adhoc.get(kwargs, 'num_return_sequences|n|=1')

    def apply_template(self, sample:dict, template:dict):
        sample["_input"] = self.format(template, "prompt", sample)

    def eval(self, model, samples: List[dict]):
        input_texts = self.column_values(samples, "_input")
        gen_args = model.filter_gen_args(**self.init_kwargs)

        output_texts = model.generate(input_texts, _n=1, _do_sample=False, **gen_args)
        self.update_values(samples, {"_output": output_texts})

        n = adhoc.get(gen_args, "num_return_sequences|n|=6")
        if "temperature" not in gen_args:
            adhoc.verbose_print("temperatureを設定してね. temperature=1.0", once="temperature=1.0")
            gen_args["temperature"] = 1.0
        output_texts = model.generate(input_texts, self.progress_bar, _n=n, _do_sample=True, **gen_args)
        self.update_kwargs(samples, _model=model.modeltag, _task=self.name)
        self.update_values(samples, {"_samples": output_texts})

    def calc(self, metric, samples: List[dict]):
        candidates = self.column_values(samples, "_output")
        if self.extractor:
            candidates = [self.extractor.extract(c)[-1] for c in candidates]
            self.update_values(samples, {"_extracted": candidates})
        _samples = self.column_values(samples, "_samples")
        results = metric.calc(candidates, _samples)
        self.update_values(samples, results)
        return results
