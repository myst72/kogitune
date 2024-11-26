from ..loads.commons import *
from .tasks_textgen import TextGeneration

@adhoc.reg("code_eval|pass@1|pass@k")
class CodeEval(TextGeneration):

    def __init__(self, **kwargs):
        self.chat_mode = adhoc.get(kwargs, 'chat_mode|=False')
        super().__init__(**kwargs)
        self.n = adhoc.get(kwargs, 'num_return_sequences|n|=1')

    def init_extractor(self, **kwargs):
        if self.chat_mode: # chat_mode かどうかでデフォルトのextractor切り替え。
            extractor = "extractor|=python"
        else:
            extractor = "extractor|=codex"
        self.extractor = adhoc.load('extractor', extractor, **kwargs)

    def apply_template(self, sample:dict, template:dict):
        sample["_input"] = self.format(template, "prompt", sample)
        sample["_reference"] = self.format(template, "reference", sample)
        sample["_test"] = self.format(template, "test", sample)

    def merge(self, prompts: List[str], tails: List[List[str]]):
        merged_list = []
        for prompt, output_texts in zip(prompts, tails):
            for output_text in listfy(output_texts):
                code = self.extractor.extract(output_text)[0]
                if not self.chat_mode: # chat_modeでないときは、プロンプトを先頭につける
                    code = prompt + code
                merged_list.append(code)
        adhoc.verbose_print('[Extracted Code]', dump=merged_list)
        return merged_list

    @property
    def default_metrics():
        return ['pass@1']

    def calc(self, metric, samples: List[dict]):
        inputs = self.column_values(samples, "_input")
        outputs = self.column_values(samples, "_output")
        candidates = self.merge(inputs, outputs)
        testcases = self.column_values(samples, "_test")
        results = metric.calc(candidates, testcases)
        self.update_values(samples, results)
        return results

