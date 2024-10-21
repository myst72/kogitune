from ..loads.commons import *
from ..loads import Model, Extractor
from .tasks_textgen import TextGeneration

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


class OpenAIStylePythonExtractor(Extractor):
    """
    OpenAI HumanEval論文のコード抽出手法
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract(self, text:str, join:Optional[str]=None):
        stop_sequences = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
        min_stop_index = len(text)
        for seq in stop_sequences:
            stop_index = text.find(seq)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return "\n" + text[:min_stop_index]

OpenAIStylePythonExtractor.registor('humaneval')

class PythonSyntacticExtractor(Extractor):
    """
    Python構文をチェックしながらPythonコードらしき部分を抽出する
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ast = adhoc.safe_import('ast')

    def extract(self, text:str, join:Optional[str]=None):
        extracted_line = []
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            if lines[i].strip() == '':
                # 空行はスキップする
                i += 1
                continue
            code = '\n'.join(lines[i:])
            next = self.get_syntax_error_line(code)
            #print(i, next, code)
            if next == 1:
                # 先頭でエラーが発生したらスキップする
                i += 1
                continue
            if next is None:
                extracted_line.append(code)
                break
            code = self.clean_code('\n'.join(lines[i:i+next-1]))
            if code is not None:
                extracted_line.append(code)
            i += next
        return '\n'.join(extracted_line)

    def get_syntax_error_line(self, code):
        try:
            self.ast.parse(code)
            return None  # エラーがない場合はNoneを返す
        except SyntaxError as e:
            return e.lineno  # エラーが発生した行番号を返す

    def clean_code(self, code):
        while True:
            error_lineno = self.get_syntax_error_line(code)
            if error_lineno is None:
                return code
            if '\n' not in code:
                break
            code, _, _ = code.rpartition('\n')
        return None


PythonSyntacticExtractor.registor('python')


