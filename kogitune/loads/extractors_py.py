from .commons import *
from .patterns_ import Extractor


@adhoc.reg('codex')
class CodexExtractor(Extractor):
    """
    OpenAI HumanEval論文のコード抽出手法
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract(self, text:str) -> List[str]:
        stop_sequences = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
        min_stop_index = len(text)
        for seq in stop_sequences:
            stop_index = text.find(seq)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return ["\n" + text[:min_stop_index]]

@adhoc.reg('python')
class PythonSyntacticExtractor(Extractor):
    """
    Python構文をチェックしながらPythonコードらしき部分を抽出する
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ast = adhoc.safe_import('ast')

    def extract(self, text:str):
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
        return ['\n'.join(extracted_line)]

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

