
from .commons import *
from .tokenizers_ import Tokenizer

import tokenize
from io import StringIO
from typing import List, Tuple

@adhoc.reg('python')
class PythonTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        self.name = "python"

    def __call__(self, code: str) -> List[str]:
        """
        Pythonコードをトークン化して、トークンのタイプと文字列のリストを返す

        Parameters:
        - code (str): Pythonコードの文字列

        Returns:
        - List[str]: トークン文字列のリスト
        """
        return get_tokens_with_whitespace(code)

def get_tokens_with_whitespace(code):
    tokens = []
    reader = StringIO(code).readline
    previous_end = (0, 0)  # 前のトークンの終端位置

    for tok in tokenize.generate_tokens(reader):
        token_string = tok.string
        start = tok.start
        end = tok.end

        # 前のトークンの終端と現在のトークンの始端の間に空白がある場合、その空白を追加
        if previous_end[0] == start[0] and previous_end[1] < start[1]:
            # print('@', previous_end, start)
            whitespace = ' ' * (start[1] - previous_end[1])
            tokens.append(whitespace)
        if token_string != '':
            tokens.append(token_string)
        previous_end = end

    return tokens

