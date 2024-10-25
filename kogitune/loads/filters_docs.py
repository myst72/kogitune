from .commons import *
from .filters_ import TextFilter
import unicodedata

class UnicodeNormalization(TextFilter):
    """
    Unicode正規化フィルターする
    """
    def __init__(self, **kwargs):
        """
        Unicode正規化フィルタを作る
        """
        super().__init__(**kwargs)
        self.get(kwargs, 'unicode_form|for|=NFKC')
        
    def filter_text(self, text:str) -> Optional[str]:
        return unicodedata.normalize(self.unicode_form, text)

class DuplicatedLineFilter(TextFilter):
    """
    重複した行を取り除く
    """
    def __init__(self, **kwargs):
        """
        重複した行を取り除くフィルタを作る
        :param prefix_length: 重複をチェックする先頭の文字数
        """
        super().__init__(**kwargs)
        super().__init__(**kwargs)
        self.get(kwargs, 'prefix_length|=8')

    def filter_text(self, text: str) -> Optional[str]:
        lines = ['']
        for line in text.split('\n'):
            prev = lines[-1]
            if self.prefix_length < len(line) < 80 and prev.startswith(line[:self.prefix_length]):
                if len(line) > len(prev):
                    lines[-1] = line
                continue
            if len(line.strip()) == 0 and prev == '':
                continue
            if 1 < len(prev) < 40 and not prev.endswith('。') and len(line) > 80 and line.count('。') > 1:
                if lines[-1] != '':
                    lines[-1] = f'\n{prev}'
            lines.append(line)
        return '\n'.join(lines[1:])


