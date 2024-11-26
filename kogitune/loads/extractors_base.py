from .commons import *
from .patterns_ import Extractor
import re

## 行単位の処理

@adhoc.reg("none")
class NoneExtractor(Extractor):
    """
    行単位の処理をするフィルター
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name='none'

    def extract(self, text:str) -> List[str]:
        return [text]

@adhoc.reg("lines")
class LinesExtractor(Extractor):
    """
    行単位の処理をするフィルター
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name='lines'

    def extract(self, text:str) -> List[str]:
        lines = []
        for line in text.splitlines():
            if line:
                lines.append(line)
            else:
                lines.append('')
        if len(lines) == 0:
            lines.append('') # lines[0] = '' を保証 
        return lines

@adhoc.reg("sentences")
class SentenceExtractor(Extractor):
    """
    行単位の処理をするフィルター
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name='sentences'

    def extract(self, text:str) -> List[str]:
        # 英語と日本語の文末句読点を含む正規表現
        sentences = re.split(r'(?<=[.?!]\s+|[。．？！])', text)
        return [s.strip() for s in sentences if s]

@adhoc.reg("chunks|chunk_lines")
class ChunkLines(Extractor):
    """
    行単位の処理をするフィルター
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.get(kwargs, 'max_length|=256')
        self.name='chunk_lines'

    def extract(self, text:str) -> List[str]:
        """
        指定されたルールに基づいてテキストを分割する関数。
        
        Parameters:
        text (str): 分割するテキスト
        max_length: 分割する単位

        Returns:
        list: 分割されたテキストのリスト
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0

        for line in lines:
            if line.strip() == "":
                # 空行が来たら分割
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
            elif current_length + len(line) + 1 > self.max_length:
                # 今までの行がmax_length文字を超えたら分割
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = len(line)
            else:
                # 行を追加
                current_chunk.append(line)
                current_length += len(line) + 1

        # 最後のチャンクを追加
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

