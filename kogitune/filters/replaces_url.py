from typing import List
import re
import regex

def RE(*patterns: List[str], flags=0):
    return regex.compile('|'.join(patterns), flags=flags)

def replace_pattern(pattern, text, replaced):
    return pattern.sub(replaced, text)

## URL


url_pattern = RE(
    r'https?://[\w/:%#\$&\?~\.\,=\+\-\\_]+', # \(\)
)

def replace_url(text, replaced=None):
    """
    text 中のURLを<url>に置き換える

    >>> replace_url("http://www.peugeot-approved.net/UWS/WebObjects/UWS.woa/wa/carDetail?globalKey=uwsa1_1723019f9af&currentBatch=2&searchType=1364aa4ee1d&searchFlag=true&carModel=36&globalKey=uwsa1_1723019f9af uwsa1_172febeffb0, 本体価格 3,780,000 円")
    '<url> uwsa1_172febeffb0, 本体価格 3,780,000 円'

    >>> replace_url("「INVADER GIRL!」https://www.youtube.com/watch?v=dgm6-uCDVt0")
    '「INVADER GIRL!」<url>'

    >>> replace_url("http://t.co/x0vBigH1Raシグネチャー")
    '<url>'

    >>> replace_url("(http://t.co/x0vBigH1Ra)シグネチャー")
    '(<url>)シグネチャー'

    >>> replace_url("kindleにあるなーw http://www.amazon.co.jp/s/ref=nb_sb_noss?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&url=search-alias%3Ddigital-text&field-keywords=%E3%82%A2%E3%82%B0%E3%83%8D%E3%82%B9%E4%BB%AE%E9%9D%A2")
    'kindleにあるなーw <url>'

    >>> replace_url("http://live.nicovideo.jp/watch/lv265893495 #nicoch2585696")
    '<url> #nicoch2585696'
    
    """
    text = replace_pattern(url_pattern, text, replaced or '<url>')
    return text

# 日付



def replace_address(text, replaced=None):
    """
    text中の電話番号を<address>に置き換える

    >>> replace_address(
    """
    text = replace_pattern(address_pattern, text, replaced or '<address>')
    return text

## ID

account_pattern = RE(
    r'(@[A-Za-z0-9_]+)',
    r'(\b[Ii][Dd]:[A-Za-z0-9\-+_\.\/]+)',
    r'(\b[a-z]+[0-9_][0-9a-z_]*\b)',
    r'([0-9]+[a-z][0-9a-z_]+)',
    r'(\<NAME\>)',
    r'(\d{4}[ \-\/]\d{4}[ \-\/]\d{4}[ \-\/]\d{4})', # クレジットカード？
)

product_id_pattern = RE(
    r'(\b[Nn][Oo][:\.][A-Za-z0-9\-+_\.\/]{3,})',
    r'(\b[A-Z]+[_\-0-9][A-Z0-9_\-]+)',
    r'(\b[0-9]{2,}[A-Z\-][A-Z0-9_\-]+)',
    r'(\b[A-Z0-9]+\-[0-9]{5,}[A-Z0-9\-]*)',
    r'([0-9]+[A-Z_\/\-]+[A-Z0-9]+)',
    r'(\b[A-Z]{4,}[_\/\.0-9][A-Z0-9_\/\.=]*)',
    r'([0-9]{6,})',
)

base64_pattern = RE(
    r'(\b[0-9\+/]+[a-z]+[0-9\+/A-Z]+[a-z]+[0-9\+/A-Z]+[0-9a-zA-Z\+/]*={0,2}\b)',
    r'(\b[0-9\+/]+[A-Z]+[0-9\+/a-z]+[A-Z]+[0-9\+/a-z]+[0-9a-zA-Z\+/]*={0,2}\b)',
    r'(\b[0-9a-zA-Z+/]{4,}={1,2}\b)',
)

uuid_pattern = RE(r'\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b')

hash_pattern = RE(
    r'(\b[a-f]+[0-9]+[a-f]+[0-9]+[a-f0-9]{3,}\b)',
    r'(\b[0-9]+[a-f]+[0-9]+[a-f]+[a-f0-9]{3,}\b)',
    r'(\b[A-F]+[0-9]+[A-F]+[0-9]+[A-F0-9]{3,}\b)',
    r'(\b[0-9]+[A-F]+[0-9]+[A-F]+[A-F0-9]{3,}\b)',
)

def replace_id(text, replaced='<name>'):
    """
    text中の名前らしい識別子を<name>に置き換える

    >>> replace_id("藍猫 (@0309LOVE) Twitter")
    '藍猫 (<name>) Twitter'

    >>> replace_id("32 M子 id:OvB44Wm. 現代書館@gendaishokanさん")
    '32 M子 <name> 現代書館<name>さん'

    >>> replace_id("hairs7777 <date>« ランキングの好み")
    '<name> <date>« ランキングの好み'

    >>> replace_id("SLRカメラBS真鍮1201S563")
    'SLRカメラBS真鍮<id>'
        
    >>> replace_id("ZMCQV102741286153207☆おすすめ商品")
    '<id>☆おすすめ商品'

    >>> replace_id("大柴邦彦|DSTD-09705 アニメ")
    '大柴邦彦|<id> アニメ'

    >>> replace_id("S3 No.447984906")
    'S3 <id>'

    >>> replace_id("FX-TL3-MA-0006 FOX フォックス")
    '<id> FOX フォックス'

    >>> replace_id("入数100 03505325-001")
    '入数100 <id>'

    >>> replace_id("2311H-40BOX EA436BB-12 03073939-001")
    '<id> <id> <id>'

    >>> replace_id("LC500 フロント左右セット URZ100 T980DF")
    '<id> フロント左右セット <id> <id>'

    >>> replace_id("アイボリー 500043532")
    'アイボリー <id>'

    >>> replace_id("着丈79.580.581.582838485")
    '着丈79.580.581.<id>'

    >>> replace_id("1641 2.0 rakutenblog facebook_feed fc2_blog")
    '1641 2.0 rakutenblog <name> <name>'

    >>> replace_id("ed8ufce1382ez0ag7k 71lje_pxldbfa3f6529gjq9xwyv1mbw 801stws0r7dfqud905aedaln-a0ik29")
    '<name> <name> <name>-<name>'

    >>> replace_id("550e8400-e29b-41d4-a716-446655440000")
    '<uuid>'

    >>> replace_id("d41d8cd98f00b204e9800998ecf8427e")
    '<hash>'
    
    >>> replace_id("YWJjZGVmZw==")  #FIXME
    'YWJjZGVmZw=='
    
    >>> replace_id("#f1f1f1")  #FIXME
    '#<name>'
    
    >>> replace_id("Induction Certificate")
    'Induction Certificate'

    """
    text = replace_pattern(uuid_pattern, text, '<uuid>')
    text = replace_pattern(hash_pattern, text, '<hash>')
#    text = replace_pattern(account_pattern, text, replaced)
    text = replace_pattern(base64_pattern, text, '<base64>')
    text = replace_pattern(product_id_pattern, text, '<id>')
    return text

def replace_uuid(text, replaced=None):
    return replace_pattern(uuid_pattern, text, replaced or '<uuid>')

def replace_hash(text, replaced=None):
    return replace_pattern(hash_pattern, text, replaced or '<hash>')

def replace_base64(text, replaced=None):
    return replace_pattern(base64_pattern, text, replaced or '<base64>')

def replace_product_id(text, replaced=None):
    return replace_pattern(product_id_pattern, text, replaced or '<id>')

## アーティクル用

bar_pattern = RE(
    r'([^\s])\1{4,}',
)

def replace_bar(text, replaced=None):
    """
    text中の-----などの４連続以上の文字を短くする

    >>> replace_bar("-----------")
    '---'

    >>> replace_bar("――――――――――――――――――――――――――――【緊急速報】")
    '―――【緊急速報】'

    >>> replace_bar("-----*-----*-----*-----")
    '---*---*---*---'

    >>> replace_bar("    a=1")   # インデント
    '    a=1'

    >>> replace_bar("乗っ取りだ～wwwwww おおお!")
    '乗っ取りだ～www おおお!'

    >>> replace_bar("FF13の戦闘やべえｗｗｗｗｗｗｗｗｗｗｗｗｗｗｗｗｗｗおおお!")
    'FF13の戦闘やべえｗｗｗおおお!'
    """

    text = replace_pattern(bar_pattern, text, r'\1\1\1')
    return text

# enclose

double_enclose_pattern = RE(
    r'(\<\<[\s\S]+?\>\>)',
    r'(\[\[[\s\S]+?\]\])',
    r'(\{\{[\s\S]+?\}\})',
    r'(\(\([\s\S]+?\)\))',
    r'(【【[\s\S]+?】】)',
)

enclose_pattern = RE(
    r'(\[[^\]]+?\])',
    r'(\([^\)]+?\))', 
    r'(\{[^\}]+?\})', 
    r'(【[^】]+?】)',
    r'(（[^）]+?）)',
    r'(〔[^〕]+?〕)',
    r'(《[^》]+?》)',
)

close_pattern = RE(
    r'(\&[\#\w\d]+\;\s*)',
)


def replace_enclose(text, replaced=None):
    """
    text中から()などを取り除く

    >>> replace_enclose("仕様(デザイン、サイズ、カラーなど)に多少のバラツキが")
    '仕様に多少のバラツキが'

    >>> replace_enclose("サイズ外形:(約)幅 39 × 奥行 41")
    'サイズ外形:幅 39 × 奥行 41'

    >>> replace_enclose("リアリティ・ショー乱造の背景【米国テレビ事情】 - OMASUKI FIGHT")
    'リアリティ・ショー乱造の背景 - OMASUKI FIGHT'

    >>> replace_enclose("《街の口コミあり》白市駅")
    '白市駅'
   
    >>> replace_enclose("賃貸住宅[賃貸マンション・賃貸一軒家]で部屋探")
    '賃貸住宅で部屋探'

    >>> replace_enclose("2020年2月 (44) 2020年1月 (54) ")
    '2020年2月  2020年1月  '

    >>> replace_enclose("ｶﾞｸｶﾞｸ {{ (>_<) }} ﾌﾞﾙﾌﾞﾙ状態")
    'ｶﾞｸｶﾞｸ  ﾌﾞﾙﾌﾞﾙ状態'
    
    >>> replace_menu("邪神〈イリス〉覚醒")    # 除去しない
    '邪神〈イリス〉覚醒'

    """
    text = replace_pattern(double_enclose_pattern, text, '')
    text = replace_pattern(enclose_pattern, text, '')
    text = replace_pattern(close_pattern, text, '')
    return text


## アーティクル用

menu_pattern = RE(
    r'\s+[\|｜>＞/／«»].{,256}\n',
    r'[\|｜＞／«»].{,256}\n',
    r'\b12345678910\b',
)

def replace_menu(text):
    """
    textの中のメニューらしいところを取り出す

    >>> replace_menu("生八つ橋のタグまとめ | エキサイトブログ\\n生八つ橋のタグまとめ\\n")
    '生八つ橋のタグまとめ<menu>\\n生八つ橋のタグまとめ\\n'

    >>> replace_menu("ガメラ３／邪神〈イリス〉覚醒\\n邪神〈イリス〉覚醒")
    'ガメラ３<menu>\\n邪神〈イリス〉覚醒'

    """
    text = replace_pattern(menu_pattern, text, '<menu>\n')
    return text

article_pattern = RE(
    r'(?P<prefix>\n#?)\d{2,}[\:\.]?\b',
    r'(?P<prefix>\>\s*)\d{2,}[\:\.]?\b',
    # No.447984906
)

def replace_article(text, replaced=None):
    """
    textの中の記事番号を短くする

    >>> replace_article("\\n99: ナナシさん")
    '\\n<article>: ナナシさん'

    >>> replace_article("\\n600.投稿日:<date> <time>")
    '\\n<article>投稿日:<date> <time>'

    >>> replace_article(">>> 7084 およ")
    '>>> <article> およ'

    """
    replaced = replaced or '<article>'
    text = replace_pattern(article_pattern, text, f'\\g<prefix>{replaced}')
    return text


number_pattern = RE(
    r'\d{5,}', 
    r'\d*\.\d{4,}'
    r'\(num\)',
)

def replace_number(text, replaced=None):
    text = replace_pattern(number_pattern, text, replaced or "<number>")
    return text



cleanup_pattern = RE(
    r'\<\*\>.{,8}\<\*\>',
)

extra_newlines_pattern = RE(
    r'\n{3,}',
)

ignore_line_pattern = RE(
    r'\<menu\>|copyright|\(c\)|\s[\|\/]\s', 
    flags=re.IGNORECASE
)

def cleanup(text):
    lines = ['']
    for line in text.split('\n'):
        if len(line) < 40 and line.count('<') > 1:
            continue
        if len(ignore_line_pattern.findall(line)) > 0:
            if len(line) > 128 and '。' in line:
                if '<menu>' not in line:
                    lines.append(line)
                    continue
                # 一行が長い場合は、たぶん改行ミスなので、最後の(menu)だけ除去
                lines.append('。'.join(line.split('。')[:-1])+'。')
            lines.append('')
            continue
        lines.append(line)
    text = '\n'.join(lines)
    text = replace_pattern(extra_newlines_pattern, text, '\n\n')
    return text.replace('<url>', '')


# def CCFilter(text):
#     text = replace_url(text)
#     text = replace_email(text)
#     text = replace_datetime(text)
#     text = replace_phone(text)
#     text = replace_address(text)
#     text = replace_enclose(text)
#     text = replace_id(text)
#     text = replace_float(text)
#     text = replace_article(text)
#     text = replace_menu(text)
#     text = replace_bar(text)
#     return cleanup(text)

def find_replace_func(pattern:str):
    func = globals().get(f'replace_{pattern}')
    if func is None:
        patterns = [s.replace('replace_', '') for s in globals() if s.startswith('replace_')]
        raise ValueError(f'replace_{pattern} is not found. Select pattern from {patterns}')
    return func

class ReplacementFilter:
    """
    置き換えフィルター
    a = 'url:<URL>|date:<date>'
    """

    def __init__(self, patterns: List[str]):
        """
        置き換えフィルターを作る
        :param patterns: 置き換える文字列パターンのリスト
        """
        if isinstance(patterns,str):
            patterns = patterns.split('|')
        self.patterns = patterns
        self._replace_funcs = [find_replace_func(pattern) for pattern in patterns]

    def __call__(self, text):
        for replace_fn in self._replace_funcs:
            text = replace_fn(text)
            if text is None:
                break
        return cleanup(text)

if __name__ == '__main__':
    import doctest # doctestのための記載
    print(doctest.testmod())
