from .patterns_ import re, register_pattern

pattern_config_commons = {
    "url": {
        "patterns": [r'https?://[\w/:%#\$&\?~\.\,=\+\-\\_]+'],
        "tests": [
            ("http://www.peugeot-approved.net/UWS/WebObjects/UWS.woa/wa/carDetail?globalKey=uwsa1_1723019f9af&currentBatch=2&searchType=1364aa4ee1d&searchFlag=true&carModel=36&globalKey=uwsa1_1723019f9af uwsa1_172febeffb0, 本体価格 3,780,000 円", '<url> uwsa1_172febeffb0, 本体価格 3,780,000 円'),
            ("「INVADER GIRL!」https://www.youtube.com/watch?v=dgm6-uCDVt0", '「INVADER GIRL!」<url>'),
            ("(http://t.co/x0vBigH1Ra)シグネチャー", '(<url>)シグネチャー'),
            ("kindleにあるなーw http://www.amazon.co.jp/s/ref=nb_sb_noss?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&url=search-alias%3Ddigital-text&field-keywords=%E3%82%A2%E3%82%B0%E3%83%8D%E3%82%B9%E4%BB%AE%E9%9D%A2",'kindleにあるなーw <url>'),
            ("http://live.nicovideo.jp/watch/lv265893495 #nicoch2585696", '<url> #nicoch2585696')
        ]
    },
    "date": {
        "patterns": [
            r'(?:19|20)\d{2}[-/\.]\d{1,2}[-/\.]\d{1,2}',
            r'(?:19|20)\d{2}\s?年\s?\d{1,2}\s?月\s?\d{1,2}\s?日(?![はにかまのも、])', # 否定先読み
            r'(?:令和|平成|昭和)\s?\d{1,2}\s?年\s?\d{1,2}\s?月\s?\d{1,2}\s?日(?![はにかまのも、])', # 否定先読み
            r'[HRS]\d{1,2}[-/\.]\d{1,2}[-/\.]\d{1,2}',  # Matches 'H27.9.18'
            r'\d{1,2}[-/\.]\d{1,2}[-/\.](?:19|20)\d{2}', 
            #アメリカ式  
            r'\bJan(?:uary)?\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bFeb(?:ruary)?\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bMar(?:ch)?\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bApr(?:il)?\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bMay\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bJun(?:e)?\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bJul(?:y)?\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bAug(?:ust)?\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bSep(?:tember)?\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bOct(?:ober)?\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bNov(?:ember)?\s+\d{1,2}\,?\s+\d{4}\b',
            r'\bDec(?:ember)?\s+\d{1,2}\,?\s+\d{4}\b',
            #英国式  
            r'\d{1,2}\s+Jan(?:uary)?\s+\d{4}\b',
            r'\b{1,2}\s+Feb(?:ruary)?\s+\d{4}\b',
            r'\b{1,2}\s+Mar(?:ch)?\s+\d{4}\b',
            r'\b{1,2}\s+Apr(?:il)?\s+\d{4}\b',
            r'\b{1,2}\s+May\s+\d{4}\b',
            r'\b{1,2}\s+Jun(?:e)?\s+\d{4}\b',
            r'\b{1,2}\s+Jul(?:y)?\s+\d{4}\b',
            r'\b{1,2}\s+Aug(?:ust)?\s+\d{4}\b',
            r'\b{1,2}\s+Sep(?:tember)?\s+\d{4}\b',
            r'\b{1,2}\s+Oct(?:ober)?\s+\d{4}\b',
            r'\b{1,2}\s+Nov(?:ember)?\s+\d{4}\b',
            r'\b{1,2}\s+Dec(?:ember)?\s+\d{4}\b',
            r'\<date\>',
            r'\(date\)',
        ],
        "flags": re.IGNORECASE,
        "tests": [
            ("March 20, 2016", '<date>'),
            ("March 20, 2016 13:24", '<date> 13:24'),
            ("Posted on: Tuesday, 01 May, 2018 18:14", 'Posted on: Tuesday, <date> 18:14'),
            ("返信・引用 FEB 12, 2004", '返信・引用 <date>'),
            ("|18-Feb-2013|", '|<date>|'),
            ("2016/08/16 17:37 - みすずのつぶやき", '<date> 17:37 - みすずのつぶやき'),
            ("2007-11-14 Wed", '<date> Wed'),
            ("| 2016-03-08 12:18", '| <date> 12:18'),
            ("HIT 2001-01-09 レス", 'HIT <date> レス'),
            ("HIT 2001-1-9 レス", 'HIT <date> レス'),
            ("(2016.3.8. 弁理士 鈴木学)", '(<date>. 弁理士 鈴木学)'),
            ("Posted: 2005.06.22", 'Posted: <date>'),
            ("HIT 2019/03/23(土) レス", 'HIT <date>(土) レス'),
            ("35: 名刺は切らしておりまして 2017/01/26(木) 23:22:16.16", '35: 名刺は切らしておりまして <date>(木) 23:22:16.16'),
            ("2013年01月15日 本澤二郎の「日本の風景」(1252) ", '<date> 本澤二郎の「日本の風景」(1252) '),
            ("2009年 08月22日 10:08 (土)", '<date> 10:08 (土)'),
            ("2018年1月30日 at 22:37", '<date> at 22:37'),
            ("1972年12月19日生まれ", '<date>生まれ'),
            ("H27.9.18 9月議会最終日。", '<date> 9月議会最終日。'),
            ("令和元年10月6日(日)", '令和元年10月6日(日)'),
            ("平成22年12月22日の時点で", '平成22年12月22日の時点で'),
            ("anond:20190414004605", 'anond:20190414004605'),
            ("その後、1988年秋に日経産業新聞から", 'その後、1988年秋に日経産業新聞から'),
            ("月を選択 2019年8月 (246) ", '月を選択 2019年8月 (246) '),
            ("東京工業大学は令和1年12月24日からORCIDメンバーとなっています", '東京工業大学は令和1年12月24日からORCIDメンバーとなっています'),
            ("http://tanakaryusaku.jp/2016/10/00014719", 'http://tanakaryusaku.jp/2016/10/00014719'),
        ]
    },
    "time": {
        "patterns": [
            r'\d{1,2}:\d{1,2}(:\d{1,2})?(\.\d{2,3})?(\s*(AM|PM))?',
            r'\(\d{1,2}:\d{1,2}(:\d{1,2})?\)',
            r'\d{1,2}[時]\s?\d{1,2}\s?[分]\s?\d{1,2}\s?[秒]',
            r'\d{1,2}[時]\s?\d{1,2}\s?[分](?![はにかまのも、])',
            r'\(time\)(?:\.\d{2,3})',
            r'\<time\>(?:\.\d{2,3})',
        ],
        "tests": [
            ("35: 名刺は切らしておりまして 2017/01/26(木) 23:22:16.16",'35: 名刺は切らしておりまして 2017/01/26(木) <time>'),
            ("2009年 08月11日 11:04 (火)", '2009年 08月11日 <time> (火)'),
            ("2017年1月9日 8:12 PM", '2017年1月9日 <time>'),

        ]
    },
    "email": {
        "patterns": [
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}',
        ],
        "flags": re.IGNORECASE,
        "tests": [
            ("Chian-Wei Teo +81-3-3201-3623　cwteo@bloomberg.net 香港", 'Chian-Wei Teo +81-3-3201-3623　<email> 香港'),
        ]
    },
    "phone": {
        "patterns": [
            r'\(0\d{1,4}\)\s*\d{2,4}-\d{3,4}',
            r'0\d{1,4}-\d{2,4}-\d{3,4}',
            r'\+81-\d{1,4}-\d{2,4}-\d{3,4}',
            r'\+\d{10,}' #+819012345678
            r'\(phone\)', r'\<phone\>',
        ],
        "tests": [
            ("Contact me at 0532-55-2222", 'Contact me at <phone>'),
            ("TEL0120-350-108", 'TEL<phone>'),
            ("大阪市中央区日本橋1-22-20-604ATEL:06-7860-2088", '大阪市中央区日本橋1-22-20-604ATEL:<phone>'),
            ("Chian-Wei Teo +81-3-3201-3623　cwteo@bloomberg.net 香港", 'Chian-Wei Teo <phone>　cwteo@bloomberg.net 香港'),
        ]
    },
    "address_jp": {
        "patterns": [
            r'〒\d{3}-\d{4}[\s|\n][^\d]{,40}\d{1,4}(?:[−\-ー]\d{1,4}(?:[−\-ー]\d{1,4})?)?',
        ],
        "tests": [
           ("〒311-2421 茨城県潮来市辻232−2  NVVKx1T0QuvSfCR", '<address>  NVVKx1T0QuvSfCR'),
        ]
    }
}

register_pattern(pattern_config_commons)
