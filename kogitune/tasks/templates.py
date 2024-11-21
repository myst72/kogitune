
def has_schema(data: dict, keys:str):
    for key in keys.split('|'):
        if key not in data:
            return False
    return True

def contains_japanese(text: str) -> bool:
    for char in text:
        if '\u3040' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF' or '\uFF66' <= char <= '\uFF9D':
            return True
    return False

def guess_template(sample: dict):
    if has_schema(sample, 'instruction|input|output'):
        # Alpaca形式
        return {
            "prompt": "{instruction}\n{input}",
            "reference": "{output}",
        }
    if has_schema(sample, 'question|answer|answer_number|equation_solution'):
        # MSGM形式
        if contains_japanese(sample['question']):
            return {
                "prompt": "{question}",
                "reference": "{answer_number}",
                "shots": [
                    {'role': 'user', 'content': '問題：ロジャーは5個のテニスボールがあります。テニスボールの缶を2つ追加で買います。それぞれの缶には3つのテニスボールが入っています。彼は今いくつのテニスボールがありますか？'},
                    {'role': 'assistant', 'content': '11'},
                    {'role': 'user', 'content': '問題：サーバールームには9台のコンピューターがありました。月曜日から木曜日まで毎日5台のコンピューターをインストールしました。今サーバールームには難題のコンピューターがありますか？'}, 
                    {'role': 'assistant', 'content': '29'},
                    {'role': 'user', 'content': '問題：リアは32個のチョコレートを持っていました、彼女の妹は42個持っていました。彼女達が35個食べたとしたら、全部で何個残っていますか？'}, 
                    {'role': 'assistant', 'content': '39'},
                    {'role': 'user', 'content': '問題：ショーンは5個のおもちゃを持っています。クリスマスに、彼は父と母からそれぞれ2つずつおもちゃをもらいました。今彼はいくつのおもちゃがありますか？'}, 
                    {'role': 'assistant', 'content': '9'},
                    {'role': 'user', 'content': '問題：マイケルは58個のゴルフボールを持っています。火曜日、彼は\n23個のゴルフボールを失くしました。水曜日、さらに2個失くしました。水曜日の終わりには、彼は何このゴルフボールを持っていましたか？'},
                    {'role': 'assistant', 'content': '33'},
                    {'role': 'user', 'content': '問題：オリビアには$23あります。彼女はそれぞれ$3のベーグルを5つ買いました。彼女にはいくらのお金が残っていますか？'}, 
                    {'role': 'assistant', 'content': '8'},
                    {'role': 'user', 'content': '問題：ジェイソンは20個の飴を持っています。彼はデニーに飴をいくつかあげました。今、ジェイソンには12個の飴があります。ジェイソンはデニーにいくつ飴をあげましたか？'},
                    {'role': 'assistant', 'content': '8'},
                    {'role': 'user', 'content': '問題：駐車場に3台の車があり、2台の車が到着するとしたら、駐車場には何台の車がありますか？'}, 
                    {'role': 'assistant', 'content': '5'},
                ],
                "extract_pattern": r"[0-9]+"
            }
        else:
            return {
                "prompt": "{question}",
                "reference": "{answer_number}",
                "shots": [
                    {'role': 'user', 'content': 'Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?'}, 
                    {'role': 'assistant', 'content': '11'},
                    {'role': 'user', 'content': 'Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?'}, 
                    {'role': 'assistant', 'content': '29'},
                    {'role': 'user', 'content': 'Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?'}, 
                    {'role': 'assistant', 'content': '39'},
                    {'role': 'user', 'content': 'Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?'}, 
                    {'role': 'assistant', 'content': '9'},
                    {'role': 'user', 'content': 'Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?'}, 
                    {'role': 'assistant', 'content': '33'},
                    {'role': 'user', 'content': 'Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?'}, 
                    {'role': 'assistant', 'content': '8'},
                    {'role': 'user', 'content': 'Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?'}, 
                    {'role': 'assistant', 'content': '8'},
                    {'role': 'user', 'content': 'Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?'}, 
                    {'role': 'assistant', 'content': '5'},
                ],
                "extract_pattern": r"[0-9]+",
            }
    if has_schema(sample, 'prompt|test|entry_point|canonical_solution'):
        # HumanEval
        return {
            "prompt": "{prompt}",
            "reference": "{canonical_solution}\n",
            "test": "\n{test}\n\ncheck({entry_point})\n",
        }
    if has_schema(sample, 'question|choice0|choice1|choice2|choice3|choice4|label'):
        # JCommonSenseQA
        return {
            "shot": [
                {"role": "user", "content": "日本一高い山は？"},
                {"role": "assistant", "content": "Answer [0]"},
            ],
            "prompt": "{question}\n選択肢(Choice): [0] {choice0} [1] {choice1} [2] {choice2} [3] {choice3} [4] {choice4}\n",
            "reference": "{label}",
            "choice": ["0", "1", "2", "3", "4"],
            "prompt_0": "{question}\n{choice0}",
            "prompt_1": "{question}\n{choice1}",
            "prompt_2": "{question}\n{choice2}",
            "prompt_3": "{question}\n{choice3}",
            "prompt_4": "{question}\n{choice4}",
        }
    if has_schema(sample, 'question|A|B|C|D|answer'):
        # JMMLU
        if contains_japanese(sample['question']):
            return {
                "shots": [
                    {'role': 'user', 'content': '質問: 日本で一番高い山は？\n選択肢: [A] 富士山 [B] 高尾山 [C] 愛宕山 [D] エベレスト\n'}, 
                    {'role': 'assistant', 'content': 'A'},
                    {'role': 'user', 'content': '質問: 東京にある大学は？\n選択肢: [A] 京大 [B] 東大 [C] 北大 [D] ハーバード\n'}, 
                    {'role': 'assistant', 'content': 'B'},
                    {'role': 'user', 'content': '質問: 望月の歌の作者は？\n選択肢: [A] 紫式部 [B] 高野辰之 [C] 藤原道長 [D] 瀧廉太郎\n'}, 
                    {'role': 'assistant', 'content': 'C'},
                    {'role': 'user', 'content': '質問: 単位を落とすことは？\n選択肢: [A] 落胆 [B] 楽たん [C] 楽単 [D] 落単\n正しいものを[A]〜[D]から選べ'}, 
                    {'role': 'assistant', 'content': 'D'},
                ],
                "extract_pattern": r"\b[A-D]\b",
                "prompt": "質問: {question}\n選択肢: [A] {A} [B] {B} [C] {C} [D] {D}\n正しいものを[A]〜[D]から選びなさい。\n",
                "reference": "{answer}",
                "choice": ["A", "B", "C", "D"],
                "choice_A": "{question}\n{A}",
                "choice_B": "{question}\n{B}",
                "choice_C": "{question}\n{C}",
                "choice_D": "{question}\n{D}",
            }
        else:
            return {
                "shots": [
                    {'role': 'user', 'content': 'Question: Which is the largest number?\Options: [A] 9999 [B] 0000 [C] 1234 [D] 9876\n'}, 
                    {'role': 'assistant', 'content': 'A'},
                    {'role': 'user', 'content': 'Question: Which is the smallest number?\Options: [A] 9999 [B] 0000 [C] 1234 [D] 9876\n'}, 
                    {'role': 'assistant', 'content': 'B'},
                    {'role': 'user', 'content': 'Question: Which is the prime number?\Options: [A] 24 [B] 8 [C] 17 [D] 15\n'}, 
                    {'role': 'assistant', 'content': 'C'},
                    {'role': 'user', 'content': 'Question: Which is the largest planet in our solar system?\Options: [A] Earth [B] Mars [C] Venus [D] Jupiter\n'}, 
                    {'role': 'assistant', 'content': 'D'},
                ],
                "extract_pattern": r"\b[A-D]\b",
                "prompt": "Question: {question}\Options: [A] {A} [B] {B} [C] {C} [D] {D}\n",
                "reference": "{answer}",
                "choice": ["A", "B", "C", "D"],
                "choice_A": "{question}\n{A}",
                "choice_B": "{question}\n{B}",
                "choice_C": "{question}\n{C}",
                "choice_D": "{question}\n{D}",
            }
    if has_schema(sample, 'text'):
        # Kogitune 事前学習形式
        return {
            "prompt": "{text}",
            "reference": "",
        }
    if has_schema(sample, 'prompt'):
        # Kogitune 標準形式
        return {
            "prompt": "{prompt}",
        }
    return None
