import os
import ast
import traceback
import signal

from ..loads.commons import *
from .metrics import Metric

##
# HumanEval

class PassAtK(Metric):
    """
    コード評価用Evaluatorクラス
    HuggingFaceのevaluate-metric/code_evalを使用してスコアを算出する。

    m = kogitune.load('metric', "pass@1")
    print(m)
    candidates = [
        ["def f(n): return 1", "def"],
        ["def f(n): return 0", "def"],
    ]
    testcases = [
        "assert f(0) == 1",
        "assert f(0) == 1",
    ]
    m.calc(candidates, testcases)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = kwargs.get("k", 1)
        self.name =f"pass@{self.k}"
        evaluate = adhoc.safe_import("evaluate")
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        self.tool = evaluate.load("code_eval")  # code_eval
        if self.k != 1:
            adhoc.verbose_print('【注意】pass@1以外はまだテストされていません。')


    def calc(self, candidates:List[str], testcases: List[str], suffix=''):
        if len(candidates) > 0 and isinstance(candidates[0], str):
            # 文字列のリストの場合は、リストの文字列にする
            """
            candidates = [
                ["def f(n): return 1"],
            ]
            testcases = [
                "assert f(0) == 1"
            ]
            """
            candidates = [[e] for e in candidates]

        pass_at_k, results = self.tool.compute(
            references=testcases, predictions=candidates, k=[self.k]
        )
        adhoc.verbose_print('Quick results//速報値', pass_at_k)
        #print('@@results', results)
        scores = []
        result_passed = []
        result_messages = []
        for i in range(len(candidates)):
            n = len(results[i])
            passed = []
            messages = []
            for d, result in results[i]:
                passed.append(int(result['passed']))
                messages.append(result['result'])    
            scores.append((sum(passed)/n)*self.scale)
            result_passed.append(singlefy_if_single(passed))
            result_messages.append(singlefy_if_single(messages))
        if self.k > 1:
            # k を増やしたい人が修正すべき（by 🐼)
            return {
                f"{self.nametag}{suffix}": ('mean', [pass_at_k[self.nametag] * self.scale]),
                f"{self.nametag}_passed{suffix}": result_passed, # エラー回避
                f"{self.nametag}_result{suffix}": result_messages,
            }
        return {
            f"{self.nametag}{suffix}": ('mean', scores),
            f"{self.nametag}_passed{suffix}": ('sum', result_passed),
            f"{self.nametag}_result{suffix}": result_messages,
        }

    def extract_pairs(self, sample: dict):
        extracted_code = [
            openai_extract_code(sample["input"], x) for x in listfy(sample["output"])
        ]
        return extracted_code, sample["test"]

    def eval_s(self, extracted_code, test_case, sample=None):
        test_cases = [test_case]
        candidates = [extracted_code]
        pass_at_k, results = self.tool.compute(
            references=test_cases, predictions=candidates, k=[self.k]
        )
        if sample is not None:
            # 別のコード抽出も試す
            extracted_code2 = [[
                extract_code_from_prompt(sample["input"], x) for x in listfy(sample["output"])
            ]]
            pass_at_k2, results2 = self.tool.compute(
                references=test_cases, predictions=candidates, k=[self.k]
            )
            sample[f"{self.name}_1"] = pass_at_k[self.name] * self.scale
            sample[f"{self.name}_2"] = pass_at_k2[self.name] * self.scale
            # スコアの良い方を記録する
            if pass_at_k[self.name] > pass_at_k2[self.name]:
                sample["generated_code"] = singlefy(extracted_code)
                sample[f"{self.name}_results"] = simplify_results(results)
                return pass_at_k[self.name]
            else:
                sample["generated_code"] = singlefy(extracted_code2)
                sample[f"{self.name}_results"] = simplify_results(results2)
                return pass_at_k2[self.name]

        return pass_at_k[self.name]


PassAtK.register("pass@k")

# {"0": [[0, {"task_id": 0, "passed": false, "result": "failed: name 'df_product_full' is not defined", "completion_id": 0}]]},


def simplify_results(d):
    if isinstance(d, dict):
        if "passed" in d and "result" in d:
            return(dict(passed=d['passed'], result=d['result']))
        for _, v in d.items():
            v = simplify_results(v)
            if v is not None:
                return v
    if isinstance(d, (list, tuple)):
        for v in d:
            v = simplify_results(v)
            if v is not None:
                return v
    return None

def openai_extract_code(prompt, generated_text):
    """
    OpenAI HumanEval論文のコード抽出手法
    """
    stop_sequences = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
    min_stop_index = len(generated_text)
    for seq in stop_sequences:
        stop_index = generated_text.find(seq)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return prompt + "\n" + generated_text[:min_stop_index]



def extract_code_from_prompt(prompt, generated_text):
    stop_sequences=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
    min_stop_index = len(generated_text)
    for seq in stop_sequences:
        stop_index = generated_text.find(seq)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    code = prompt + "\n" + generated_text[:min_stop_index]
    return extract_python_code(code)


def extract_python_code(text):
    result = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].strip() == '':
            # 空行はスキップする
            i += 1
            continue
        code = '\n'.join(lines[i:])
        next = get_syntax_error_line(code)
        #print(i, next, code)
        if next == 1:
            # 先頭でエラーが発生したらスキップする
            i += 1
            continue
        if next is None:
            result.append(code)
            break
        code = clean_code('\n'.join(lines[i:i+next-1]))
        if code is not None:
            result.append(code)
        i += next
    return '\n'.join(result)

def clean_code(code):
    while True:
        error_lineno = get_syntax_error_line(code)
        if error_lineno is None:
            return code
        if '\n' not in code:
            break
        code, _, _ = code.rpartition('\n')
    return None

def get_syntax_error_line(code):
    try:
        ast.parse(code)
        return None  # エラーがない場合はNoneを返す
    except SyntaxError as e:
        return e.lineno  # エラーが発生した行番号を返す

##
# Experimental コードのエラー番号を読み取る


# タイムアウトの例外を定義
class TimeoutException(Exception):
    pass

# タイムアウト時に呼び出されるハンドラー
def timeout_handler(signum, frame):
    raise TimeoutException("タイムアウトしました！")



TEMPLATE_CODE_FIX = '''\
The following error has occurred. 
Please fix the code so that it can be executed without errors.

### Code
{code}

### Error
{error_message}
{stack_trace}
{error_message}

'''

def get_error_line_number():
    stack_trace = traceback.format_exc()
    # スタックトレースの最後の呼び出し部分から行番号を抽出
    tb_lines = stack_trace.splitlines()
    line_number = len(tb_lines)
    # print('@@@', stack_trace)
    for line in tb_lines:
        if 'File "<string>"' in line and ", line" in line:
            # 行番号を抽出
            try:
                _,_,linenum = line.partition(", line ")
                linenum,_,_ = linenum.partition(',')
                line_number = int(linenum)
            except:
                pass
    return line_number

def format_error_lines(code, line_number):
    code_lines = code.strip().split('\n')
    formatted_code = ""
    for i, line in enumerate(code_lines, 1):
        if i == line_number:
            formatted_code += f"----> {i} {line}\n"
        elif line_number - 2 <= i <= line_number + 1:
            formatted_code += f"      {i} {line}\n"
    return formatted_code

def get_code_fix_prompt(code_str, test_code):
    if isinstance(code_str, list):
        return [get_code_fix_prompt(x, test_code) for x in code_str]
    code = (code_str+test_code).strip()
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    
    try:
        signal.alarm(10)
        # コードを実行
        exec(code)
        signal.alarm(0)
        return ''
    except Exception as e:
        # エラーが発生した場合、エラーメッセージとスタックトレースを回収
        error_message = f'{type(e).__name__}: {str(e)}'
        # _, _, tb = sys.exc_info()
        line_number = get_error_line_number()
        formatted_code = format_error_lines(code, line_number)
        prompt = TEMPLATE_CODE_FIX.format(
            error_message=error_message, 
            stack_trace=formatted_code, 
            code=code_str)
        return prompt
    finally:
        signal.signal(signal.SIGALRM, original_handler)

