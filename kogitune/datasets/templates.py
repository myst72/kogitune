from ..loads.commons import *
from .templates_da import da

# パターンに一致するすべての部分を検索

SEC_IN = "### instruction\n"
SEC_OUT = "\n### output\n"

# まず、基本的な正規表現パターンを設計します。
# {}内の文字をマッチングするためには、{([^}]+)}のようなパターンを使用します。
# これは、{}で囲まれた、}を含まない1文字以上の任意の文字列にマッチします。
# 次に、抽出した文字列からフォーマット指定を除いてキー名のみを取り出すために、さらに処理を加えます。


## テスト
# test_string = "価格は{price:.2f}円で、数量は{quantity}個です。割引率は{discount:.2%}です。"
# extract_format_keys(test_string)


class TemplateProcessor(object):
    def __init__(self, template_config: dict):
        self.options = template_config or {}
        self.enforce_da = True
        self.random_choice = True
        # self.prompt = kwargs.get("prompt", "")
        # self.output = kwargs.get("output", kwargs.get("reference", ""))
        # self.template_keys = extract_format_keys(self.prompt + self.output)
        # self.SEC_IN = kwargs.get("section_in", SEC_IN)
        # self.SEC_OUT = kwargs.get("section_out", SEC_OUT)
        # da_policy = kwargs.get("da_policy", "dynamic")
        # if da_policy == "dynamic":
        #     self.enforce_da = True
        #     self.random_choice = True
        # elif da_policy == "notion":
        #     self.enforce_da = False
        # else:  # static
        #     self.enforce_da = True
        #     self.random_choice = False
        # self.options = kwargs

    def __repr__(self):
        return f'Template:{self.options}'


    def apply(self, input_type: str, sample: dict, enforce=True):
        if "input" in sample and sample["input"] == input_type:
            return sample
        if input_type == "choice":
            if "choice" not in self.options:
                adhoc.notice("テンプレートにchoiceがありません")
                raise ValueError()
            sample["input_type"] = input_type
            sample["input"] = [
                self.format(f"prompt_{choice}", sample)
                for choice in self.options["choice"]
            ]
            sample["reference"] = self.format("reference", sample)
            sample["choice"] = self.options["choice"]
            return "output"
        if input_type == "loss":
            sample["input_type"] = input_type
            prompt = self.format("prompt", sample)
            reference = self.format("reference", sample)
            sample["input"] = f"{prompt}\n{reference}"
            return "loss"
        sample["input_type"] = ""
        sample["input"] = self.format("prompt", sample)
        sample["reference"] = self.format("reference", sample)
        if self.has_option("test"):
            sample["test"] = self.format("test", sample)
        return "output"

    def has_option(self, key):
        return key in self.options

    def format(self, key: str, sample: dict):
        if key in self.options:
            text = self.options[key].format(**sample)
            if self.enforce_da:
                text = da(text, random_choice=self.random_choice)
            return text
        return ""

    # def create(self, key, sample: dict):
    #     text = self.options[key].format(**sample)
    #     if self.enforce_da:
    #         text = da(text, random_choice=self.random_choice)
    #     return text

    # def create_prompt(self, sample: dict):
    #     prompt = self.prompt.format(**sample)
    #     if self.enforce_da:
    #         prompt = da(prompt, random_choice=self.random_choice)
    #     return prompt

    # def create_output(self, sample: dict):
    #     output = self.output.format(**sample)
    #     if self.enforce_da:
    #         output = da(output, random_choice=self.random_choice)
    #     return output

    # def create_reference(self, sample: dict):
    #     return self.create_output(sample)

    # def load_sample(
    #     self, eval_type: str, datalist: List[dict], sample_list: List[dict]
    # ):
    #     assert len(datalist) == len(sample_list)
    #     for i in range(len(datalist)):
    #         source = datalist[i]
    #         sample = sample_list[i]
    #         if eval_type == "choice":
    #             result_key = self._load_choice(source, sample)
    #         elif eval_type == "loss":
    #             result_key = self._load_loss(source, sample)
    #         else:
    #             result_key = self._load_generation(source, sample)
    #     return result_key

    # def _load_generation(self, source, sample):
    #     if "input" not in sample:
    #         sample["input"] = self.create_prompt(source)
    #     if "reference" not in sample:
    #         sample["reference"] = self.create_output(source)
    #     if self.has_option("test"):
    #         sample["test"] = self.create("test", source)
    #     return "output"

    # def _load_loss(self, source, sample):
    #     input_text = self.create_prompt(source)
    #     reference = self.create_reference(source)
    #     sep = "" if input_text.endswith("\n") else "\n"
    #     sample["input"] = f"{input_text}{sep}{reference}"
    #     return "loss"

    # def _load_choice(self, source, sample):
    #     if "choice" not in self.options:
    #         adhoc.notice("テンプレートにchoiceがありません")
    #         raise ValueError()
    #     sample["choice"] = self.options["choice"]
    #     sample["input"] = [
    #         self.create(f"prompt_{choice}", source) for choice in self.options["choice"]
    #     ]
    #     sample["reference"] = self.create_output(source)
    #     return "output"

    # def create_instruction(self, sample: dict):
    #     prompt = self.create_prompt(sample)
    #     output = self.create_output(sample)
    #     return f"{self.SEC_IN}{prompt}{self.SEC_OUT}{output}"

    def formatting_for_trainer(self, example):
        output_texts = []
        sample = {}
        key = self.template_keys[0]
        for i in range(len(example[key])):
            for key in self.template_keys:
                sample[key] = example[key][i]
            output_texts.append(self.create_instruction(sample))
        return output_texts

    # def test_template(self, sample: dict, verbose=True):
    #     try:
    #         prompt = self.create_prompt(sample)
    #     except KeyError as e:
    #         adhoc.warn(key_error=e, template=self.prompt, sample=sample)
    #     try:
    #         reference = self.create_output(sample)
    #     except KeyError as e:
    #         adhoc.warn(key_error=e, template=self.output, sample=sample)
    #     if verbose:
    #         adhoc.print(
    #             f"プロンプトを確認してね\n{prompt}\n（期待される出力）\n{reference}"
    #         )

    # def filter(
    #     self,
    #     dataset,
    #     tokenizer,
    #     max_length=None,
    #     min_length=None,
    #     head=None,
    #     return_as_dict=False,
    # ):
    #     sample_list = []
    #     for sample in dataset:
    #         prompt_length = len(tokenizer.encode(self.create_prompt(sample)))
    #         total_length = (
    #             len(tokenizer.encode(self.create_output(sample))) + prompt_length
    #         )
    #         if max_length:
    #             if prompt_length > max_length or total_length > max_length:
    #                 continue
    #         if min_length and total_length < min_length:
    #             continue
    #         if head and len(sample_list) >= head:
    #             break
    #         sample_list.append(sample)
    #     if return_as_dict:
    #         return sample_list
    #     data_columns = {
    #         key: [dic[key] for dic in sample_list] for key in sample_list[0]
    #     }
    #     return datasets.Dataset.from_dict(data_columns)

@adhoc.from_kwargs
def template_from_kwargs(_sample:dict = None, **kwargs):
    template_config = adhoc.get(kwargs, 'template_config|template')
    if isinstance(template_config, str):
        template_config = adhoc.load_config(template_config)
    elif not isinstance(template_config, dict) and _sample is not None:
        template_config = guess_template(_sample)
    return TemplateProcessor(template_config)

### Template


def has_schema(data: dict, keys: str):
    for key in keys.split("|"):
        if key not in data:
            return False
    return True


def guess_template(sample: dict):
    if has_schema(sample, "instruction|input|output"):
        # Alpaca形式
        return {
            "prompt": "{instruction}\n{input}",
            "reference": "{output}",
        }
    if has_schema(sample, "prompt|test|entry_point|canonical_solution"):
        # HumanEval
        return {
            "prompt": "{prompt}",
            "reference": "{canonical_solution}\n",
            "test": "\n{test}\n\ncheck({entry_point})\n",
        }
    if has_schema(sample, "question|choice0|choice1|choice2|choice3|choice4|label"):
        # JCommonSenseQA
        return {
            "prompt": "{question}\n[選択肢|Choice]: [@(0)|0.|[0]] {choice0} [@(1)|1.|[1]] {choice1} [@(2)|2.|[2]] {choice2} [@(3)|3.|[3]] {choice3} [@(4)|4.|[4]] {choice4}\n",
            "reference": "{label}",
            "choice": ["0", "1", "2", "3", "4"],
            "prompt_0": "{question}\n{choice0}",
            "prompt_1": "{question}\n{choice1}",
            "prompt_2": "{question}\n{choice2}",
            "prompt_3": "{question}\n{choice3}",
            "prompt_4": "{question}\n{choice4}",
        }
    if has_schema(sample, "question|A|B|C|D|answer"):
        # JMMLU
        return {
            "prompt": "{question}\n[選択肢|Choice]: [@(A)|A.|[A]] {A} [@(B)|B.|[B]] {B} [@(C)|C.|[C]] {C} [@(D)|D.|[D]] {D} \n",
            "reference": "{answer}",
            "choice": ["A", "B", "C", "D"],
            "prompt_A": "{question}\n{A}",
            "prompt_B": "{question}\n{B}",
            "prompt_C": "{question}\n{C}",
            "prompt_D": "{question}\n{D}",
        }
    return None
