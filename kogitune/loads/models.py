from typing import List, Union
#import torch

from .commons import *
from .files import basename

MODEL_MAP = {}

class ModelLoader(adhoc.AdhocLoader):

    def load_modules(self, path, kwargs):
        from .models_api import OpenAIModel
        from .models_vllm import vLLMModel

    def add_kwargs(self, path, kwargs):
        if '_tag' not in kwargs:
            tag = basename(path, split_ext=False)
            kwargs['_tag'] = kwargs.get('modeltag', tag)
        return path, kwargs
        
    def load_default(self, path, kwargs):
        if path.startswith("dummy"):
            return Model(**kwargs)
        else:
            return HFModel(**kwargs)

ModelLoader(MODEL_MAP).register("model")

class Model(adhoc.AdhocObject):
    def __init__(self, **kwargs):
        """
        Base class for abstracting a model.
        """
        super().__init__(**kwargs)
        self.progress_bar = None

    @property
    def modeltag(self):
        if self.tag != '':
            return self.tag
        return basename(self.path, split_ext=False)

    def lazy_load(self):
        """
        遅延ロードに対応したモデル用のダミーメソッド
        """
        pass

    def compute_loss(self, input_texts: Union[List[str],str], progress_bar=None) -> float:
        raise NotImplementedError()
 
    def supported_gen_args(self) -> List[str]:
        return []

    def filter_gen_args(self, **kwargs):
        return adhoc.safe_kwargs(kwargs, self.supported_gen_args(), unsafe='GEN')

    @classmethod
    def is_valid_prompt(cls, input_text: Union[List[dict], str]):
        if isinstance(input_text, str):
            return True
        if isinstance(input_text, list) and len(input_text) > 0:
            for msg in input_text:
                if isinstance(msg, dict):
                    if 'role' in msg and 'content' in msg:
                        continue
                return False
            return True
        return False
    
    def get_default_messages(self, input_text:str):
        if isinstance(input_text, str):
            return [
                {"role": "user", "content": input_text}
            ]
        return input_text

    def transform_messages(self, input_texts: Union[List[str],str], heading=None) -> Union[List[str], str]:
        """
        少数ショットに対応したメッセージの作成プロンプト
        """
        heading = heading or []
        output_texts = []
        for input_text in self.listfy_prompt(input_texts):
            messages = heading[:]
            messages.append(self.get_default_messages(input_text))
            output_texts.append(messages)
        return singlefy_if_single(output_texts)

    def listfy_prompt(self, input_text: Union[List[dict],str]):
        if isinstance(input_text, list) and len(input_text) > 0 and isinstance(input_text[0],dict):
            return [input_text]
        if isinstance(input_text, str):
            return [input_text]
        return input_text

    def generate(self, prompts: List[Union[List[dict],str]], progress_bar=None, /, **kwargs) -> Union[List[str], str]:
        """
        テキストを生成する

        主なパラメータ
        prompts: テキストかメッセージ(dict形式)のリスト
        
        帰り値
        生成されたテキストのリスト
        """

        gen_args = self.filter_gen_args(kwargs)
        output_texts = []
        for prompt in self.listfy_prompt(prompts):
            output_texts.append(self.generate_s(prompt, **gen_args))
            if progress_bar:
                progress_bar.update(1)
        return singlefy_if_single(output_texts)

    def generate_s(self, prompt: Union[List[dict], str], /, **kwargs) -> Union[List[str], str]:
        """
        generate_s の単一入力バージョン（generateから呼び出される)

        主なパラメータ
        prompt: テキストかメッセージ(dict形式)
        
        帰り値
        生成されたテキスト、もしくは、そのリスト
        """
        
        gen_args = self.filter_gen_args(kwargs)

        return NotImplementedError()

    @classmethod
    def regiser(cls, scheme):
        global MODEL_MAP
        MODEL_MAP[scheme] = cls

## HF

class TokenizerModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lazy_kwargs = {**kwargs}
        tokenizer_path = adhoc.get(kwargs, f'tokenizer_path|_subpath|model_path|={self.path}')
        self.tokenizer = adhoc.load('_tokenizer', tokenizer_path, **kwargs)
        # なぜか必要らしい（↓）
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if not kwargs.get('_lazy', False):
            self.lazy_load()

    @property
    def modeltag(self):
        if self.tag != '':
            return self.tag
        return basename(self.path, split_ext=False)

    def lazy_load(self):
        if self.lazy_kwargs is None:
            return
        pass

    def format_text_prompt(self, input_text:Union[List[str],str]) -> str:
        if isinstance(input_text, str):
            return input_text
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(input_text, tokenize=False)
        ss=[]
        for msg in input_text:
            ss.append(msg['content'])
        return '\n'.join(ss)


class HFModel(TokenizerModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lazy_load(self):
        if self.lazy_kwargs is None:
            return
        transformers = adhoc.safe_import('transformers')
        kwargs = self.lazy_kwargs
        self.lazy_kwargs = None

        model_path = adhoc.get(kwargs, f'_subpath|model_path|={self.path}')
        self._model, self.pathargs = load_hfmodel(model_path, **kwargs)
        self.device = next(self._model.parameters()).device
        adhoc.verbose_print(f"Model has loaded on {self.device}.//モデルは{self.device}上にロードされました")
        hf_token = adhoc.get(kwargs, "use_auth_token|HF_TOKEN")
        self.generator = transformers.pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            use_auth_token=hf_token,
        )

    @property
    def model(self):
        self.lazy_load()
        return self._model

    def unwrap(self):
        self.lazy_load()
        return self._model

    def supported_gen_args(self) -> List[str]:
        return [
            "_n|num_return_sequences|n",
            "_do_sample|do_sample",  # (bool, optional, defaults to False) — Whether or not to use sampling ; use greedy decoding otherwise.
            "_max_tokens|max_new_tokens|max_tokens|=256",  # (int, optional) — The maximum numbers of tokens to generate
            "temperature",  # (float, optional, defaults to 1.0) — The value used to modulate the next token probabilities.
            "top_k",  # (int, optional, defaults to 50) — The number of highest probability vocabulary tokens to keep for top-k-filtering.
            "top_p",  # (float, optional, defaults to 1.0) — If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            "repetition_penalty",  # (float, optional, defaults to 1.0) — The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
#            "max_length",  # (int, optional, defaults to 20) — The maximum length the generated tokens can have.
        ]

    def filter_gen_args(self, **kwargs):
        gen_args = super().filter_gen_args(**kwargs)
        if "return_full_text" not in gen_args:
            gen_args["return_full_text"] = False
        if "max_length" in gen_args and "max_new_tokens" in gen_args:
            gen_args.pop("max_length")
        return gen_args

    def generate(self, input_texts: Union[List[str],str], progress_bar=None, /, **kwargs) -> Union[List[str], str]:
        self.lazy_load()
        gen_args = self.filter_gen_args(**kwargs)
        
        input_texts = [self.format_text_prompt(s) for s in self.listfy_prompt(input_texts)]
        adhoc.verbose_print('[Prompt]', dump=input_texts, once="formatted_prompt")

        outputs = self.generator(text_stream(input_texts, progress_bar), **gen_args)
        output_texts = []
        for results in outputs:
            generated_texts = [item["generated_text"] for item in results]
            output_texts.append(singlefy_if_single(generated_texts))
        return singlefy_if_single(output_texts)
        
    def compute_loss(self, input_texts: Union[List[str],str], progress_bar=None) -> Union[List[float], float]:
        torch = adhoc.safe_import('torch')
        self.lazy_load()
        values = []
        for input_text in listfy(input_texts):
            inputs = self._tokenizer(input_text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()
            # 不要なキーを除去
            inputs.pop("token_type_ids", None)
            with torch.no_grad():
                outputs = self._model(**inputs, labels=labels)
                loss = outputs.loss
            values.append(loss.item())
            if progress_bar:
                progress_bar.update(1)
        return singlefy_if_single(values)

HFModel.regiser("hf")

model32_kw_list = [
    "use_auth_token", 
    "trust_remote_code", 
    "device_map", 
    "attn_implementation"
]

def load_hfmodel32(model_path, /, **kwargs):
    from transformers import AutoModelForCausalLM

    model_args = adhoc.safe_kwargs(kwargs, model32_kw_list, unsafe='MODEL')
    adhoc.verbose_print('Loading the model//モデルロード', 
                        model_path, model_args, once=model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
    except BaseException as e:
        adhoc.report_ArgumentError(
            message='Failed to load the model//モデルのロード失敗', 
            called = adhoc.function_called("AutoModelForCausalLM.from_pretrained", 
                                            model_path, **model_args),
            throw=e)
    return model, model_args


model4bit_kw_list = [
    "use_auth_token", 
    "trust_remote_code", 
    "device_map", 
    "attn_implementation"
]

def load_hfmodel4bit(model_path, /, **kwargs):
    try:
        adhoc.safe_import('bitsandbytes')
        import torch
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM
        
        model_args = adhoc.safe_kwargs(kwargs, model4bit_kw_list, unsafe='MODEL')

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, **model_args
        )
        return model, model_args
    except BaseException as e:
        adhoc.print(
            f"Unable to load 4Bit Quantimization Model///4ビット量子化モデルがロードできません", str(e)
        )
        adhoc.print("Trying a normal model...///とりあえず、ノーマルモデルを試します")
    return load_hfmodel32(model_path, **kwargs)


def load_hfmodel(model_path, /, **kwargs):
    torch = adhoc.safe_import('torch')

    if "use_auth_token" not in kwargs:
        kwargs["use_auth_token"] = adhoc.get(os.environ, "HF_TOKEN")
    if "trust_remote_code" not in kwargs:
        kwargs["trust_remote_code"] = True

    # MacOS 上でエラーになる
    if torch.cuda.is_available() and "device_map" not in kwargs:
        kwargs["device_map"] = "auto"
    if kwargs.get("attn_implementation") == "flash_attention_2":
        kwargs["torch_dtype"] = torch.bfloat16
    if "torch_dtype" in kwargs:
        kwargs["torch_dtype"] = parse_dtype(kwargs["torch_dtype"])

    if adhoc.get(kwargs, "use_4bit|=False"):
        return load_hfmodel4bit(model_path, **kwargs)
    else:
        return load_hfmodel32(model_path, **kwargs)

@adhoc.parse_value
def parse_dtype(dtype, torch=None):
    if torch:
        torch = adhoc.safe_import('torch')

    dtype_mapping = {
        "float": torch.float,
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.double,
        "float16": torch.float16,
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "int": torch.int,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.long,
        "int16": torch.int16,
        "short": torch.short,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if isinstance(dtype, str):
        if dtype in dtype_mapping:
            return dtype_mapping.get(dtype)
        raise ValueError(f"unknown {dtype} dtype in PyTorch")
    return dtype

@adhoc.from_kwargs
def hfmodel_from_kwargs(**kwargs):
    model = kwargs.get('model', '')
    if not isinstance(model, str):
        return model

    adhoc_keys = 'model_path|model'
    use_default = kwargs.get('use_default', False)
    if use_default:
        if not isinstance(use_default, str):
            use_default = os.environ.get("MODEL_PATH", "kogi-jwu/chico-0.03b")
        keys = f'{keys}|!{use_default}'
    else:
        keys = f'{keys}|!!'
    model_path = adhoc.get(kwargs, adhoc_keys)
    return load_hfmodel(model_path, **kwargs)[0]


## 私は現在、データセットとバッチ処理でゼロショットテキスト分類器パイプラインを使用しています。
# 「GPUでパイプラインを順番に使用しているようです。効率を最大化するために、データセットを使用してください」という警告は、
# 私のループの反復ごとに表示されます。
# 私はデータセットを使用しており、バッチ処理しています。この警告がバグなのか、それとも本当の問題を診断するのに十分な説明的ではないのかはわかりません。
## https://github.com/huggingface/transformers/issues/22387


def text_stream(texts: List[str], progress_bar=None):
    if not progress_bar:
        progress_bar = adhoc.progress_bar() # dummy
    if len(texts) == 1:
        progress_bar.update(1)
        yield texts[0]
    else:
        for text in texts:
            progress_bar.update(1)
            yield text



