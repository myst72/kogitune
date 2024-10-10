from typing import List, Union
import math
import numpy as np
import torch

from .commons import *
from .files import basename

MODEL_MAP = {}

class ModelLoader(adhoc.AdhocLoader):

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

    # def start_progress_bar(self, total):
    #     self.lazy_load()
    #     self.progress_bar = adhoc.progress_bar(total=total, desc=f"{self.modeltag}")

    # def end_progress_bar(self):
    #     if self.progress_bar:
    #         self.progress_bar.close()
    #         self.progress_bar = None

    def compute_loss(self, input_texts: Union[List[str],str], progress_bar=None) -> float:
        raise NotImplementedError()
 
    def supported_gen_args(self) -> List[str]:
        return []

    def safe_gen_args(self, **kwargs):
        return adhoc.safe_kwargs(kwargs, self.supported_gen_args(), unsafe='GEN')

    def generate(self, input_texts: Union[List[str],str], n=1, progress_bar=None, /, **kwargs) -> Union[List[str], str]:
        output_texts = []
        for input_text in listfy(input_texts):
            output_texts.append(self.generate_s(input_text, n, **kwargs))
            if progress_bar:
                progress_bar.update(1)
        return singlefy_if_single(output_texts)

    def generate_s(self, input_text: str, n=1, /, **kwargs):
        return singlefy_if_single([f"({input_text})"] * n)

    @classmethod
    def regiser(cls, scheme):
        global MODEL_MAP
        MODEL_MAP[scheme] = cls

###
# OpenAI

class OpenAIModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        openai = adhoc.safe_import('openai')
        api_key = adhoc.get(kwargs, "openai_api_key|api_key|!!")
        self.client = openai.OpenAI(api_key=api_key)

    def supported_gen_args(self) -> List[str]:
        return [
            ## https://platform.openai.com/docs/api-reference/chat/create
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens|max_new_tokens",
            "presence_penalty",
            "response_format",
            "seed",
            "service_tier",
            "stop",
            # "stream",
            "temperature",
            "top_p",
        ]

    def generate(self, input_texts: Union[List[str], str], n=1, progress_bar=None, /, **kwargs):
        gen_args = adhoc.parse_value_of_args(**kwargs)
        if "max_new_tokens" in gen_args:
            # すごくアドホックな解決策
            gen_args["max_tokens"] = gen_args.pop("max_new_tokens")
        if "num_return_sequences" in self.gen_args:
            gen_args.pop("num_return_sequences")
        return super().generate(input_texts, n, progress_bar, **gen_args)

    def generate_s(self, input_text: str, n=1, /, **gen_args):
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=[{"role": "user", "content": input_text}],
            n=n,
            **gen_args,
        )
        responses = [choice.message.content for choice in response.choices]
        return singlefy_if_single(responses)

OpenAIModel.regiser("openai")

# class BedrockModel(Model):
#     def __init__(self, model_path, kwargs):
#         super().__init__(model_path, kwargs)
#         try:
#             import boto3

#             self.bedrock = boto3.client(
#                 "bedrock-runtime",
#                 aws_access_key_id=adhoc.get(kwargs, "aws_access_key_id"],
#                 aws_secret_access_key=adhoc.get(kwargs, "aws_secret_access_key"],
#                 region_name=adhoc.get(kwargs, "region_name|=ap-northeast-1"],
#             )
#         except ModuleNotFoundError as e:
#             raise e
#         default_args = {
#             "max_tokens_to_sample": adhoc.get(kwargs, "max_tokens|max_length|=512"],
#             "temperature": adhoc.get(kwargs, "temperature|=0.2"],
#             "top_p": adhoc.get(kwargs, "top_p|=0.95"],
#         }
#         self.generate_args = default_args

#     def check_and_append_claude_format(self, prompt: str) -> str:
#         ## FIXME: 改行の位置はここでいいのか？
#         human_str = "\n\nHuman:"
#         assistant_str = "\n\nAssistant:"

#         if human_str not in prompt:
#             prompt = human_str + prompt

#         if assistant_str not in prompt:
#             prompt += assistant_str

#         return prompt

#     def generate_text(self, prompt: str) -> str:
#         prompt = self.check_and_append_claude_format(prompt)
#         body = json.dumps(
#             {
#                 "prompt": prompt,
#                 "anthropic_version": "bedrock-2023-05-31",
#                 **self.generate_args,
#             }
#         )
#         response = self.bedrock.invoke_model(body=body, modelId=self.model_path)
#         response_body = json.loads(response.get("body").read())
#         return response_body.get("completion")


# BedrockModel.regiser("bedrock")

## HF

model32_kw_list = [
    "use_auth_token", 
    "trust_remote_code", 
    "device_map", 
    "attn_implementation"
]

def load_hfmodel32(model_path, /, **kwargs):
    from transformers import AutoModelForCausalLM

    model_args = adhoc.safe_kwargs(kwargs, model32_kw_list)
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
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM

        model_args = adhoc.safe_kwargs(kwargs, model4bit_kw_list)

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
        kwargs["torch_dtype"] = to_dtype(kwargs["torch_dtype"])

    if adhoc.get(kwargs, "use_4bit|=False"):
        return load_hfmodel4bit(model_path, **kwargs)
    else:
        return load_hfmodel32(model_path, **kwargs)

@adhoc.parse_value
def parse_dtype(dtype):
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
    if len(texts) == 1:
        if progress_bar:
            progress_bar.update(1)
        yield texts[0]
    else:
        for text in texts:
            if progress_bar:
                progress_bar.update(1)
            yield text


class HFModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lazy_kwargs = {**kwargs}
        self.progress_bar = None
        if not kwargs.get('_lazy', False):
            self.lazy_load()

    def lazy_load(self):
        if self.lazy_kwargs is None:
            return
        transformers = adhoc.safe_import('transformers')
        kwargs = self.lazy_kwargs
        self.lazy_kwargs = None

        tokenizer_path = adhoc.get(kwargs, f'tokenizer_path|_subpath|model_path|={self.path}')
        self._tokenizer = adhoc.load('_tokenizer', tokenizer_path, **kwargs)

        # なぜか必要らしい（↓）
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # if 'max_length' in kwargs:
        #     self.tokenizer.trancation = True
        model_path = adhoc.get(kwargs, f'_subpath|model_path|={self.path}')
        self._model, self.pathargs = load_hfmodel(model_path, **kwargs)
        self.device = next(self._model.parameters()).device
        adhoc.verbose_print(f"Model has loaded on {self.device}.///モデルは{self.device}上にロードされました")
        hf_token = adhoc.get(kwargs, "use_auth_token|HF_TOKEN")
        self.generator = transformers.pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            use_auth_token=hf_token,
        )
        # if 'max_length' in gen_args:
        #     gen_args['trancation'] = True
        # if "return_full_text" not in self.gen_args:
        #     self.gen_args["return_full_text"] = False
        # if "pad_token_id" not in self.gen_args:
        #     self.gen_args["pad_token_id"] = self.tokenizer.eos_token_id

    @property
    def modeltag(self):
        if self.tag != '':
            return self.tag
        return basename(self.path, split_ext=False)

    @property
    def model(self):
        self.lazy_load()
        return self._model

    @property
    def tokenizer(self):
        self.lazy_load()
        return self._tokenizer

    def supported_gen_args(self) -> List[str]:
        return [
            ## 4.39.0 https://huggingface.co/docs/transformers/main_classes/text_generation
            "max_length",  # (int, optional, defaults to 20) — The maximum length the generated tokens can have.
            "max_new_tokens",  # (int, optional) — The maximum numbers of tokens to generate
            "min_length",  # (int, optional, defaults to 0) — The minimum length of the sequence to be generated
            "min_new_tokens",  # (int, optional) — The minimum numbers of tokens to generate
            "early_stopping",  # (defaults to False) Controls the stopping condition for beam-based methods, like beam-search.
            "do_sample",  # (bool, optional, defaults to False) — Whether or not to use sampling ; use greedy decoding otherwise.
            "num_beams",  # (int, optional, defaults to 1) — Number of beams for beam search. 1 means no beam search.
            "num_beam_groups",  # (int, optional, defaults to 1) — Number of groups to divide num_beams into in order to ensure diversity among different groups of beams. this paper for more details.
            "penalty_alpha",  # (float, optional) — The values balance the model confidence and the degeneration penalty in contrastive search decoding.
            "temperature",  # (float, optional, defaults to 1.0) — The value used to modulate the next token probabilities.
            "top_k",  # (int, optional, defaults to 50) — The number of highest probability vocabulary tokens to keep for top-k-filtering.
            "top_p",  # (float, optional, defaults to 1.0) — If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            "typical_p",  # (float, optional, defaults to 1.0) — Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation. See this paper for more details.
            "epsilon_cutoff",  # (float, optional, defaults to 0.0) — If set to float strictly between 0 and 1, only tokens with a conditional probability greater than epsilon_cutoff will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model. See Truncation Sampling as Language Model Desmoothing for more details.
            "eta_cutoff",  # (float, optional, defaults to 0.0) — Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between 0 and 1, a token is only considered if it is greater than either eta_cutoff or sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))). The latter term is intuitively the expected next token probability, scaled by sqrt(eta_cutoff). In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model. See Truncation Sampling as Language Model Desmoothing for more details.
            "diversity_penalty",  # (float, optional, defaults to 0.0) — This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time. Note that diversity_penalty is only effective if group beam search is enabled.
            "repetition_penalty",  # (float, optional, defaults to 1.0) — The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
            "encoder_repetition_penalty",  # (float, optional, defaults to 1.0) — The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the original input. 1.0 means no penalty.
            "length_penalty",  # (float, optional, defaults to 1.0) — Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
            "no_repeat_ngram_size",  # (int, optional, defaults to 0) — If set to int > 0, all ngrams of that size can only occur once.
            "bad_words_ids",  # (List[List[int]], optional) — List of list of token ids that are not allowed to be generated. Check NoBadWordsLogitsProcessor for further documentation and examples.
            "force_words_ids",  # (List[List[int]] or List[List[List[int]]], optional) — List of token ids that must be generated. If given a List[List[int]], this is treated as a simple list of words that must be included, the opposite to bad_words_ids. If given List[List[List[int]]], this triggers a disjunctive constraint, where one can allow different forms of each word.
            "renormalize_logits",  # (bool, optional, defaults to False) — Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones). It’s highly recommended to set this flag to True as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization.
            "constraints",  # (List[Constraint], optional) — Custom constraints that can be added to the generation to ensure that the output will contain the use of certain tokens as defined by Constraint objects, in the most sensible way possible.
            "forced_bos_token_id",  # (int, optional, defaults to model.config.forced_bos_token_id) — The id of the token to force as the first generated token after the decoder_start_token_id. Useful for multilingual models like mBART where the first generated token needs to be the target language token.
            "forced_eos_token_id",  # (Union[int, List[int]], optional, defaults to model.config.forced_eos_token_id) — The id of the token to force as the last generated token when max_length is reached. Optionally, use a list to set multiple end-of-sequence tokens.
            "remove_invalid_values",  # (bool, optional, defaults to model.config.remove_invalid_values) — Whether to remove possible nan and inf outputs of the model to prevent the generation method to crash. Note that using remove_invalid_values can slow down generation.
            "exponential_decay_length_penalty",  # (tuple(int, float), optional) — This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated. The tuple shall consist of: (start_index, decay_factor) where start_index indicates where penalty starts and decay_factor represents the factor of exponential decay
            "suppress_tokens",  # (List[int], optional) — A list of tokens that will be suppressed at generation. The SupressTokens logit processor will set their log probs to -inf so that they are not sampled.
            "begin_suppress_tokens",  # (List[int], optional) — A list of tokens that will be suppressed at the beginning of the generation. The SupressBeginTokens logit processor will set their log probs to -inf so that they are not sampled.
            "forced_decoder_ids",  # (List[List[int]], optional) — A list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling. For example, [[1, 123]] means the second generated token will always be a token of index 123.
            "sequence_bias",  # (Dict[Tuple[int], float], optional)) — Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the sequence being selected, while negative biases do the opposite. Check SequenceBiasLogitsProcessor for further documentation and examples.
            "guidance_scale",  # (float, optional) — The guidance scale for classifier free guidance (CFG). CFG is enabled by setting guidance_scale > 1. Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer quality.
            "low_memory",  # (bool, optional) — Switch to sequential beam search and sequential topk for contrastive search to reduce peak memory. Used with beam search and contrastive search.
        ]

    def unwrap(self):
        self.lazy_load()
        return self._model

    def generate(self, input_texts: Union[List[str],str], n=1, progress_bar=None, /, **kwargs) -> Union[List[str], str]:
        self.lazy_load()
        gen_args = adhoc.parse_value_of_args(kwargs)
        gen_args['num_return_sequences'] = n
        if "return_full_text" not in gen_args:
            gen_args["return_full_text"] = False
        if "max_length" in gen_args and "max_new_tokens" in gen_args:
            gen_args.pop("max_length")
        gen_args.pop('n', None)
        gen_args.pop('progress_bar', None)
        input_texts = listfy(input_texts)
        outputs = self.generator(text_stream(input_texts, progress_bar), **gen_args)
        output_texts = []
        for results in outputs:
            generated_texts = [item["generated_text"] for item in results]
            output_texts.append(singlefy_if_single(generated_texts))
        return singlefy_if_single(output_texts)
        
    def compute_loss(self, input_texts: Union[List[str],str], progress_bar=None) -> Union[List[float], float]:
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

# class vLLMModel(Model):
#     def __init__(self, model_path, kwargs):
#         from vllm import LLM, SamplingParams

#         super().__init__(model_path, kwargs)
#         self.llm = LLM(model=model_path)
#         self.SamplingParams = SamplingParams
#         self.gen_args = {}

#     def eval_loss(self, sample_list: Union[List[dict], dict]):
#         sampling_params = self.SamplingParams(**self.gen_args)
#         sample_list = listfy(sample_list)
#         prompts = [sample["input"] for sample in sample_list]
#         outputs = self.llm.generate(prompts, sampling_params)
#         for i, output in enumerate(outputs):
#             sample = sample_list[i]
#             sample["loss"] = math.log(output.outputs[0].perplexity)

#     def eval_gen(
#         self, sample_list: Union[List[dict], dict], n=1, **kwargs
#     ) -> List[str]:
#         args = self.gen_args | dict(
#             n=n,
#         )
#         sampling_params = self.SamplingParams(**args)
#         sample_list = listfy(sample_list)
#         prompts = [sample["input"] for sample in sample_list]
#         outputs = self.llm.generate(prompts, sampling_params)
#         for i, output in enumerate(outputs):
#             sample = sample_list[i]
#             sample["output"] = [item.text for item in output.outputs]
#             if n == 1:
#                 sample["output"] = sample["output"][0]


