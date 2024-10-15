import math
from .commons import *
from .models import Model

def load_model(**kwargs):
    from vllm import LLM
    
    llm = LLM(model=adhoc.get(kwargs, 'model_path|model|!!'), 
            #   gpu_memory_utilization=float(0.8),
            #   tensor_parallel_size=torch.cuda.device_count(),
            #   max_model_len=adhoc.get(kwargs, "max_model_length|=4096",
              trust_remote_code=True)
    return llm

class vLLMModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lazy_kwargs = {**kwargs}
        self.progress_bar = None
        vllm = adhoc.safe_import('vllm')
        self.SamplingParams = vllm.SamplingParams
        if not kwargs.get('_lazy', False):
            self.lazy_load()

    def lazy_load(self):
        if self.lazy_kwargs is None:
            return
        kwargs = self.lazy_kwargs
        self.lazy_kwargs = None

        tokenizer_path = adhoc.get(kwargs, f'tokenizer_path|_subpath|model_path|={self.path}')
        self._tokenizer = adhoc.load('_tokenizer', tokenizer_path, **kwargs)

        self._vllm = load_model(**kwargs)

    @property
    def modeltag(self):
        if self.tag != '':
            return self.tag
        return basename(self.path, split_ext=False)

    @property
    def vllm(self):
        self.lazy_load()
        return self._vllm

    @property
    def tokenizer(self):
        self.lazy_load()
        return self._tokenizer

    def unwrap(self):
        self.lazy_load()
        return self._vllm

    def generate(self, input_texts: Union[List[str],str], n=1, progress_bar=None, /, **kwargs) -> Union[List[str], str]:
        self.lazy_load()
        gen_args = adhoc.parse_value_of_args(kwargs)
        gen_args.pop('n', None)
        gen_args.pop('progress_bar', None)

        sampling_params = self.SamplingParams(gen_args)
        outputs = self._vllm.generate(input_texts, sampling_params)

        output_texts = []
        for output in outputs:
            generated_texts = [output.outputs[0].text for i in len(output.outputs)]
            output_texts.append(singlefy_if_single(generated_texts))
            if progress_bar:
                progress_bar.update(1)
        return singlefy_if_single(output_texts)
        
    def compute_loss(self, input_texts: Union[List[str],str], progress_bar=None) -> Union[List[float], float]:
        self.lazy_load()
        sampling_params = self.SamplingParams()
        outputs = self.llm.generate(listfy(input_texts), sampling_params)
        values = []
        for i, output in enumerate(outputs):
            values.append(math.log(output.outputs[0].perplexity))
            if progress_bar:
                progress_bar.update(1)
        return singlefy_if_single(values)

vLLMModel.regiser("vllm")

