model_id = "/opt/models/microsoft/phi-2/"

import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer, StoppingCriteria


from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
from intel_extension_for_transformers.transformers.pipeline import pipeline

woq_config = WeightOnlyQuantConfig(weight_dtype="int4", compute_dtype="int8")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=woq_config,trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=False)


class MyStop(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_token_id = [tokenizer.eos_token_id]

    def __call__(
        self, input_ids, scores, **kwargs
    ) -> bool:
        return tokenizer.decode(input_ids[0]).endswith("\n") or input_ids[0][-1] in self.stop_token_id 

from intel_extension_for_transformers.transformers.pipeline import pipeline
llm = HuggingFacePipeline(pipeline=pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    stopping_criteria=MyStop(tokenizer)
))
