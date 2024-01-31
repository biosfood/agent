import torch
import re
import sys

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList

model_id = "/opt/models/microsoft/phi-2/"

from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
from intel_extension_for_transformers.transformers.pipeline import pipeline

woq_config = WeightOnlyQuantConfig(weight_dtype="int4", compute_dtype="int8")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=woq_config,trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=False)


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(
        self, input_ids, scores, **kwargs
    ) -> bool:
        return tokenizer.decode(input_ids[0]).endswith("\n\n") or input_ids[0][-1] in self.stop_token_id 

stopping_criteria = StoppingCriteriaList(
    [
        StopOnTokens(
            stop_token_id=[tokenizer.eos_token_id]
        )
    ]
)

from intel_extension_for_transformers.transformers.pipeline import pipeline
llm = HuggingFacePipeline(pipeline=pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    stopping_criteria=stopping_criteria
))

from langchain.prompts import PromptTemplate

template = """You are a very intelligent AI assistant but were not updated recently. You do not have knowlege of recent events or scientific developments.
Your goal is to provide accurate and helpful information. To do this, you will first have to look for relevant scientific articles.
Provide short and consice answers wherever possible.
Instruct: List 3 keywords for scientific articles that are most relevant to answer the question "{question}"!
Output: Articles that will help answer the question can be found by the following keywords:
- \""""

prompt = PromptTemplate.from_template(template)
chain = prompt | llm
print(chain.invoke({"question": "How much coffee can I consume in a day?"}))