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
        return tokenizer.decode(input_ids[0]).endswith("\n") or input_ids[0][-1] in self.stop_token_id 

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

template = """The assistant is a state of the art assistant but has not received recent updates. However, the assistant is allowed to search for articles to form a well researched anwser to its question.
When searching for aritcles, it is important to only use few words and just include relevant keywords. Some examples for good queries are: "transformer language model architecture" or "methanol consumption effects".
Istruct: {question}
Output: Relevant articles to answer this question can be found by the following query: \""""

question = "How much coffee can I consume in a day?"

keyword_chain = PromptTemplate.from_template(template) | llm

query = keyword_chain.invoke({"question": question}).split("\"")[0]

print(f"seaching with keyword \"{query}\"")

import arxiv
client = arxiv.Client()

search = arxiv.Search(
  query = query,
  max_results = 3,
  sort_by = arxiv.SortCriterion.Relevance
)

results = client.results(search)

for result in results:
    print(result)