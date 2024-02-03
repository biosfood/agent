from model import llm

from langchain.prompts import PromptTemplate

import arxiv
client = arxiv.Client()


template = """The assistant is a state of the art assistant but has not received recent updates. However, the assistant is allowed to search for articles to form a well researched anwser to its question.
When searching for aritcles, it is important to only use few words and just include relevant keywords. Some examples for good queries are: "transformer language model architecture" or "methanol consumption effects".
Istruct: {question}
Output: Relevant articles to answer this question can be found by the following query: \""""

question = "How much coffee can I consume in a day without fearing negative health outcomes?"

keyword_chain = PromptTemplate.from_template(template) | llm

query = keyword_chain.invoke({"question": question}).split("\"")[0]

print(f"seaching with keyword \"{query}\"")

search = arxiv.Search(
  query = query,
  max_results = 100,
  sort_by = arxiv.SortCriterion.Relevance
)
results = client.results(search)

template = """The assistant is a state of the art assistant but has not received recent updates. However, the assistant is allowed to search for articles to form a well researched anwser to its question.
After searching for articles matching the query, we will now have to judge them on their likelihood of being helpful.
Istruct: You are trying to answer the question: \"{question}\". In an effort to answer this, you have searched for articles using the query \"{query}\". One specifc article we found has the title \"{title}\". Will this article be useful and worth investigating further.
Output: On a scale from 1 to 10, the likelyhood of this article helping me answer the question \"{question}\" is: \""""

value_chain = PromptTemplate.from_template(template) | llm

for result in results:
    value = value_chain.invoke({"question": question, "query": query, "title": result.title}).split("\"")[0]
    print(f"\"{result.title}\": {value}")
