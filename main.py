from model import llm

from langchain.prompts import PromptTemplate

import arxiv
client = arxiv.Client()

import requests

template = """The assistant is a state of the art assistant but has not received recent updates. However, the assistant is allowed to search for articles to form a well researched anwser to its question.
When searching for aritcles, it is important to only use few words and just include relevant keywords. Some examples for good queries are: "transformer language model architecture" or "methanol consumption effects".
Istruct: {question}
Output: Relevant articles to answer this question can be found by the following query: \""""

question = "How much coffee can I consume in a day without fearing negative health outcomes?"

keyword_chain = PromptTemplate.from_template(template) | llm

query = keyword_chain.invoke({"question": question}).split("\"")[0]

print(f"seaching with keyword \"{query}\"")

results = []

search = arxiv.Search(
  query = query,
  max_results = 20,
  sort_by = arxiv.SortCriterion.Relevance
)
arxiv_results = client.results(search)

for result in arxiv_results:
    results.append({"title": result.title, "abstract": result.summary})

payload = {
    "q": query,
    "sort": "relevance"
}

result = requests.get("https://api.core.ac.uk/v3/search/works", params=payload)
for result in result.json()["results"]:
    if result["abstract"] == None or len(result["abstract"]) < 10:
        continue
    results.append(result)

template = """The assistant is a state of the art assistant but has not received recent updates. However, the assistant is allowed to search for articles to form a well researched anwser to its question.
After searching for articles matching the query, we will now have to judge them on their likelihood of being helpful.
Istruct: You are trying to answer the question: \"{question}\". In an effort to answer this, you have searched for articles using the query \"{query}\". One specifc article we found has the title \"{title}\". Will this article be useful and worth investigating further.
Output: On a scale from 1 to 10, the likelyhood of this article helping me answer the question \"{question}\" is: \""""

value_chain = PromptTemplate.from_template(template) | llm

# deduplicate result titles
titles = set()
for result in results:
    if result["title"] not in titles:
        titles.add(result["title"])
    else:
        results.remove(result)

print(f"found {len(results)} results")

for result in results:
    title = result["title"]
    value = value_chain.invoke({"question": question, "query": query, "title": title}).split("\"")[0]
    result["value"] = int(value)

template = """The assistant is a state of the art assistant but has not received recent updates. However, the assistant is allowed to search for articles to form a well researched anwser to its question.
After finding a suitable article, we now have to create a concise answer using only the information provided.

Context:
{context}

Answer the question only using the provided context and keep your answer to a single paragraph
Istruct: {question}
Output: 
"""

answer_chain = PromptTemplate.from_template(template) | llm

print("top 5 results:")
results = sorted(results, key=lambda x: x["value"], reverse=True)[:5]
for result in results:
    print(f"\"{result['title']}\": {result['value']}")
    answer = answer_chain.invoke({"context": result["abstract"], "question": question})
    print(answer)
