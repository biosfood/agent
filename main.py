from model import llm
from prompts import *

import arxiv
client = arxiv.Client()

import requests

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("arxiv")
logger.setLevel(logging.WARNING)

question = input("Query (leave empty for default testing query): ")

if question == "":
    question = "How much coffee can I consume in a day without fearing negative health outcomes?"

keyword_chain = keyword_template | llm
summarize_chain = summarize_template | llm
value_chain = value_template | llm
result_chain = result_template | llm

query = keyword_chain.invoke({"question": question}).split("\"")[0]

print(f"seaching with query \"{query}\"")

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

results = sorted(results, key=lambda x: x["value"], reverse=True)
opinions = []
for result in results[:5]:
    if result["value"] < 5:
        continue
    print(f"summarizing \"{result['title']}\" (got a score of {result['value']} / 10)")
    answer = summarize_chain.invoke({"context": result["abstract"], "question": question})
    opinions.append(answer)

opinions_text = "\n".join(opinions)

answer = result_chain.invoke({"opinions": opinions_text, "question": question})
print(answer)
