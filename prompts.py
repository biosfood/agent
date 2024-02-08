from langchain.prompts import PromptTemplate

template = """The assistant is a state of the art assistant but has not received recent updates. However, the assistant is allowed to search for articles to form a well researched anwser to its question.
When searching for aritcles, it is important to only use few words and just include relevant keywords. Some examples for good queries are: "transformer language model architecture" or "methanol consumption effects".
Istruct: {question}
Output: Relevant articles to answer this question can be found by the following query: \""""
keyword_template = PromptTemplate.from_template(template)

template = """The assistant is a state of the art assistant but has not received recent updates. However, the assistant is allowed to search for articles to form a well researched anwser to its question.
After searching for articles matching the query, we will now have to judge them on their likelihood of being helpful.
Istruct: You are trying to answer the question: \"{question}\". In an effort to answer this, you have searched for articles using the query \"{query}\". One specifc article we found has the title \"{title}\". Will this article be useful and worth investigating further.
Output: On a scale from 1 to 10, the likelyhood of this article helping me answer the question \"{question}\" is: \""""
value_template = PromptTemplate.from_template(template)

template = """The assistant is a state of the art assistant but has not received recent updates. However, the assistant is allowed to search for articles to form a well researched anwser to its question.
After finding a suitable article, we now have to create a concise answer using only the information provided.

Context:
{context}

Answer the question only using the provided context and keep your answer to a single paragraph
Istruct: {question}
Output: 
"""
summarize_template = PromptTemplate.from_template(template)

template = """The assistant is a state of the art assistant but has not received recent updates. However, the assistant is allowed to search for articles to form a well researched anwser to its question.
After looking at some different articles, we were able to summarize them in the following set of expert opinions on the topic:

{opinions}

Answer the question by summarizing the opinions in a single paragraph and also point out weather all of the opinions align.
Istruct: {question}
Output: 
"""
result_template = PromptTemplate.from_template(template)
