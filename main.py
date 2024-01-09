import torch
import re
import sys

print("loading model...")

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, StoppingCriteria, StoppingCriteriaList

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True)

class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, text):
        self.text = text

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        is_done = (tokenizer.decode(input_ids[0])).endswith(self.text)
        return is_done

stop = StoppingCriteriaList([StopSequenceCriteria("\n"), StopSequenceCriteria("##")])

model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, streamer=streamer)

class Pipe:
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        return self.p(x, stopping_criteria=stop)
    
    @property
    def task(self):
        return "text-generation"
    
    @property
    def _postprocess_params(self):
        return self.p._postprocess_params

llm = HuggingFacePipeline(pipeline=Pipe(pipe))

from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = {
    "Wikipedia": (wikipedia, "Search Wikipedia for information for a topic"),
}

template = """### System: You are a helpful agent but have limited knowlege about recent events and also are no expert in any topic.
To gather more information about a topic, you have the following tools available:
{tools_description}
When you think you have all the information necessary to answer the question posed by the user, respond with a short summary of your findings. Then, use the format ## Answer: <summary text>

### User: Where are the tanaica montes located?

### Assistant:
## Thought: I need to find out more about the tanaica montes.
## Action: Wikipedia[tanaica montes]
## Wikipedia Result: Tanaica Montes is a mountain range on the planet Mars. It has a diameter of 177 kilometres (110 mi). 
## Thought: The wikipedia article answered my question of where the tanaica montes are located.
## Answer: The tanaica montes are located on Mars.


### User: What is the difference between red and white wine?

### Assistant:
## Thought: I should compare the articles about red and white wine to find out the difference and answer the question.
## Action: Wikipedia[red wine]
## Wikipedia Result: Red wine is a type of wine made from dark-colored grape varieties. The color of the wine can range from intense violet, typical of young wines, through to brick red for mature wines and brown for older red wines. The juice from most purple grapes is greenish-white, the red color coming from anthocyan pigments present in the skin of the grape. Much of the red wine production process involves extraction of color and flavor components from the grape skin. 
## Thought: This gives me some information about red wine. I can compare this to the entry for white wine.
## Action: Wikipedia[white wine]
## Wikipedia Result: White wine is a wine that is fermented without skin contact. The colour can be straw-yellow, yellow-green, or yellow-gold.[1] It is produced by the alcoholic fermentation of the non-coloured pulp of grapes, which may have a skin of any colour. White wine has existed for at least 4,000 years. 
## Thought: By comparing the two articles, I know how to describe the difference between red and white wine.
## Answer: The difference between red and white wine comes from wether the skin of the grapes is left on or not. For red wine, red or violet grapes are typically used and the flavour and color is also extracted from the grape skin. For white wine, the the skin is removed from the grapes.


### User: {question}

### Assistant:
{agent_scratchpad}"""

from langchain.prompts import PromptTemplate
p = PromptTemplate.from_template(template)

pipeline = p | llm.bind(stop=["\n"])
def describeTools(tools):
    return "\n".join(f"- {name}: {description}" for name, (tool, description) in tools.items())

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.agent import AgentOutputParser

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
@tool
def thought(text: str):
    """Just a buffer for the model to think"""
    return ""

@tool("Wikipedia")
def wiki(text: str):
    """Access wikipedia"""
    result = wikipedia.run(text)
    result = " ".join(result.split("\n\n\n")[0].split("\n")[1:]).split("Summary: ")[1]
    return result

class ReActSingleInputOutputParser(AgentOutputParser):
    def get_format_instructions(self):
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str):
        if text.startswith(" "):
            text = text[1:]
        text = text.replace("\n", "")
        if text.startswith("Answer:"):
            return AgentFinish({"output": text[len("Answer:") :]}, text)
        if text.startswith("Action:"):
            regex = re.compile(r"Action: (\w+)\[(.*)\]")
            result = regex.match(text)
            if result is None:
                raise OutputParserException(f"Could not parse action: {text}")
            function = result.group(1)
            argument = result.group(2)
            print(f"running action: {function} with argument: {argument}")
            return AgentAction(function, argument, f"{function}[{argument}]")
        if text.startswith("Thought:"):
            thought = text[len("Thought: ") :]
            return AgentAction("thought", thought, f"Thought: {thought}")
        raise OutputParserException(f"Could not parse action: {[text]}")

def formatMessages(messages):
    result = ""
    print("")
    for message in messages:
        if message[0].tool == "thought":
            result += f"## Thought: {message[0].tool_input}\n"
        else:
            result += f"## Action: {message[0].tool}[{message[0].tool_input}]\n"
            result += f"## {message[0].tool} Result: {message[1]}\n"
    result += "##"
    return result

agent = (
    {
        "question": lambda x: x["question"],
        "agent_scratchpad": lambda x: formatMessages(x["intermediate_steps"]),
        "tools_description": lambda x: describeTools(tools),
    }
    | p
    | llm.bind(stopping_criteria=stop)
    | ReActSingleInputOutputParser()
)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=[wiki, thought], verbose=True)
agent_executor.invoke({"question": sys.argv[1]})
