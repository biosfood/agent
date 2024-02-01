from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM

model_name = "microsoft/phi-2" # "Intel/neural-chat-7b-v3-3"#"./phi-2.Q8_0.gguf" # "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"# "microsoft/phi-2"
# model_name = "~/.cache/huggingface/hub/models--Intel--neural-chat-7b-v3-3/snapshots/7b86016aa1d2107440c1928694a7bba926509887"
# model_name = "/home/lukas/.cache/huggingface/hub/models--Intel--neural-chat-7b-v3-3/snapshots/7b86016aa1d2107440c1928694a7bba926509887"
# model_name = "ToolBench/ToolLLaMA-2-7b-v2"

model_name = "../../neural-chat-7b-v3-3"
model_name = "../../ToolLLaMA"
model_name = "/home/lukas/projects/neural-chat-7b-v3-3/"

prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, local_files_only=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
print(outputs)
