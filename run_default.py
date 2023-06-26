#%%
'''
This a sample inference of the OpenLlama 3B.
This is directly lifted from https://huggingface.co/openlm-research/open_llama_3b.
You may modify this codebase as you see fit.
'''
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = 'openlm-research/open_llama_3b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32, device_map='auto',
    offload_folder="offload",
)

prompt = 'Q: What is the smallest mammal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))
