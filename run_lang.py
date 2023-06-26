#%%
'''
This a sample inference of the OpenLlama 3B with the context of a Question and Answer assistant.
This implements 
You may modify this codebase as you see fit.
'''

import time
import torch
import transformers
from questions import questions_chat
from transformers import LlamaTokenizer, LlamaForCausalLM
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

start_t = time.time()

# model definition
model_path = "openlm-research/open_llama_3b"
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32, # torch.float16 if you are using GPUs
    device_map="auto",
    offload_folder="offload",
)

print(f"loading model done in {time.time() - start_t}  seconds")

# tokenizer definition
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# stopping criteria - https://github.com/pinecone-io/examples/blob/master/generation/llm-field-guide/open-llama/open-llama-huggingface-langchain.ipynb
from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
stop_token_ids = [
    tokenizer.convert_tokens_to_ids(x)
    for x in [
        [""],
        ["User", ":"],
        ["system", ":"],
        [tokenizer.convert_ids_to_tokens([9427])[0], ":"],
    ]
]
stop_token_ids = [torch.LongTensor(x).to("cpu") for x in stop_token_ids] # change the "cpu" to "gpu" if you are using GPUs

class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


stopping_criteria = StoppingCriteriaList([StopOnTokens()])

start_t = time.time()
# text generation pipeline
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task="text-generation",
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=256,  # max number of tokens to generate in the output
    repetition_penalty=1.2,  # without this output begins repeating
)
print(f"text generation pipeline done in {time.time() - start_t} seconds")

# Setup Pre-prompt
template = """You are a question and answer assistant and not a chatbot, the questions are not related to each other and you will answer the users query in a short but precise answer.
Remember, Assitant responses are short. Here is the conversation:

User: {query}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)

# Configure the Pipeline and Chain
start_t = time.time()
llm = HuggingFacePipeline(pipeline=generate_text)
llm_chain = LLMChain(llm=llm, prompt=prompt)
print(f"pipeline and chain configuration done in {time.time() - start_t} seconds")

# execute queries
def run_query(query):
    start = time.time()
    # ask a question
    print(f"Q: {query}")
    output = llm_chain.predict(query=query).lstrip().removesuffix("User:").replace("The user is satisfied with the response and moves on to another question.\n", "").replace("\n", "") # the llm chan generates other boilerplate to be removed.
    end = time.time()
    print(f"A: {output} ")
    print(f"Runtime: {end-start} seconds\n")

for question in questions_chat:
    run_query(question)
