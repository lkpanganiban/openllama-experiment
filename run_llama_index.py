#%%
'''
This a sample Q and A using llama_index as the implementation.
You may modify this codebase as you see fit.
'''
import torch
from typing import Optional, List, Mapping, Any
from transformers import LlamaTokenizer, LlamaForCausalLM
from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import LLMPredictor, ServiceContext, LangchainEmbedding

print("loading models")
# set context window size
context_window = 2048
# set number of output tokens
num_output = 256
model_path = "openlm-research/open_llama_3b"
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32, # torch.float16 if you are using GPUs
    device_map="auto",
    offload_folder="offload",
)

tokenizer = LlamaTokenizer.from_pretrained(model_path)

from transformers import StoppingCriteria, StoppingCriteriaList, pipeline

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

pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task="text-generation",
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    repetition_penalty=1.2,  # without this output begins repeating
)

print("setting up embeddings")
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
print("defining OpenLlamaLLM")
class OpenLlamaLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]
        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": model_path}

    @property
    def _llm_type(self) -> str:
        return "custom"


llm_predictor = LLMPredictor(llm=OpenLlamaLLM())

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    context_window=context_window,
    num_output=num_output,
    embed_model=embed_model
)


print("finished loading model")
print("loading data")

documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

print("running the query")
query_engine = index.as_query_engine()
query = "In the spring of 2000, what is his idea?"
print("Q:", query)
response = query_engine.query(query)
print(response)