#%%
'''
This a sample Q and A for your documents using a custom LLM Embeding using OpenLlama.
You may modify this codebase as you see fit.
'''
import time
import torch
from pydantic import BaseModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from langchain.embeddings.base import Embeddings
from typing import List, Optional
from questions import questions_q_and_a

# reference: https://github.com/paolorechia/learn-langchain/blob/c6ede53d271a2587beb88a4907de089338a64195/servers/hf_loader.py
# model definition
model_path = "openlm-research/open_llama_3b"
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32, # torch.float16 if you are using GPUs
    device_map="auto",
    offload_folder="offload",
)
# tokenizer definition
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# generalized embedding function
def get_embeddings(prompt):
    if not isinstance(prompt, str): prompt = prompt.page_content
    input_ids = tokenizer(prompt).input_ids
    input_embeddings = model.get_input_embeddings()
    embeddings = input_embeddings(torch.LongTensor([input_ids]))
    mean = torch.mean(embeddings[0], 0).cpu().detach()
    return [float(x) for x in mean]

# create the openllama embeddings using the openllama3b
class OpenLlamaEmbeddings(BaseModel, Embeddings):
    def _call(self, prompt: str) -> str:
        return get_embeddings(prompt)

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Embed the documents.
        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.
        Returns:
            List of embeddings, one for each text.
        """
        return list(map(self.embed_query, texts))

    def embed_query(self, text) -> List[float]:
        """Embed the query or prompt.
        Args:
            text: The text to embed.
        Returns:
            Embedding for the text.
        """
        embedding = self._call(text)
        return embedding


# embed the texts using the openllamaembeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

embeddings = OpenLlamaEmbeddings()

# store the documents into a vectordb
loader = TextLoader('data/paul_graham_essay.txt') # replace with your text data
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
db = FAISS.from_documents(texts, embeddings)

# trigger the queries for the documents
for question in questions_q_and_a:
    start_t = time.time()
    docs = db.similarity_search(question)
    print(f"Question: {question}")
    print(f"Answer: {docs[0].page_content}")
    print("Execution time", time.time() - start_t)
    print(100*"=")
