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
        results = []
        for text in texts:
            response = self.embed_query(text)
            results.append(response)
        return results

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
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

embeddings = OpenLlamaEmbeddings()

# store the documents into a vectordb
with open('data/paul_graham_essay.txt', "r") as data_file:
    book = data_file.read()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(book)
db = Chroma.from_texts(texts, embeddings,  metadatas=[{"source": str(i)} for i in range(len(texts))])

# trigger the queries for the documents
queries = [
    "Before college what are the two main things he worked on?",
    "Who are allowed to take the exam?"
]
for query in queries:
    start_t = time.time()
    docs = db.similarity_search(query)
    print(f"Query: {query}")
    print(f"Answer: {docs[0].page_content}")
    print("Execution time", time.time() - start_t)
    print(100*"=")
