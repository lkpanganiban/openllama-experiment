#%%
'''
This a sample Q and A for your documents using Huggingface Embeddings.
You may modify this codebase as you see fit.
'''

import time

# load the data to be queried
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.document_loaders import TextLoader
from questions import questions_q_and_a

# load embeddings => this can be derived from the actual LLM
embeddings = HuggingFaceEmbeddings()

# load data to vector store
loader = TextLoader('data/paul_graham_essay.txt') # replace with your text data
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
db = FAISS.from_documents(texts, embeddings)

# do the the query of document
for question in questions_q_and_a:
    start_t = time.time()
    docs = db.similarity_search(question)
    print(f"Question: {question}")
    print(f"Answer: {docs[0].page_content}")
    print("Execution time", time.time() - start_t)
    print(100*"=")

