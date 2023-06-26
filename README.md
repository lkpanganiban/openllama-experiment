# OpenLlama with LangChain

This is a repository for my experimentation with OpenLlama 3B with LangChain. This implements a CPU execution OpenLlama3B. Note that depending on your hardware it may take a long time to execute. The aim of this repository is to have working implementation of the LLM Q and A setup without the need of external APIs like OpenAI or specialized hardware like GPUs. 

## Requirements
### Hardware Requirements
  - >16 GB RAM *(The more RAM the better)*
### System Requirements
  - Git LFS
### Python Libraries
  - torch
  - transformers
  - langchain
  - sentencepiece
### Model
  - [OpenLlama 3B](https://huggingface.co/openlm-research/open_llama_3b)

## Setup
### General Setup
1. Create the following folders `openlm-research` and `offload`.
    ```
    mkdir -p openlm-research offload
    ```
2. Clone the following [https://huggingface.co/openlm-research/open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b) under the `openlm-research`. Refer to the hugging face link on how to clone the model using Git LFS.
3. Install the Python Libraries by using `pip`.

### Running a generalized Assistant
1. Modify the `question_chat` in `questions.py`.
2. Execute `run_lang.py` and wait for it to generate the answers.

### Running a question and answer assistant for your documents
1. Modify the `question_q_and_a` in `questions.py` and the data directory. 
2. Modify the `run_doc_openllamaembed.py` or `run_doc_huggingfaceembed.py` if you made any changes in the data to be queried.
3. Execute `run_doc_openllamaembed.py` or `run_doc_huggingfaceembed.py` and wait for it to generate the answers.

NOTE: There is a `run.py` which has a sample implementation directly lifted from the huggingface repository of the OpenLlama 3B.

## Execution Notes
- This repository is running with the following specifications
    - CPU: Ryzen 4500u
    - RAM: 8GB with 16GB Swap
    - GPU: None
    - OS: Ubuntu 20.04 under WSL2
- The execution is a pure CPU implementation. If you have access to GPUs and other specialized hardware then you'll have better developer exeperience and execution times - you will need to modify the codebase the use the GPU.
- The `questions.py` contains the list of questions to do the benchmarking and execution.
- In the `run_lang.py` there is a line containing the pre-prompt used. You can modify the pre-prompt to improve the accuracy and performance.
- The model used the OpenLlama 3B this is due to the limitation of the hardware which is only 8GB of RAM. 
- The OpenLlama 3B model consumes about 16-20GB of memory. 
- A swap file is configured which is about 16GB. If you can allocate more RAM then the need of a SWAP file is reduced.
- Depending on the complexity of the question and the hardware used (swap, CPU, memory, etc.), the execution times may run from 2 mins to 10 mins.