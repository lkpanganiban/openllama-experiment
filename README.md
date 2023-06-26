# OpenLlama with LangChain

This is a repository for my experimentation with OpenLlama 3B with LangChain. This implements a CPU execution of Q and A. Note that depending on your hardware it may take a long time to execute. The aim of this repository is to have working implementation of the LLM Q and A setup without the need of external APIs like OpenAI or specialized hardware like GPUs. 

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
1. Create the following folders `openlm-research` and `offload`.
    ```
    mkdir -p openlm-research offload
    ```
2. Clone the following [https://huggingface.co/openlm-research/open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b) under the `openlm-research`. Refer to the hugging face link on how to clone the model using Git LFS.
3. Install the Python Libraries by using `pip`.
4. Modify the `questions.py`
5. Execute `run_lang.py` and wait for it to generate the answers.

NOTE: There is a `run.py` which has a sample implementation directly lifted from the huggingface repository of the OpenLlama 3B.

## Execution Notes
- The execution is a pure CPU implementation. If you have access to GPUs and other specialized hardware then you'll have better developer exeperience and execution times - you will need to modify the codebase the use the GPU.
- The `questions.py` contains the list of questions to do the benchmarking and execution.
- In the `run_lang.py` there is a line containing the pre-prompt used. You can modify the pre-prompt to improve the accuracy and performance. 
- The model used the OpenLlama 3B this is due to the limitation of the hardware which is only 8GB of RAM. 
- The OpenLlama 3B model consumes about 16-20GB of memory. 
- A swap file is configured which is about 16GB. If you can allocate more RAM then the need of a SWAP file is reduced.
- Depending on the complexity of the question and the hardware used (swap, CPU, memory, etc.), the execution times may run from 2 mins to 10 mins.