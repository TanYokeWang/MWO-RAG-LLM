# Retrieval-Augmented Large Language Model for Wind Turbine Maintenance Work-Oder Classification
This project ivestigates whether the adaptation of retrieval-augmented generation (RAG) can provide task-specific contextual information during inference to improve classification performance and narrow the performance gap between large proprietary models and small-scale open-source LLMs in wind turbine (WT) maintenance-text classification.    

Publication - 
> Under review 


## Overview
This repository contains the full source code used to produce the results presented in the publication. While the original dataset is confidential and could not be made public, a sample dataset is provided in the repository to allow users to investigate the functionality of the code. A compiled classification results for the evaluated models were also provided to allow users to recreate the figures attached in the publication. 


```text
## Repository Structure
configs/                                    # User-configurable experiment settings
├── label_database.json                     # Predefined label space and RAG context corpus
└── prompt_template.json                    # Templates for Base Prompt and RAG configurations

maintenance_llm/                            # Core source code for the classification framework
├── LLM_client/                             # LLM inference client classes
│   ├── gpt_client.py                       # OpenAI API inference client
│   └── ollama_client.py                    # Local Ollama inference client
├── prompts/                                # Prompt construction utilities
│   └── prompt_builder.py                   # Builds prompts from templates and input context
├── rag_module/                             # Retrieval and context-generation components
│   ├── bge.py                              # BGE-based semantic retriever
│   ├── context_compiler.py                 # Compiles retrieved context for prompt input
│   ├── minilm.py                           # MiniLM-based semantic retriever
│   └── tfidf.py                            # TF-IDF lexical retriever
└── utils/                                  # Shared utility functions
    ├── output_parser.py                    # Parses and validates LLM outputs
    ├── paths.py                            # Centralised project path definitions
    └── schema_builder.py                   # Builds structured-output schemas

notebook/                                   # Experiment execution and analysis notebooks
├── figure_plotter.ipynb                    # Generates publication figures
├── gpt_batch_run.ipynb                     # Submits and processes OpenAI Batch API jobs
├── gpt_client.ipynb                        # Interactive GPT inference workflow
├── ollama_client.ipynb                     # Interactive local Ollama inference workflow
└── Output_analysis.ipynb                   # Classification performance analysis

data/                                       # Input dataset directory
└── dataset.csv                             # Maintenance work-order dataset

results_public/                                     # Processed results used for public reporting
└── figures_data/                                   # Data used to generate publication figures
    ├── base_prompt_class_level_f1_score.csv
    ├── base_vs_rag.csv
    └── RAG_class_level_f1_score.csv
```


## Installation and Environment 

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/MWO-RAG-LLM.git
cd MWO-RAG-LLM
```

### 2. Create Python virtual environment 
The experiemtns reported in this study were conducted using Python 3.13.2. 
```bash
python -m venv .venv
```

### 3. Install Python dependencies 
```bash
pip install -r requirements.txt
```

### 4. Local LLM inference with Ollama
Ollama is required for the experiments conducted using locally deployed LLMs. Users need to install Ollama and their preferred open-source LLMs based on available comptutational resources. The copmplete official installation instructions are available on the official site: https://ollama.com/ 

The models identifiers used in the experiments are listed as below: 
- mistral:7b
- qwen3.5:9b 
- llama3:8b
- gemma3:4b 

### 5. API Key Configurations
An OpenAI API key is required for experiments using GPT models. The API keys should be saved as environmental variables and be careful not to commit them to any repositories.  



## Workflow
### 1. Prepare the dataset
The dataset_example attached in this repository has been preprocessed to correct typographical errors and abbreviations and removing whitespace, symbols, and numbers that were not part of known recognised component identifiers. As the preprocessing steps may differ according to different datasets, users are expected to setup their own preprocessing pipeline that is suitable for their unique datasets. Please note that it is best to maintain the [column names] described in the example to ensure smooth code execution, as many operations within the code were setup based on them.   

### 2. Configure the label database 
The label database defined in this work were based on previous study involving manual labelling by domain expert. The defined defintions, label boundaries and keywords may not reflect alternative dataset that users may want to investigate. To maximize the relevance of the context corpus, users should make necessary changes to the label space and contextual information accordingly. 

### 3. Analysis Workflow
Separate Notebooks were created to control each configurations investigated in this study. Detailed comments are available in each notebook to guide users on the execution of the analytical pipelines. The LLM workflow notebooks are separated as follow: 

Notebook for LLM inference -  
- gpt_batch_run.ipynb : notebook for running inference using GPT Batch API
- gpt_client.ipynb : notebook for running interactive (iterative) inference using GPT models via API access
- ollama_client.ipynb : notebook for running interactive (iterative) inference using local models

Notebook for LLM Output Analysis -  
- output_analysis.ipynb : notebook for extracting LLM inference outputs and generate hallucination and classification reports
- figure_plotter.ipynb : notebook for recreating the figures presented in the study

 