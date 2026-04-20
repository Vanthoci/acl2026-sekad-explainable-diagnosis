# Towards Explainable Diagnosis: A Self-learned Explanatory Knowledge Base Approach

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

Abstract:

Explainable diagnosis requires that authoritative medical knowledge provide the rationales linking a patient’s clinical manifestations to the diagnostic conclusion. Although large language models (LLMs) hold great potential to facilitate explainable diagnosis, their effectiveness is often constrained by insufficient diagnostic expertise. To address this limitation, we propose Self-learned Explainable Knowledge Augmented Diagnosis (SEKAD), a unified LLM-based framework for faithful and explainable diagnosis. Our approach builds a high-quality diagnostic knowledge base through a record-driven explanation learning paradigm, as well as applies this knowledge via an explanation-based diagnostic process that ensures faithful inference. Experiments on the DiReCT and JAMA benchmarks show that SEKAD consistently outperforms strong baselines across the metrics. In particular, on the DiReCT benchmark, SEKAD improves the explanation completeness metric from 64.5% to 76.9% over the best existing methods, highlighting its effectiveness in enhancing diagnostic explainability and showing that our text mining approach produces knowledge that is both reliable in quality and large in quantity

## Attribution
> The evaluation logic and core metrics in this repository are adapted and cited from the [DiReCT](https://github.com/wbw520/DiReCT) project. We thank the authors for their foundational work in medical diagnostic evaluation.

---

## Setup and Installation

### 1. Environment Setup
We recommend using Conda to manage your environment:

```bash
# Create the environment
conda create -n sekad python=3.10 -y
conda activate sekad

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory (refer to `.env.example`) to configure your LLM API:

```env
OPENAI_API_KEY="your_api_key"
OPENAI_BASE_URL="https://api.your-provider.com/v1/"
OPENAI_CHAT_MODEL="gpt-4o-mini"
```

---

## Data Structure

### Test Data & Samples
Due to copyright restrictions, the full dataset is not included in this repository. 
- **Samples**: A subset of test cases is provided in the `samples/` directory for demonstration.
- **Full Dataset**: To obtain the complete dataset, please contact the authors of the [DiReCT](https://github.com/wbw520/DiReCT) project.

### Expert Knowledge (Gold Rubrics)
The expert-curated diagnostic guidelines and knowledge graphs used for augmentation and evaluation are stored in the `diagnostic_kg/` directory.

### Task Configuration
The specific test tasks and directories are managed via the `Qsets` class in `config.py`. By default, the runner looks for data in the path specified by `Qsets.sample`.

---

## Usage

### 1. Run Diagnosis
To execute the full SEKAD diagnostic pipeline (including evidence extraction and chain-of-thought reasoning):

```bash
python run_diagnosis.py
```
*This script will generate predictions in the `predicts/` directory and automatically trigger the evaluation phase upon completion.*

### 2. Run Evaluation Only
If you have already generated predictions and wish to re-run the evaluation:

```bash
python run_evaluation.py predicts/your_prediction_folder
```

---

## Citation

If you use this work, please cite the following paper from the [DiReCT](https://github.com/wbw520/DiReCT) project:

```bibtex
@inproceedings{wangdirect,
 author = {Wang, Bowen and Chang, Jiuyang and Qian, Yiming and Chen, Guoxin and Chen, Junhao and Jiang, Zhouqiang and Zhang, Jiahao and Nakashima, Yuta and Nagahara, Hajime},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {74999--75011},
 title = {DiReCT: Diagnostic Reasoning for Clinical Notes via Large Language Models},
 volume = {38},
 year = {2024}
}
```
