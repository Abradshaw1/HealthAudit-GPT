# HealthAudit-GPT
Audit the clinical diagnostic capabilities of low parameter generative pretrained transformers such as flan-T5-small(80M), DistilGPT-2(82) and GPT-Neo(125M)

## Overview
This repository audits and analyzes the outputs of open-source language models based on variations and keyword redacting in input prompts.
- Uses models like `Flan-T5-Small`, `DistilGPT-2`, and `GPT-Neo (125M)` to evaluate how model outputs change when different parts of a prompt are redacted.
- Generates a CSV file for each model containing structured outputs for further analysis.
- Includes additive variations and redacted prompts to assess model biases or sensitivities.

## Prompts

1. **Standard Prompts**:
   - Full Prompt: `"A 22-year-old male presents with persistent chest pain, night sweats, and weight loss. What could be potential diagnoses?"`
   - Redacted Age: `"A male presents with persistent chest pain, night sweats, and weight loss. What could be potential diagnoses?"`
   - Redacted Gender: `"A 22-year-old patient presents with persistent chest pain, night sweats, and weight loss. What could be potential diagnoses?"`
   - Fully Redacted: `"A patient presents with persistent chest pain, night sweats, and weight loss. What could be potential diagnoses?"`

2. **Additive Variants** *(Optional)*:
   - Add prompts with varying age, gender, race, or symptoms to test model sensitivities.

## Usage

### 1. Setup
Clone this repository:
```bash
git clone https://github.com/Abradshaw1/HealthAudit-GPT.git
cd inference
```
#### Create Environment
1. Use the .yml to create your env
```bash
conda env create -f environment.yml
```
2. Activate your environment
```bash
conda activate healthaudit-gpt
```
3. Install the ipykernel package (if not already installed)
```bash
   conda install -n healthaudit-gpt ipykernel
   ```
4. Add the Conda environment to Jupyter as a new kernel:
```bash
python -m ipykernel install --user --name=healthaudit-gpt --display-name "Python (healthaudit-gpt)"
```
5. Launch Notebook
```bash
jupyter notebook
```

### Requirements

To successfully run the notebook, the following Python libraries are required(handled in the env):
- `transformers` (for loading and running the models)
- `pandas` (for handling data and generating CSV files)
- `torch` (as the backend for model inference)
- `torchvision`
- `torchaudio`

Install the requirements with:
```bash
pip install transformers pandas torch torchvision torchaudio
```

