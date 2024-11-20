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

## Model Clustering Analysis

The `modelclustering.qmd` script performs a clustering analysis on model outputs to assess diagnostic sensitivity and adaptability to demographic variations.

### Libraries

The following R libraries are required:

```r
library(dplyr)
library(ggplot2)
library(tidytext)
library(text2vec)
library(umap)
library(data.table)
library(topicmodels)
library(widyr)
library(igraph)
library(ggraph)
```

### Loading the Data

Specify the path to your folder and load each model's data:

```r
path <- "your_path"

distilgpt2 <- read.csv(file.path(path, "distilgpt2_outputs.csv"))
flan_t5_small <- read.csv(file.path(path, "flan_t5_small_outputs.csv"))
gpt_neo_125 <- read.csv(file.path(path, "gpt_neo_125m_outputs.csv"))

all_models <- bind_rows(
  distilgpt2 %>% mutate(Model = "DistilGPT-2"),
  flan_t5_small %>% mutate(Model = "Flan-T5-Small"),
  gpt_neo_125 %>% mutate(Model = "GPT-Neo-125")
)
```
### Analyzing Diagnostic Sensitivity

The script calculates the cancer diagnosis rate across prompt types (Full, Redacted Age, Redacted Gender, Fully Redacted).

```r
all_models <- all_models %>%
  mutate(Correct_Diagnosis = grepl("cancer", Model.Output, ignore.case = TRUE))
```

### Chi-Square Test of Independence
The Chi-square test evaluates the independence between prompt type and diagnostic accuracy.

```r
chi_square_results <- all_models %>%
  group_by(Model) %>%
  summarise(
    Chi_Square_Test = list(
      chisq.test(table(Correct_Diagnosis, Prompt.Type))
    )
  )
```
### Clustering Model Outputs Based on Text Embeddings

1. **Preprocess Text Data**: Converts to lowercase and removes punctuation.
2. **Load GloVe Embeddings**: Downloads GloVe embeddings and loads them into R.
3. **Generate Sentence Embeddings**: Computes average word embeddings for each output.
4. **K-means Clustering**: Groups outputs into clusters to explore model behavior.
5. **UMAP Visualization**: Reduces embeddings to 2D and visualizes clusters.

```r
glove_clustering <- ggplot(umap_df, aes(x = V1, y = V2, color = Cluster, shape = Prompt.Type)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(
    title = "Clustering of Model Outputs Based on Text Embeddings",
    x = "UMAP Dimension 1",
    y = "UMAP Dimension 2"
  ) +
  theme_minimal()
```

