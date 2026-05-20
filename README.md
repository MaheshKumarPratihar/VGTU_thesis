# Textual Data Clustering Based on Self-Organizing Maps and Word Embeddings

This repository contains the implementation and experimental results for a Master's thesis on textual data clustering using **Self-Organizing Maps (SOMs)** and different text embedding models. The main goal of the project is to evaluate how different text representation methods influence SOM-based clustering quality across several real-world text datasets.

## Project Overview

Text data is difficult to cluster because it is usually high-dimensional, noisy, and semantically complex. Traditional clustering approaches often struggle to capture the actual meaning of text, especially when documents are short, informal, or contain overlapping classes.

This project combines text embeddings with Self-Organizing Maps to cluster textual data and visualize the structure of the embedding space. The pipeline compares traditional neural embeddings and transformer-based sentence embeddings, with optional PCA dimensionality reduction before SOM training.

## Main Objectives

- Analyze the use of Self-Organizing Maps for textual data clustering.
- Compare different text embedding models within the same SOM-based framework.
- Study the effect of PCA dimensionality reduction on clustering performance.
- Evaluate SOM results using clustering metrics and SOM-specific quality measures.
- Visualize clustering behavior using U-Matrix plots and class-count markers.

## Datasets Used

The experiments were performed on six real-world datasets from different domains and languages:

| Dataset | Language | Classes | Domain |
|---|---|---:|---|
| Hillary Clinton and Donald Trump Tweets | English | 2 | Political tweets |
| BBC News | English | 5 | News categorization |
| Twitter Sentiment Analysis | English | 3 / 4 | Sentiment analysis |
| SMS Spam | English | 2 | Spam detection |
| Lithuanian Financial News | Lithuanian | 3 | Financial sentiment |
| Lithuanian Sentiment Analysis | Lithuanian | 3 | Review sentiment |

## Embedding Models

The following embedding models were tested:

| Model | Type | Description |
|---|---|---|
| Word2Vec average | Traditional neural embedding | Averages word vectors to create document vectors |
| Doc2Vec | Traditional neural embedding | Learns document-level vectors directly |
| MiniLM | Transformer-based sentence embedding | Efficient sentence representation model |
| MPNet | Transformer-based sentence embedding | Strong semantic sentence representation model |
| DistilBERT_STS | Transformer-based sentence embedding | DistilBERT model optimized for semantic textual similarity |

## Experimental Configuration

For each dataset, the following configurations were tested:

- **PCA components:** 50, 100, 200, 300, 400, 500, and No PCA
- **Embedding models:** Word2Vec_avg, Doc2Vec, MiniLM, MPNet, DistilBERT_STS
- **SOM grid sizes:** 5x5, 8x8, 10x10
- **Sigma values:** 0.5, 1.0, 1.5, 2.0
- **Learning rates:** 0.1, 0.3, 0.5
- **Iterations:** 1000

This gives:

```text
7 PCA settings × 5 embedding models × 3 grid sizes × 4 sigma values × 3 learning rates = 1260 configurations per dataset
```

Across six datasets:

```text
1260 × 6 = 7560 total configurations
```

## Evaluation Metrics

The clustering results were evaluated using five metrics:

| Metric | Purpose | Preferred Value |
|---|---|---|
| Silhouette Score | Measures cluster separation | Higher is better |
| Adjusted Rand Index (ARI) | Compares predicted clusters with true labels | Higher is better |
| Purity | Measures class homogeneity inside clusters | Higher is better |
| Quantization Error (QE) | Measures how well the SOM represents input vectors | Lower is better |
| Topographic Error (TE) | Measures how well the SOM preserves neighborhood structure | Lower is better |

## Key Findings

- Transformer-based embeddings generally performed better than traditional neural embeddings.
- MPNet was the most consistent model, achieving the best balanced results in four out of six datasets.
- DistilBERT_STS produced the strongest result for the BBC News dataset.
- MiniLM performed best for the Hillary Donald Tweets dataset.
- BBC News was the easiest dataset to cluster because topic-based categories are more clearly separable.
- Sentiment datasets were harder to cluster because positive, negative, and neutral classes often overlap semantically.
- A 5x5 SOM grid provided the best overall balance between clustering quality and topology preservation.
- Larger grids slightly improved Purity in some cases but often increased Topographic Error.

## Project Pipeline

The general workflow is:

```text
Dataset loading
    ↓
Text cleaning and preprocessing
    ↓
Text vectorization using embedding models
    ↓
Optional PCA dimensionality reduction
    ↓
SOM training
    ↓
Metric calculation
    ↓
Result saving and visualization
```

## Installation

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn minisom gensim sentence-transformers
```

If using Google Colab, most packages are already available, but the following may need to be installed:

```bash
pip install minisom sentence-transformers gensim
```

## Example Usage

A simplified example of SOM training is shown below:

```python
from minisom import MiniSom

som = MiniSom(
    x=5,
    y=5,
    input_len=data.shape[1],
    sigma=2.0,
    learning_rate=0.1,
    random_seed=42
)

som.random_weights_init(data)
som.train_random(data, 1000)
```

For cosine distance, MiniSom can be initialized as:

```python
som = MiniSom(
    x=5,
    y=5,
    input_len=data.shape[1],
    sigma=2.0,
    learning_rate=0.1,
    activation_distance="cosine",
    random_seed=42
)
```

## Outputs

The project generates:

- CSV files containing metric results for each configuration.
- Summary tables of best configurations.
- Metric comparison graphs.
- U-Matrix visualizations.
- Class-count marker maps showing how true labels are distributed across SOM neurons.

## Thesis Context

This repository supports the Master's thesis:

**Textual Data Clusterization Based on Self-Organizing Map and Word Embeddings**

Author: **Mahesh Kumar Pratihar**  
Faculty of Fundamental Sciences  
Vilnius Gediminas Technical University  
2026

## Citation / Acknowledgement

This work was conducted as part of a Master's thesis and was also presented at the **29th Conference for Lithuanian Junior Researchers**.

## Notes

- The results may vary slightly if random seeds, preprocessing steps, or embedding model versions are changed.
- For reproducibility, fixed random seeds should be used during data sampling, PCA, and SOM initialization.
- Transformer-based embeddings require more computational resources than Word2Vec and Doc2Vec.

## License

This repository is intended for academic and research purposes. Add a license file if the project is made publicly available.
