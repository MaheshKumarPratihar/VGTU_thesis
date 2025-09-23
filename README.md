This document provides a brief explanation of the stages and methodology for the thesis titled **Textual Data Clusterization Based on Self-organizing Map and Word Embeddings**.

The overall aim of the research is to **improve text clustering** by addressing high dimensionality and enhancing contextual meaning to improve text quality. The novelty lies in the innovative use of Self-Organizing Maps (SOMs) combined with advanced word embedding techniques to tackle complex challenges in clustering textual data.

***

# Textual Data Clusterization Based on Self-organizing Map and Word Embeddings

## Thesis Overview

This thesis investigates the inability of traditional clustering techniques to manage the **high dimensionality** of textual data and linguistic intricacies like polysemy and synonymy. The proposed solution integrates the pattern-recognition capabilities of **Self-Organizing Maps (SOMs)** with the advanced semantic understanding provided by **BERT** (Bidirectional Encoder Representations from Transformers).

SOMs are utilized because they map high-dimensional data onto a two-dimensional grid while preserving the data's structural and topological relationships. This approach aims to create more efficient and meaningful clusters of educational texts, contributing to fields such as adaptive learning and content curation.

## Stages of the Thesis

The thesis is organized into distinct stages, covering analysis, methodology proposal, implementation, and evaluation.

### Stage 1: Related Work Analysis

The first task is **to analyze related works on unstructured data classification in the context of clustering techniques and embedding methods**.

*   **Review of Traditional Methods:** Examination of initial statistical techniques like Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF). These models are noted for shortcomings such as the **loss of contextual meaning** and the **high dimensionality** that results in sparse matrices.
*   **Evaluation of Word Embeddings:** Reviewing the evolution from context-independent models (Word2Vec, GloVe), which assign a fixed vector to a word regardless of context, to context-aware models like **BERT**. BERT stands out for capturing the semantic nuances of words based on their surrounding context.
*   **Study of SOMs:** Analysis of SOMs as a type of artificial neural network designed for clustering and visualization. SOMs effectively model data density and preserve topological properties, leading to cluster formation closer to the **Zipf’s law distribution** characteristic of natural language.

### Stage 2: Proposed Approach and Implementation

This phase involves **proposing an approach for classifying textual data using SOMs and word embeddings** and **implementing the proposed approach as a prototype and evaluating its efficiency**. The approach is a pipeline leveraging different vectorization techniques.

1.  **Data Collection and Cleaning:** Raw text data is collected. This is followed by **Dataset Cleaning** (removing noise/symbols like HTML tags, punctuation, and stopwords) and **Text Pre-processing** (tokenization, stemming/lemmatization, case normalization).
2.  **Text Vectorization:** The clean text is converted into numerical representations, comparing two distinct paths:
    *   **Bag-of-Words (BoW):** A straightforward, computationally light method that is easy to interpret.
    *   **BERT Model:** A context-aware model that produces dense embeddings and significantly enhances performance on complex NLP tasks.
3.  **Clustering with SOM:** The vectorized data (from BoW or BERT) is input into the **Self-Organizing Map (SOM)**. This step includes a dedicated **SOM hyperparameter analysis** to fine-tune performance under various configurations.

The methodology is tested on a total of six benchmark real-world English datasets of different sizes and domains, including:
*   Hillary Clinton and Donald Trump Tweets
*   Movie Review (Polarity\_dataset\_v2.0)
*   AG (News Articles)
*   BBC News
*   Tweets (Entity-level sentiment analysis)
*   SMS Spam

### Stage 3: Experimental Study and Evaluation (Initial Results)

The prototype is evaluated for its efficiency and clustering quality, benchmarking the SOM-based clustering against traditional clustering techniques.

*   **Metrics:** Clustering quality is assessed using external metrics that require ground truth, such as **Adjusted Rand Index (ARI)** and **Purity**, and internal metrics, such as the **Silhouette Score**.
*   **Preliminary Findings:** Initial experiments using **DistilBERT** embeddings clustered by SOM on the Hillary Clinton and Donald Trump Tweets dataset yielded the following metrics:
    *   **Cluster Purity:** 0.7875.
    *   **Adjusted Rand Index (ARI):** 0.0109 (close to random assignment).
    *   **Silhouette Score:** -0.0005 (points are not well separated).
*   These results suggest that while SOMs provide visually interpretable topological mappings, the combined model struggles with fine-grained semantic distinctions in short, informal texts like tweets, necessitating further improvement.

### Stage 4: Conclusion and Future Work

The final stage summarizes the findings, concluding that combining SOMs with modern embeddings like BERT provides a **powerful and scalable approach** to clustering textual data, capturing deeper meaning compared to classic techniques.

Future research is planned to enhance performance by:

*   Experimenting with **alternative embedding models** such as **RoBERTa**, **Electra**, or **GloVe**.
*   Exploring **hybrid variants of SOMs**, including hierarchical or convolutional SOMs.
*   Conducting systematic **hyperparameter tuning** to optimize SOM performance.
*   Testing the proposed technique on **more large-sized datasets**.
