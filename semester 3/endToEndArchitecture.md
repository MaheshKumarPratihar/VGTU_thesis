The end-to-end system for the research work, **Textual Data Clusterization Based on Self-organizing Map and Word Embeddings**, is a comprehensive pipeline designed to overcome the challenges of high dimensionality and poor contextual representation inherent in large textual datasets.

The methodology proposes integrating advanced, context-aware **Word Embeddings** (like BERT) with the topological preservation and clustering capabilities of **Self-Organizing Maps (SOMs)**.

The overall process is divided into four main, sequential phases:

***

## End-to-End System: Textual Data Clusterization

### Phase 1: Data Collection and Pre-processing

This initial phase focuses on preparing the unstructured text data for numerical processing, ensuring consistency and relevance.

| Step | Description | Key Techniques/Details |
| :--- | :--- | :--- |
| **1. Data Collection** | Gathering raw text data from various sources. | Uses several benchmark **English datasets** of varying sizes and domains, including Hillary Clinton and Donald Trump Tweets, Movie Reviews, AG News, BBC News, Tweets (Entity-level sentiment analysis), and SMS Spam. |
| **2. Dataset Cleaning** | Removing non-textual or irrelevant noise from the raw text. | Includes removal of HTML tags, punctuation, stopwords, hyperlinks, and irrelevant symbols. |
| **3. Text Pre-processing** | Standardizing the cleaned text data. | Includes operations like **tokenization**, **stemming** (or lemmatization), and **case normalization** (e.g., converting text to lower case for simplicity, though BERT can be case-sensitive). |

### Phase 2: Comparative Text Vectorization

The cleaned text is converted into numerical representations (vectors) through two parallel paths, allowing for comparative analysis between traditional statistical methods and modern deep learning models.

| Vectorization Path | Model/Approach | Rationale and Output |
| :--- | :--- | :--- |
| **Path A: Traditional** | **Bag-of-Words (BoW)** model | Represents text based on word frequency, disregarding context. This approach is computationally light and easy to interpret, resulting in a sparse, high-dimensional representation. |
| **Path B: Advanced (Semantic)** | **BERT** (Bidirectional Encoder Representations from Transformers) | Generates **context-aware embeddings** by processing text bidirectionally, capturing semantic nuances like polysemy and synonymy. Initial experiments used **DistilBERT**. This produces dense vector representations, helping to deal with the curse of dimensionality. |

### Phase 3: Clustering, Pattern Formation, and Optimization (SOM)

The vectorized data is processed using the Self-Organizing Map to perform dimensionality reduction, visualization, and pattern recognition.

| Step | Description | Key Rationale/Method |
| :--- | :--- | :--- |
| **1. SOM Application** | Inputting the high-dimensional vectorized data (from either BoW or BERT) into the **Self-Organizing Map**. | SOMs map high-dimensional data onto a low-dimensional grid while **preserving topological relationships**. Unlike K-means, SOMs effectively model data density, resulting in clusters closer to the **Zipf’s law distribution** characteristic of natural language. |
| **2. SOM Hyperparameter Analysis** | Examining the model's behavior under various configurations to optimize clustering quality. | Tuning parameters may include learning rate, neighborhood size, and map dimensions to fine-tune SOM performance. |
| **3. Clustering Concepts (Optional, based on WEClustering)** | If following the dense vector approach (like WEClustering), word embeddings are clustered into "concepts" using Minibatch K-means to drastically reduce the feature space. | This step creates a **Concept-Document (CD) Matrix**, which replaces the initial high-dimensional vocabulary with a smaller set of semantic concepts, making final document clustering more efficient. |
| **4. Final Document Clustering** | Applying a traditional clustering technique (if a CD matrix was created) or using the SOM output directly for document clustering. | The reduced dimensionality enables traditional algorithms like **Agglomerative clustering** (Ward linkage used) or **K-means** (K-means++ initialization used) to perform nicely on the CD matrix. |

### Phase 4: Evaluation and Future Directions

The final phase involves assessing the system's performance and defining necessary improvements.

| Step | Description | Key Metrics/Initial Findings |
| :--- | :--- | :--- |
| **1. Performance Evaluation** | Assessing the efficiency and quality of the resulting document clusters. | Metrics used include: **Adjusted Rand Index (ARI)** and **Purity** (external metrics, requiring ground truth). **Silhouette Score** and **Davies–Bouldin Index (DBI)** (internal metrics, assessing cluster density and separation). |
| **2. Documentation of Results** | Recording all findings, comparisons (e.g., BoW vs. BERT, SOM vs. K-means), and conclusions. | Initial experiments using **DistilBERT** embeddings clustered by SOM on the Hillary Clinton/Donald Trump Tweets dataset showed low ARI (0.0109) and low Silhouette Score (-0.0005), suggesting semantic separation required improvement in that specific context. |
| **3. Future Work** | Proposing directions for improving model robustness and accuracy. | Future efforts include experimenting with alternative embedding models (e.g., **RoBERTa**, **Electra**, or **GloVe**), exploring **hybrid SOM variants** (e.g., hierarchical or convolutional SOMs), and conducting systematic **hyperparameter tuning**. |
