# Textual Clusterization using SOMs and Word Embeddings
## X → Y Framework: From Raw Text to Clustered Visualization

---

## 🔷 The Complete Process Flow

### **X (INPUT) → PROCESSING PIPELINE → Y (OUTPUT)**

```
┌─────────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
│                     │      │                      │      │                     │
│    X (INPUT)        │ ───► │  PROCESSING STEPS    │ ───► │    Y (OUTPUT)       │
│                     │      │                      │      │                     │
│  • Documents        │      │  1. Word Embeddings  │      │  • SOM Clusters     │
│  • Articles         │      │  2. Feature Reduce   │      │  • U-Matrix         │
│  • Reviews          │      │  3. SOM Training     │      │  • Visualization    │
│  • Comments         │      │                      │      │  • Metrics          │
│  • News             │      │                      │      │                     │
└─────────────────────┘      └──────────────────────┘      └─────────────────────┘
```

---

## 📊 STEP 1: Word Embeddings

### Techniques & Hyperparameters

#### **Word2Vec**
- **Architecture**: CBOW (Continuous Bag of Words) / Skip-gram
- **Hyperparameters**:
  - Vector dimensions: 50, 100, 200, 300
  - Context window size: 5, 10, 15
  - Min word frequency: 1, 5, 10
  - Training epochs: 10, 50, 100, 200
  - Learning rate: 0.01, 0.025, 0.05

#### **GloVe (Global Vectors)**
- **Architecture**: Co-occurrence matrix factorization
- **Hyperparameters**:
  - Vector dimensions: 50, 100, 200, 300
  - Context window: 5, 10, 15
  - X_max: 10, 50, 100
  - Alpha: 0.75
  - Training iterations: 50, 100, 200

#### **FastText**
- **Architecture**: Subword embeddings
- **Hyperparameters**:
  - Vector dimensions: 50, 100, 200, 300
  - Context window size: 5, 10, 15
  - Min word frequency: 1, 5, 10
  - Character n-grams: 3-6, 2-5
  - Training epochs: 10, 50, 100

#### **BERT (Bidirectional Encoder Representations from Transformers)**
- **Architecture**: Transformer-based contextual embeddings
- **Models**:
  - BERT-Base: 768 dimensions, 12 layers
  - BERT-Large: 1024 dimensions, 24 layers
  - DistilBERT: 768 dimensions (lighter version)
  - Domain-specific BERT (BioBERT, SciBERT, etc.)
- **Hyperparameters**:
  - Pooling strategy: [CLS] token, mean pooling, max pooling
  - Layer selection: Last layer, last 4 layers average, concatenate layers
  - Max sequence length: 128, 256, 512
  - Fine-tuning: Frozen vs. fine-tuned on domain data
  - Batch size: 8, 16, 32

**Output**: Dense vector representation of each document

---

## 📉 STEP 2: Dimensionality Reduction (Optional)

### Why Reduce Dimensions?
- Computational efficiency for SOM training
- Noise reduction
- Improved visualization
- Better clustering performance

### Techniques

#### **PCA (Principal Component Analysis)**
- **Hyperparameters**:
  - Number of components: 30, 50, 75, 100
  - Variance retained: 0.90, 0.95, 0.99

#### **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- **Hyperparameters**:
  - Perplexity: 5, 10, 30, 50
  - Learning rate: 10, 100, 200, 1000
  - Number of iterations: 1000, 2500, 5000
  - Target dimensions: 2, 3, 10, 30

#### **UMAP (Uniform Manifold Approximation and Projection)**
- **Hyperparameters**:
  - Number of neighbors: 5, 15, 30, 50
  - Min distance: 0.0, 0.1, 0.25, 0.5
  - Target dimensions: 2, 3, 10, 30

#### **Autoencoder**
- **Architecture**:
  - Encoder layers: [300 → 150 → 75 → 30]
  - Decoder layers: [30 → 75 → 150 → 300]
- **Hyperparameters**:
  - Latent dimension: 20, 30, 50, 100
  - Activation function: ReLU, tanh, sigmoid
  - Learning rate: 0.001, 0.0001
  - Epochs: 50, 100, 200

#### **Feature Selection Methods**
- Variance threshold
- Chi-squared test
- Mutual information
- Recursive feature elimination

---

## 🗺️ STEP 3: Self-Organizing Map (SOM) Training

### SOM Hyperparameters

#### **Grid Architecture**
- **Grid size**: 5×5, 10×10, 15×15, 20×20, 25×25, 30×30
- **Topology**: Hexagonal / Rectangular
- **Distance metric**: 
  - Euclidean
  - Manhattan
  - Cosine similarity
  - Chebyshev

#### **Training Parameters**
- **Initial learning rate (α₀)**: 0.1, 0.3, 0.5, 0.7, 0.9
- **Final learning rate (α_final)**: 0.01, 0.001
- **Learning rate decay**: Linear, exponential
- **Initial neighborhood radius (σ₀)**: 
  - Based on grid size / 2
  - Custom: 1, 3, 5, 10
- **Final neighborhood radius (σ_final)**: 0.5, 1.0
- **Radius decay function**: Linear, exponential
- **Training iterations**: 1000, 5000, 10000, 20000, 50000
- **Training mode**: Batch vs. online learning

#### **Initial
