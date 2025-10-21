# Textual Clusterization using SOMs and Word Embeddings
## X → Y Framework: From Raw Text to Clustered Visualization

---

## The Complete Process Flow

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

## STEP 1: Word Embeddings

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

## STEP 2: Dimensionality Reduction (Optional)

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

## STEP 3: Self-Organizing Map (SOM) Training

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

#### **Initialization Methods**
- Random initialization
- PCA-based initialization
- Linear initialization

#### **Neighborhood Functions**
- Gaussian
- Bubble (step function)
- Mexican hat
- Epanechnikov

---

## Y (OUTPUT): SOM Visualizations & Metrics

### Visualization Techniques

#### **1. U-Matrix (Unified Distance Matrix)**
- Shows distances between neighboring neurons
- Reveals cluster boundaries
- Dark areas = boundaries, Light areas = clusters

#### **2. Component Planes**
- Separate map for each input feature
- Shows distribution of feature values
- Helps interpret what each cluster represents

#### **3. Hit Histogram / Density Map**
- Shows how many documents mapped to each neuron
- Identifies popular vs. empty neurons
- Helps assess map utilization

#### **4. Cluster Map**
- Color-coded regions showing distinct clusters
- Result of post-processing (k-means, hierarchical clustering)

#### **5. Sample Distribution Map**
- Shows which documents belong to which neurons
- Allows inspection of clustering quality

### Evaluation Metrics

#### **Clustering Quality Metrics**
1. **Silhouette Score** (-1 to 1)
   - Measures how similar objects are to their own cluster
   - Higher is better

2. **Davies-Bouldin Index** (0 to ∞)
   - Ratio of within-cluster to between-cluster distances
   - Lower is better

3. **Calinski-Harabasz Index** (0 to ∞)
   - Ratio of between-cluster to within-cluster dispersion
   - Higher is better

4. **Dunn Index** (0 to ∞)
   - Ratio of minimum inter-cluster to maximum intra-cluster distance
   - Higher is better

#### **SOM-Specific Metrics**
1. **Quantization Error**
   - Average distance between data points and their BMU (Best Matching Unit)
   - Lower is better

2. **Topographic Error**
   - Proportion of data points for which 1st and 2nd BMUs are not adjacent
   - Lower is better
   - Measures topology preservation

3. **Topographic Product**
   - Measures how well the map preserves topological structure

---

## Experimental Configuration Matrix

| Exp ID | Embedding | Vector Dim | Reduction | Reduced Dim | SOM Grid | Topology | Learning Rate | Iterations | Quantization Error | Silhouette |
|--------|-----------|------------|-----------|-------------|----------|----------|---------------|------------|-------------------|------------|
| E001   | Word2Vec  | 300        | None      | 300         | 20×20    | Hex      | 0.5 → 0.01    | 10000      | TBD               | TBD        |
| E002   | Word2Vec  | 300        | PCA       | 50          | 20×20    | Hex      | 0.5 → 0.01    | 10000      | TBD               | TBD        |
| E003   | Word2Vec  | 100        | None      | 100         | 15×15    | Rect     | 0.3 → 0.01    | 15000      | TBD               | TBD        |
| E004   | GloVe     | 300        | None      | 300         | 20×20    | Hex      | 0.5 → 0.01    | 10000      | TBD               | TBD        |
| E005   | GloVe     | 200        | t-SNE     | 30          | 25×25    | Hex      | 0.7 → 0.01    | 12000      | TBD               | TBD        |
| E006   | FastText  | 300        | PCA       | 75          | 20×20    | Hex      | 0.5 → 0.01    | 10000      | TBD               | TBD        |
| E007   | FastText  | 200        | UMAP      | 30          | 15×15    | Rect     | 0.4 → 0.01    | 20000      | TBD               | TBD        |
| E008   | BERT-Base | 768        | PCA       | 100         | 25×25    | Hex      | 0.3 → 0.01    | 15000      | TBD               | TBD        |
| E009   | BERT-Base | 768        | UMAP      | 50          | 20×20    | Hex      | 0.5 → 0.01    | 10000      | TBD               | TBD        |
| E010   | BERT      | 768        | t-SNE     | 30          | 30×30    | Hex      | 0.6 → 0.01    | 12000      | TBD               | TBD        |
| E011   | DistilBERT| 768        | PCA       | 75          | 20×20    | Rect     | 0.4 → 0.01    | 15000      | TBD               | TBD        |

### Experiment Design Strategy

**Phase 1: Baseline Establishment**
- Test each embedding type with default/recommended parameters
- Establish baseline metrics

**Phase 2: Systematic Variation**
- Vary one parameter at a time while keeping others constant
- Identify sensitive parameters

**Phase 3: Combination Optimization**
- Test promising combinations
- Focus on configurations that show improvement

**Phase 4: Fine-tuning**
- Optimize the best-performing configurations
- Test edge cases and robustness

---

## Real-World Applications & Uses

### 1. **Document Organization & Management**
- **Automatic document categorization**: Large repositories (legal, medical, corporate)
- **Email classification**: Smart inbox management and auto-labeling
- **News article clustering**: Topic-based organization
- **Library systems**: Digital library organization
- **Knowledge bases**: Automatic tagging and organization

### 2. **Customer Intelligence & Sentiment Analysis**
- **Review clustering**: Group similar customer feedback
- **Product feedback analysis**: Identify common issues and feature requests
- **Support ticket routing**: Automatic assignment to relevant departments
- **Voice of customer (VoC)**: Aggregate customer opinions by theme
- **Brand perception**: Monitor brand mentions and sentiment

### 3. **Research & Academic Applications**
- **Literature review**: Organize papers by research themes
- **Research trend identification**: Discover emerging topics
- **Citation network analysis**: Understand research communities
- **Patent analysis**: Group similar patents and identify innovation gaps
- **Bibliometric analysis**: Map scientific knowledge domains

### 4. **Content Discovery & Recommendation**
- **Recommendation systems**: Suggest similar articles/products
- **Content personalization**: User-specific content curation
- **Similar item search**: Find related content
- **Content gap analysis**: Identify missing topics
- **Cross-selling opportunities**: Discover related products

### 5. **Social Media & Web Analytics**
- **Trend detection**: Identify emerging discussions
- **Community opinion mapping**: Understand public sentiment
- **Influencer identification**: Find topic experts
- **Crisis detection**: Early warning for negative trends
- **Campaign analysis**: Measure marketing effectiveness

### 6. **Business Intelligence & Market Research**
- **Competitive analysis**: Monitor competitor positioning
- **Market segmentation**: Identify customer groups
- **Product positioning**: Understand product landscape
- **Customer journey mapping**: Analyze touchpoints
- **Survey analysis**: Group open-ended responses

### 7. **Healthcare & Medical Applications**
- **Clinical notes clustering**: Group similar patient cases
- **Symptom analysis**: Identify disease patterns
- **Medical literature organization**: Evidence-based medicine
- **Drug interaction analysis**: From medical reports
- **Patient feedback analysis**: Improve care quality

### 8. **Legal & Compliance**
- **Case law clustering**: Find similar legal precedents
- **Contract analysis**: Group similar clauses
- **Compliance monitoring**: Identify regulatory issues
- **Due diligence**: Organize investigation documents
- **E-discovery**: Legal document review

### 9. **Education & E-Learning**
- **Student essay clustering**: Identify common themes
- **Course content organization**: Topic-based structuring
- **Question categorization**: Auto-tag questions by topic
- **Learning path recommendations**: Personalized content
- **Assessment analysis**: Group similar answers

### 10. **Fraud Detection & Security**
- **Fraud pattern detection**: Group suspicious activities
- **Phishing detection**: Identify similar scam messages
- **Incident clustering**: Group security events
- **Threat intelligence**: Categorize cyber threats
- **Anomaly detection**: Identify unusual patterns

---

## Thesis Contribution

### **Research Objectives**
1. **Systematic exploration** of word embedding techniques for text clustering
2. **Comparative analysis** of dimensionality reduction impact on clustering quality
3. **Optimization framework** for SOM hyperparameters in text clustering
4. **Best practice guidelines** for selecting appropriate configurations

### **Expected Outcomes**
- Identification of optimal embedding-SOM combinations for different dataset characteristics
- Understanding of trade-offs between computational cost and clustering quality
- Framework for practitioners to select appropriate techniques
- Benchmarking results across multiple evaluation metrics

### **Novelty**
- Comprehensive comparison including modern BERT embeddings with classical approaches
- Systematic evaluation of complete pipeline (embedding → reduction → SOM)
- Practical guidelines based on empirical evidence
- Open-source implementation for reproducibility

### **Impact**
- Helps practitioners make informed decisions about text clustering approaches
- Reduces trial-and-error in clustering pipeline design
- Provides baseline results for future research
- Demonstrates practical applications across multiple domains

---

## Evaluation Framework

### **Quantitative Metrics**
- Clustering quality scores (Silhouette, Davies-Bouldin, etc.)
- Computational efficiency (training time, memory usage)
- Scalability (performance with varying dataset sizes)

### **Qualitative Analysis**
- Visual inspection of cluster coherence
- Manual evaluation of cluster interpretability
- Domain expert validation (if applicable)

### **Statistical Analysis**
- Significance testing between configurations
- Confidence intervals for metrics
- Sensitivity analysis for hyperparameters

---

## Implementation Considerations

### **Software & Tools**
- **Python libraries**: Gensim (Word2Vec, FastText), scikit-learn (PCA, t-SNE), MiniSom/SOMPY, Transformers (BERT)
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Experiment tracking**: MLflow, Weights & Biases
- **Version control**: Git/GitHub

### **Computational Resources**
- GPU acceleration for BERT embeddings
- Parallel processing for multiple experiments
- Cloud computing for large-scale experiments

### **Dataset Considerations**
- Dataset size and diversity
- Text preprocessing requirements
- Language considerations
- Domain-specific vocabulary
