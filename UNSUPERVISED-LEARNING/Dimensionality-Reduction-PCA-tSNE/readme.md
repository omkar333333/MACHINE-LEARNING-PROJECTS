# 📉 Dimensionality Reduction

Dimensionality Reduction is an **unsupervised learning technique** used to reduce the number of features in a dataset while preserving the most important information.

It is especially useful when a dataset contains many features and becomes difficult to visualize, process, or train on.

---

## 🧠 What is Dimensionality Reduction?

Suppose a dataset contains **100 features**.

Instead of working with all 100 features, dimensionality reduction can transform them into a smaller number of dimensions:

```text
100 Features
     ↓
Dimensionality Reduction
     ↓
2–10 Important Components
```

The goal is to reduce complexity while retaining as much useful information as possible.

---

## 🎯 Why Use Dimensionality Reduction?

* 📉 Reduce the number of features
* ⚡ Improve computational efficiency
* 📊 Visualize high-dimensional datasets
* 🧹 Remove redundant information
* 🤖 Reduce model complexity
* 🔍 Discover hidden patterns
* 📦 Make large datasets easier to work with

---

# 🔬 Techniques Covered

## 1. PCA — Principal Component Analysis

**PCA** is one of the most widely used dimensionality reduction techniques.

It transforms the original features into new features called **Principal Components**.

### Key points

* PC1 captures the maximum variance.
* PC2 captures the next highest variance.
* Principal components are uncorrelated.
* PCA is sensitive to feature scaling.

### Workflow

```text
Original Data
      ↓
Standardization
      ↓
Covariance Matrix
      ↓
Eigenvalues & Eigenvectors
      ↓
Select Components
      ↓
Reduced Data
```

### Python Example

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Load dataset
X, y = load_iris(return_X_y=True)

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2))
])

# Transform data
X_reduced = pipeline.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)
```

### Result

```text
Original shape: (150, 4)
Reduced shape: (150, 2)
```

The original Iris dataset has **4 features**, which are reduced to **2 principal components**.

---

# 2. t-SNE

**t-SNE (t-Distributed Stochastic Neighbor Embedding)** is mainly used for **visualizing high-dimensional data**.

It tries to keep similar data points close together in the lower-dimensional representation.

```text
High-Dimensional Data
        ↓
       t-SNE
        ↓
     2D / 3D
        ↓
 Visualization
```

### Common use cases

* Cluster visualization
* Image data visualization
* Feature representation analysis
* Exploring complex datasets

---

# 3. Kernel PCA

**Kernel PCA** is an extension of PCA that can handle **non-linear relationships** in data.

Regular PCA:

```text
Linear relationships
        ↓
       PCA
```

Kernel PCA:

```text
Non-linear relationships
        ↓
   Kernel Function
        ↓
    Kernel PCA
```

### Python Example

```python
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline

X, y = make_circles(
    n_samples=500,
    noise=0.05,
    factor=0.3,
    random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kpca", KernelPCA(
        n_components=2,
        kernel="rbf"
    ))
])

X_reduced = pipeline.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)
```

---

# 📊 Comparison

| Technique  | Type         | Main Purpose                        | Non-Linear Data |
| ---------- | ------------ | ----------------------------------- | --------------- |
| PCA        | Unsupervised | General dimensionality reduction    | ❌               |
| t-SNE      | Unsupervised | Visualization                       | ✅               |
| Kernel PCA | Unsupervised | Non-linear dimensionality reduction | ✅               |
| UMAP       | Unsupervised | Visualization & pattern discovery   | ✅               |
| LDA        | Supervised   | Class separation                    | ✅/❌             |

---

# 🔄 General Workflow

```text
             Dataset
                ↓
        Data Preprocessing
                ↓
        Feature Scaling
                ↓
    Dimensionality Reduction
                ↓
       ┌────────┴────────┐
       ↓        ↓        ↓
      PCA     t-SNE   Kernel PCA
       ↓        ↓        ↓
       └────────┬────────┘
                ↓
         Reduced Dataset
                ↓
       Visualization / ML
```

---

# 🧰 Technologies Used

* 🐍 Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# 📁 Project Structure

```text
DIMENSIONALITY-REDUCTION/
│
├── PCA/
│   └── PCA.ipynb
│
├── T-SNE/
│   └── TSNE.ipynb
│
├── KERNEL-PCA/
│   └── Kernel_PCA.ipynb
│
└── README.md
```

---

# 🧠 Key Concepts

### Variance

Measures how much the data varies.

PCA tries to preserve the maximum possible variance.

### Principal Component

A new feature created as a combination of the original features.

### Explained Variance

Shows how much information is captured by each principal component.

Example:

```text
PC1 → 70%
PC2 → 20%
PC3 → 7%
PC4 → 3%
```

PC1 + PC2:

```text
70% + 20% = 90%
```

Therefore, two components preserve **90% of the variance**.

---

# 🚀 Learning Goals

Through these projects, I am learning:

* Understanding dimensionality reduction
* PCA implementation
* Explained variance
* Principal components
* Feature scaling
* t-SNE visualization
* Kernel PCA
* Scikit-learn Pipelines
* Comparing dimensionality reduction techniques
* Visualizing high-dimensional data

---

## 👨‍💻 Author

**Omkar Mote**

AI & Data Science Engineering Student

Learning and building practical projects in:

**Python • Machine Learning • Data Science • Artificial Intelligence**
