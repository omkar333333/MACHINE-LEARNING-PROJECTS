# 🤖 Machine Learning with Scikit-Learn

A beginner-friendly Machine Learning repository covering Data Preprocessing, Feature Engineering, Model Training, Model Evaluation, and Real-World ML Projects using Python and Scikit-Learn.

---

# 📚 Topics Covered

## 1. Data Preprocessing

### Missing Data Handling
- Detect Missing Values
- Remove Missing Values
- Fill Missing Values
- Mean Imputation
- Median Imputation
- Mode Imputation

### Encoding Categorical Data
- Label Encoding
- One-Hot Encoding

### Feature Scaling
- StandardScaler
- MinMaxScaler

### Train-Test Split
- Training Data
- Testing Data
- Random State

---

# 2. Supervised Machine Learning

## Regression

### Linear Regression

Used to predict numerical values.

### Examples
- Student Score Prediction
- Salary Prediction
- House Price Prediction

### Formula

```text
y = mx + c
```

---

## Classification

### Logistic Regression

Used to predict Yes/No outcomes.

### Examples
- Pass / Fail
- Spam / Not Spam
- Disease / No Disease

---

### K-Nearest Neighbors (KNN)

Predicts based on nearest data points.

### Examples
- Fruit Classification
- Recommendation System

---

### Decision Tree

Predicts using a tree of decisions.

### Examples
- Loan Approval
- Customer Classification
- Medical Diagnosis

---

# 3. Model Problems

## Overfitting

### Definition

A model learns the training data too well and performs poorly on unseen data.

### Characteristics

- High Training Accuracy
- Low Testing Accuracy
- Poor Generalization

### Example

A student memorizes all previous exam answers but cannot solve new questions.

---

## Underfitting

### Definition

A model fails to learn important patterns from training data.

### Characteristics

- Low Training Accuracy
- Low Testing Accuracy
- Poor Performance

### Example

A student studies only one chapter and fails most questions.

---

# 4. Model Evaluation Metrics

## Classification Metrics

### Accuracy

Measures overall correct predictions.

### Formula

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

---

### Precision

Measures how many predicted positives are actually positive.

### Formula

```text
Precision = TP / (TP + FP)
```

---

### Recall

Measures how many actual positives are correctly identified.

### Formula

```text
Recall = TP / (TP + FN)
```

---

### F1-Score

Balance between Precision and Recall.

### Formula

```text
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

---

# Confusion Matrix

| Actual / Predicted | Positive | Negative |
|-------------------|----------|----------|
| Positive | TP | FN |
| Negative | FP | TN |

### Terms

- TP = True Positive
- TN = True Negative
- FP = False Positive
- FN = False Negative

---

# Regression Metrics

## MAE (Mean Absolute Error)

Measures average absolute error.

### Formula

```text
MAE = Σ |Actual - Predicted| / n
```

---

## MSE (Mean Squared Error)

Measures average squared error.

### Formula

```text
MSE = Σ (Actual - Predicted)² / n
```

---

## RMSE (Root Mean Squared Error)

Square root of MSE.

### Formula

```text
RMSE = √MSE
```

---

# 🛠 Technologies Used

- Python
- Pandas
- Polars
- NumPy
- Scikit-Learn

---

# 🚀 Mini Projects

## Student Score Prediction
Predict exam scores using study hours.

## Salary Prediction
Predict salary using years of experience.

## House Price Prediction
Predict house prices using area and features.

## Pass/Fail Prediction
Predict student performance.

## Fruit Classification
Classify Apple or Orange using KNN.

## Loan Approval Prediction
Predict loan approval using Decision Trees.

---

# 💻 Installation

```bash
pip install pandas numpy scikit-learn polars matplotlib
```

---

# ▶️ Run

```bash
python main.py
```

---

# 🎯 Interview Questions

### What is Machine Learning?

Machine Learning is a branch of AI that enables computers to learn from data without explicit programming.

### What is Supervised Learning?

Learning using labeled data.

### What is Regression?

Used to predict numerical values.

### What is Classification?

Used to predict categories or classes.

### What is Overfitting?

Model learns too much and performs poorly on new data.

### What is Underfitting?

Model learns too little and performs poorly.

### What is Accuracy?

Percentage of correct predictions.

### What is Precision?

Predicted positive accuracy.

### What is Recall?

Actual positive detection rate.

### What is F1-Score?

Balance between Precision and Recall.

---

# 👨‍💻 Author

**Omkar Baban Mote**  
Third Year AI & Data Science Engineering Student  
Savitribai Phule Pune University (SPPU)
