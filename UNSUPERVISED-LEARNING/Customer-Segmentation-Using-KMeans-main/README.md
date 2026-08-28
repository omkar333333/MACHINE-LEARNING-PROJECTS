# Customer Segmentation Using K-Means Clustering

## Overview

This project applies the K-Means Clustering algorithm to segment mall customers based on their purchasing behavior and demographic information.

The objective is to group customers with similar characteristics, helping businesses understand customer behavior and create targeted marketing strategies.

---

## Dataset

The project uses the Mall Customers dataset containing:

- Age
- Annual Income (k$)
- Spending Score (1-100)

These features are used to identify customer groups through unsupervised machine learning.

---

## Project Workflow

1. Load and explore the dataset
2. Select important features
3. Determine the optimal number of clusters using the Elbow Method
4. Train the K-Means Clustering model
5. Assign customers to clusters
6. Visualize customer segments
7. Predict the cluster for a new customer

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-Learn

---

## Machine Learning Algorithm

### K-Means Clustering

K-Means is an unsupervised learning algorithm that groups similar data points into clusters.

The algorithm works by:

1. Selecting K centroids
2. Assigning data points to the nearest centroid
3. Updating centroid positions
4. Repeating until convergence

---

## Elbow Method

The Elbow Method is used to determine the optimal number of clusters.

It calculates the WCSS (Within Cluster Sum of Squares) for different values of K and plots them on a graph.

The point where the curve forms an elbow indicates the optimal number of clusters.

For this project:

```python
K = 5
```

was selected as the optimal number of clusters.

---

## Visualizations

### Customer Segmentation

- Age vs Annual Income
- Age vs Spending Score
- Cluster Centroids Visualization

These plots help understand customer groups and spending behavior.

---

## User Input Prediction

The project allows users to enter:

- Age
- Annual Income
- Spending Score

The trained K-Means model predicts which customer segment the new customer belongs to.

---

## Project Structure

```text
├── Mall_Customers.csv
├── Customer_Segmentation.ipynb
├── README.md
```

---

## Key Concepts Demonstrated

- Unsupervised Learning
- K-Means Clustering
- Elbow Method
- WCSS
- Customer Segmentation
- Data Visualization
- Cluster Analysis

---

## Results

The model successfully segments customers into five distinct groups based on:

- Age
- Annual Income
- Spending Score

These segments can be used for customer behavior analysis and targeted marketing strategies.

---

## Author

**Omkar Baban Mote**

AI & DS Engineering Student

GitHub: https://github.com/omkar333333