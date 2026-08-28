# 📊 Student Final Grade Prediction Using Linear Regression

A beginner-friendly Machine Learning project that predicts a student's **Final Grade** based on **Study Hours** using the **Linear Regression** algorithm.

---

## 🚀 Project Overview

The purpose of this project is to understand the complete Machine Learning workflow, including:

* Data Loading
* Data Cleaning
* Feature Selection
* Model Training
* Prediction
* Model Evaluation
* Data Visualization

The model learns the relationship between study hours and student grades and predicts future grades based on study time.

---

## 📂 Dataset

The dataset contains student academic performance information.

### Input Feature

| Feature     | Description                       |
| ----------- | --------------------------------- |
| Study Hours | Number of hours a student studies |

### Target Variable

| Feature    | Description           |
| ---------- | --------------------- |
| FinalGrade | Student's final grade |

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

---

## ⚙️ Installation

Install required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## ▶️ Run The Project

```bash
python main.py
```

or

```bash
jupyter notebook ml_project2.ipynb
```

---

## 🧠 Machine Learning Workflow

### Step 1: Load Dataset

```python
data = pd.read_csv("student_perf.csv")
```

### Step 2: Handle Missing Values

```python
data = data.dropna()
```

### Step 3: Select Features

```python
x = data[['Study Hours']]
y = data['FinalGrade']
```

### Step 4: Train Linear Regression Model

```python
model = LinearRegression()
model.fit(x, y)
```

### Step 5: Generate Predictions

```python
predicted_score = model.predict(x)
```

### Step 6: Evaluate Model

The following metrics are used:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

---

## 📊 Model Performance

| Metric   | Value  |
| -------- | ------ |
| MAE      | 8.14   |
| MSE      | 92.18  |
| RMSE     | 9.60   |
| R² Score | 0.0002 |

### Interpretation

* The model has an average error of about **8 marks**.
* The R² score is very low, indicating that **Study Hours alone are not sufficient to accurately predict Final Grades**.
* Student performance depends on multiple factors such as attendance, assignments, participation, and exams.

---

## 📈 Visualizations

### Histogram

Displays the distribution of student grades.

```python
plt.hist(data['FinalGrade'])
```

### Scatter Plot

Shows the relationship between Study Hours and Final Grade.

```python
plt.scatter(x, y)
```

### Regression Line

Displays the line learned by the Linear Regression model.

```python
plt.plot(x, predicted_score)
```

---

## 🎯 Example Prediction

```python
new_hours = 9
predicted_new_score = model.predict([[new_hours]])
```

Output:

```text
Predicted final score for 9 study hours per week.
```

---

## 📁 Project Structure

```text
Student-Final-Grade-Prediction/
│
├── student_perf.csv
├── ml_project2.ipynb
├── README.md
└── requirements.txt
```

---

## 🔮 Future Improvements

* Multiple Linear Regression
* Train-Test Split
* Decision Tree Regression
* Random Forest Regression
* Feature Engineering
* Flask Web Application
* Streamlit Dashboard
* Model Comparison

---

## 📚 Learning Outcomes

Through this project, I learned:

* Data Preprocessing
* Handling Missing Values
* Linear Regression
* Model Training
* Model Evaluation
* Data Visualization
* Making Predictions Using Machine Learning

---

## 📄 License

This project is licensed under the MIT License.
