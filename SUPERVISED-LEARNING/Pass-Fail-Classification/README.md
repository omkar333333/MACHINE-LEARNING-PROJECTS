# 🎓 Student Pass-Fail Prediction Using Logistic Regression

## 📖 Introduction

The Student Pass-Fail Prediction project is a Machine Learning classification project developed using Python and Scikit-Learn. The purpose of this project is to predict whether a student will pass or fail based on two important academic factors: attendance percentage and study hours per week.

This project demonstrates the complete machine learning workflow, starting from loading the dataset to making predictions for new students using a trained model.

---

## 🎯 Project Objective

The main objective of this project is to build a predictive model that can analyze student attendance and study habits and determine the likelihood of passing an examination.

The project helps in understanding how machine learning algorithms can be applied to educational data for academic performance prediction.

---

## 📂 Dataset Description

The dataset contains information about students and their academic performance.

### Features Used

**attendance_pct**

* Represents the attendance percentage of a student.
* Higher attendance generally indicates better participation in classes.

**study_hours_per_week**

* Represents the number of hours a student studies in a week.
* More study hours often contribute to better academic performance.

### Target Variable

**pass**

* 1 = Pass
* 0 = Fail

The model learns patterns from these features and predicts whether a student will pass or fail.

---

## ⚙️ Project Workflow

### 1. Data Loading

The dataset is loaded into Python using the Pandas library.

```python
Data = pd.read_csv("Pass-Fail Data.csv")
```

This converts the CSV file into a DataFrame that can be used for analysis and model training.

---

### 2. Feature Selection

The independent variables are selected as input features.

```python
X = Data[["attendance_pct", "study_hours_per_week"]]
```

The target variable is selected separately.

```python
y = Data["pass"]
```

---

### 3. Splitting the Dataset

The dataset is divided into training and testing sets using the train_test_split() function.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

* 80% data is used for training.
* 20% data is used for testing.
* random_state=42 ensures consistent results.

---

### 4. Model Training

The Logistic Regression algorithm is used to train the model.

```python
logre = LogisticRegression()
logre.fit(X_train, Y_train)
```

The model learns the relationship between attendance, study hours, and pass/fail outcomes.

---

### 5. Model Prediction

After training, predictions are made on the testing dataset.

```python
y_pred = logre.predict(X_test)
```

The model predicts whether each student in the test set will pass or fail.

---

### 6. Model Evaluation

The performance of the model is evaluated using multiple classification metrics.

#### Accuracy Score

Measures the overall correctness of the model.

```python
accuracy_score(Y_test, y_pred)
```

#### Precision Score

Measures how many predicted passes are actually passes.

```python
precision_score(Y_test, y_pred)
```

#### Recall Score

Measures how many actual passing students were correctly identified.

```python
recall_score(Y_test, y_pred)
```

#### F1 Score

Provides a balanced measure of Precision and Recall.

```python
f1_score(Y_test, y_pred)
```

These metrics help determine the effectiveness of the trained model.

---

## 🖥️ User Prediction System

The project allows users to enter their own attendance percentage and study hours.

```python
attendance = int(input("Enter Attendance Percentage: "))
studyhour = int(input("Enter Study Hours Per Week: "))
```

The entered values are converted into a format suitable for prediction.

```python
student_data = pd.DataFrame(
    [[attendance, studyhour]],
    columns=["attendance_pct", "study_hours_per_week"]
)
```

The trained model predicts the result.

```python
prediction = logre.predict(student_data)
```

The output is then converted into a user-friendly format.

```python
result = "Pass" if prediction[0] == 1 else "Fail"
```

Finally, the result is displayed to the user.

```python
print(f"When attendance is {attendance}% and study hours are {studyhour}, the student will likely {result}.")
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-Learn
* Jupyter Notebook

---

## 📊 Output

The system predicts whether a student will:

* Pass ✅
* Fail ❌

based on attendance percentage and study hours per week.

---

## 📚 Learning Outcomes

Through this project, the following machine learning concepts were implemented:

* Data Loading using Pandas
* Feature Selection
* Binary Classification
* Train-Test Split
* Logistic Regression
* Model Evaluation
* User-Based Prediction System

---

## 👨‍💻 Author

Omkar Baban Mote

4 Year Artificial Intelligence & Data Science Engineering Student

Savitribai Phule Pune University (SPPU)
