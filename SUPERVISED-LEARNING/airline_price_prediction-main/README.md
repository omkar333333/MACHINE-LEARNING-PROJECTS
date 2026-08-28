# ✈️ Flight Ticket Price Prediction System

A Machine Learning project that predicts the price of airline tickets based on different flight details such as airline, source city, destination city, departure time, arrival time, travel class, number of stops, flight duration, and the number of days left before departure.

This project was built using **Python**, **Pandas**, **NumPy**, and **Scikit-learn**. It demonstrates the complete Machine Learning workflow, from data preprocessing to model training and prediction.

---

# 📖 Table of Contents

- Project Overview
- Problem Statement
- Objectives
- Dataset Information
- Technologies Used
- Machine Learning Workflow
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Prediction Example
- Project Structure
- Installation
- How to Run
- Results
- Future Improvements
- Learning Outcomes
- Author

---

# 📌 Project Overview

Flight ticket prices change frequently depending on several factors such as:

- Airline company
- Source city
- Destination city
- Travel class
- Number of stops
- Flight duration
- Departure time
- Arrival time
- Days left before departure

It is difficult for customers to know whether a ticket is expensive or reasonably priced.

The goal of this project is to train a Machine Learning model that can estimate the ticket price based on these flight details.

---

# ❓ Problem Statement

Airline ticket prices vary every day due to many factors.

Customers often don't know whether they should book now or wait.

This project solves this problem by predicting an estimated ticket price using historical flight data.

---

# 🎯 Objectives

The main objectives of this project are:

- Understand real-world data preprocessing.
- Learn how to handle categorical data.
- Apply One-Hot Encoding.
- Train a Linear Regression model.
- Evaluate model performance.
- Predict ticket prices for new flight information.
- Gain hands-on experience with Scikit-learn.

---

# 📂 Dataset Information

The dataset contains flight booking information.

## Input Features

| Feature | Description |
|----------|-------------|
| Airline | Name of airline company |
| Source City | City where flight starts |
| Departure Time | Morning, Afternoon, Evening, Night |
| Stops | Number of stops |
| Arrival Time | Flight arrival time |
| Destination City | Destination |
| Class | Economy or Business |
| Duration | Flight duration in hours |
| Days Left | Number of days before departure |

## Target Variable

**Price**

This is the value our Machine Learning model predicts.

---

# 🛠 Technologies Used

Programming Language

- Python

Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib (optional)
- Math

Development Environment

- Jupyter Notebook
- VS Code

---

# ⚙ Machine Learning Workflow

This project follows the complete Machine Learning pipeline.

## Step 1 — Import Libraries

The required libraries are imported.

Example:

- Pandas for data handling
- NumPy for numerical operations
- Scikit-learn for Machine Learning

---

## Step 2 — Load Dataset

The dataset is loaded into a Pandas DataFrame.

Example

```python
df = pd.read_csv("flight_price.csv")
```

Now the dataset is ready for preprocessing.

---

## Step 3 — Explore the Dataset

The dataset is checked using:

- head()
- info()
- shape
- describe()

This helps understand

- Number of rows
- Number of columns
- Data types
- Missing values

---

## Step 4 — Data Preprocessing

Before training a Machine Learning model, the data must be cleaned.

This includes

- Removing unnecessary columns
- Checking missing values
- Handling categorical data
- Preparing features

---

## Step 5 — Encoding Categorical Data

Machine Learning algorithms only understand numbers.

Columns like

- Airline
- Source City
- Destination City
- Class

contain text values.

These are converted into numerical values using

### Pandas get_dummies()

or

### OneHotEncoder

Example

Airline

```
Air India
Indigo
Vistara
```

becomes

```
AirIndia  Indigo  Vistara

1          0        0
0          1        0
0          0        1
```

Now the model can understand these values.

---

## Step 6 — Define Features and Target

The dataset is divided into

### Features (X)

These are the input columns.

Example

- Airline
- Source
- Destination
- Stops
- Duration

### Target (Y)

Price

The model learns the relationship between X and Y.

---

## Step 7 — Train Test Split

The dataset is divided into

- 80% Training Data
- 20% Testing Data

Training data teaches the model.

Testing data checks how well it performs on unseen data.

---

## Step 8 — Train the Model

The project uses

# Linear Regression

Linear Regression predicts continuous numerical values.

Since ticket price is a numerical value, Linear Regression is an appropriate algorithm.

Example

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

---

# 📊 Model Evaluation

After training, predictions are compared with actual prices.

Several metrics are used.

## Mean Absolute Error (MAE)

Shows the average prediction error.

Lower MAE means better performance.

Example

MAE = 1.67

It means the model's prediction is off by about 1.67 units on average.

---

## Mean Squared Error (MSE)

Squares every error.

Large mistakes receive higher penalties.

Lower is better.

---

## Root Mean Squared Error (RMSE)

Square root of MSE.

Easy to understand because it has the same unit as the target.

---

## R² Score

Measures how well the model explains the data.

Range

0 → Poor

1 → Perfect

Higher is better.

---

# 🔮 Prediction

After training, users can enter new flight details.

Example

```
Airline : Air India

Source : Delhi

Destination : Mumbai

Stops : 1

Class : Economy

Duration : 2.5 Hours

Days Left : 15
```

The model predicts

```
Predicted Ticket Price

₹6,850
```

(The value above is only an example.)

---

# 📁 Project Structure

```
Flight-Ticket-Price-Prediction/

│
├── Flight Ticket Price Prediction System.ipynb
├── dataset.csv
├── README.md
├── requirements.txt
└── images
```

---

# ▶ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Flight-Ticket-Price-Prediction.git
```

Move into the folder

```bash
cd Flight-Ticket-Price-Prediction
```

Install required libraries

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
```

Or

```bash
pip install -r requirements.txt
```

---

# ▶ How to Run

Open

```
Flight Ticket Price Prediction System.ipynb
```

Run every notebook cell from top to bottom.

The notebook will

- Load the dataset
- Preprocess the data
- Train the model
- Evaluate performance
- Predict ticket prices

---

# 📈 Results

The Linear Regression model successfully learned the relationship between flight information and ticket prices.

The project demonstrates the complete Machine Learning workflow including

- Data preprocessing
- Feature encoding
- Model training
- Prediction
- Performance evaluation

---

# 🚀 Future Improvements

This project can be improved by

- Random Forest Regressor
- Decision Tree Regressor
- XGBoost
- Hyperparameter Tuning
- Feature Scaling
- Streamlit Web App
- Flask Deployment
- Model Comparison
- Real-time Flight API Integration

---

# 📚 Learning Outcomes

By completing this project, I learned

✅ Data Cleaning

✅ Data Preprocessing

✅ Feature Engineering

✅ One-Hot Encoding

✅ Pandas

✅ NumPy

✅ Train-Test Split

✅ Linear Regression

✅ Model Evaluation

✅ Machine Learning Pipeline

✅ Predictive Analytics

---

# 👨‍💻 Author

**Omkar Baban Mote**

Third Year B.E. Artificial Intelligence & Data Science

Savitribai Phule Pune University (SPPU)

### Skills

- Python
- Machine Learning
- Pandas
- NumPy
- Scikit-learn
- SQL
- Data Analysis

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It motivates me to build more Machine Learning projects and continue improving my skills.
