"""
Streamlit Portfolio Web Application
Developed by: Omkar Mote (https://github.com/omkar333333)
Interactive Demonstration of Machine Learning Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Omkar Mote | ML Portfolio Showcase",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58A6FF, #BC8CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Profile & Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/178635847?v=4", width=110)
    st.markdown("### **Omkar Mote**")
    st.caption("🎓 B.E. Artificial Intelligence & Data Science")
    st.markdown("📍 *Pune, India*")
    
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat&logo=github)](https://github.com/omkar333333)
    [![Repository](https://img.shields.io/badge/Repo-MACHINE--LEARNING--PROJECTS-blue?style=flat&logo=github)](https://github.com/omkar333333/MACHINE-LEARNING-PROJECTS)
    """)
    
    st.markdown("---")
    st.markdown("### 🧭 Select ML Model")
    selected_app = st.radio(
        "Choose an Interactive System:",
        [
            "🩺 Diabetes Risk Predictor",
            "🛍️ Customer Segmentation (K-Means)",
            "🎓 Student Pass/Fail Predictor",
            "ℹ️ Portfolio & Architecture Overview"
        ]
    )
    
    st.markdown("---")
    st.markdown("#### 🛠️ Tech Stack")
    st.markdown("`Python` • `Scikit-Learn` • `Pandas` • `NumPy` • `Matplotlib` • `Streamlit`")


# ---------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_diabetes_model():
    df = pd.read_csv("SUPERVISED-LEARNING/Diabetes-Prediction/diabetes_prediction_dataset.csv")
    df_sample = df.sample(n=min(25000, len(df)), random_state=42)
    
    # Preprocessing
    df_sample['gender'] = df_sample['gender'].map({'Female': 0, 'Male': 1, 'Other': 0}).fillna(0)
    df_sample['smoking_history'] = df_sample['smoking_history'].map({
        'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5
    }).fillna(0)
    
    features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    X = df_sample[features]
    y = df_sample['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, acc, features

@st.cache_resource
def load_mall_customers():
    return pd.read_csv("UNSUPERVISED-LEARNING/Customer-Segmentation-KMeans/Mall_Customers.csv")

@st.cache_resource
def load_and_train_pass_fail_model():
    df = pd.read_csv("SUPERVISED-LEARNING/Pass-Fail-Classification/Pass-Fail Data.csv")
    features = ['attendance_pct', 'homework_pct', 'midterm_score', 'study_hours_per_week']
    X = df[features]
    y = df['pass']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    
    return model, scaler, acc, features


# ---------------------------------------------------------
# View 1: Diabetes Risk Predictor
# ---------------------------------------------------------
if selected_app == "🩺 Diabetes Risk Predictor":
    st.markdown('<div class="main-header">🩺 Diabetes Diagnostic Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Supervised Learning • Binary Classification via Random Forest</div>', unsafe_allow_html=True)
    
    with st.spinner("Initializing Random Forest Classifier..."):
        model, acc, feature_names = load_and_train_diabetes_model()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Architecture", "Random Forest (100 Trees)")
    col2.metric("Validation Accuracy", f"{acc * 100:.2f}%")
    col3.metric("Training Dataset", "25,000+ Verified Records")
    
    st.markdown("---")
    st.subheader("📋 Enter Patient Clinical Parameters")
    
    with st.form("diabetes_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Biological Sex", ["Female", "Male", "Other"])
            age = st.slider("Age (Years)", 1, 100, 35)
            hypertension = st.selectbox("Hypertension Diagnosed?", ["No (0)", "Yes (1)"])
        
        with c2:
            heart_disease = st.selectbox("Heart Disease History?", ["No (0)", "Yes (1)"])
            smoking = st.selectbox("Smoking History", ["never", "current", "former", "ever", "not current", "No Info"])
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.4, step=0.1)
            
        with c3:
            hba1c = st.slider("HbA1c Level (%)", 3.5, 9.0, 5.7, step=0.1)
            glucose = st.slider("Fasting Blood Glucose Level (mg/dL)", 70, 300, 110, step=1)
            
        submit = st.form_submit_button("⚡ Run Diagnostic Prediction")
        
    if submit:
        # Encode inputs
        gender_val = 1 if gender == "Male" else 0
        hyp_val = 1 if "Yes" in hypertension else 0
        heart_val = 1 if "Yes" in heart_disease else 0
        smoke_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
        smoke_val = smoke_map.get(smoking, 0)
        
        input_data = pd.DataFrame([[
            gender_val, age, hyp_val, heart_val, smoke_val, bmi, hba1c, glucose
        ]], columns=feature_names)
        
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        st.markdown("### 🎯 Diagnostic Result")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prediction == 1 or prob >= 0.5:
                st.error(f"⚠️ **High Probability of Diabetes: {prob * 100:.1f}%**")
                st.write("Clinical indicators suggest elevated diabetes markers. Consult a physician for confirmatory laboratory blood tests.")
            else:
                st.success(f"✅ **Low Risk / Non-Diabetic: {(1 - prob) * 100:.1f}% Confidence**")
                st.write("Parameters fall within customary non-diabetic reference ranges.")
                
        with res_col2:
            st.progress(prob, text=f"Risk Score Meter: {prob * 100:.1f}%")
            
        # Feature Importance Plot
        st.markdown("---")
        st.subheader("🔍 Model Feature Importance Breakdown")
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        importances.plot(kind='barh', ax=ax, color='#58a6ff')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.set_title("Which features most influence this model?", color='white', fontsize=12)
        st.pyplot(fig)


# ---------------------------------------------------------
# View 2: Customer Segmentation Visualizer
# ---------------------------------------------------------
elif selected_app == "🛍️ Customer Segmentation (K-Means)":
    st.markdown('<div class="main-header">🛍️ Customer Segmentation Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised Learning • K-Means Clustering on Mall Shopper Profiles</div>', unsafe_allow_html=True)
    
    df_mall = load_mall_customers()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers in Dataset", len(df_mall))
    m2.metric("Income Range", f"${df_mall['Annual Income (k$)'].min()}k - ${df_mall['Annual Income (k$)'].max()}k")
    m3.metric("Spending Score Range", f"{df_mall['Spending Score (1-100)'].min()} - {df_mall['Spending Score (1-100)'].max()}")
    
    st.markdown("---")
    col_ctrl, col_viz = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("⚙️ Clustering Controls")
        k_val = st.slider("Select Number of Clusters (k):", min_value=2, max_value=8, value=5)
        st.caption("💡 *Note: k=5 corresponds to the mathematically optimal Elbow Point on this dataset.*")
        
        X_mall = df_mall[['Annual Income (k$)', 'Spending Score (1-100)']].values
        kmeans = KMeans(n_clusters=k_val, init='k-means++', random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_mall)
        
        sil = silhouette_score(X_mall, clusters)
        st.metric("Silhouette Coefficient", f"{sil:.3f}")
        
        st.markdown("#### 🎯 Cluster Archetypes (at k=5):")
        st.markdown("""
        1. 💎 **High Income, High Spending** (Luxury Shoppers)
        2. 🎯 **Low Income, High Spending** (Carefree Spenders)
        3. 🛡️ **High Income, Low Spending** (Prudent Savers)
        4. ⚖️ **Medium Income, Medium Spending** (Middle Tier)
        5. 📉 **Low Income, Low Spending** (Budget Conscious)
        """)

    with col_viz:
        st.subheader(f"📊 Cluster Scatter Projection (k={k_val})")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        
        colors = ['#58a6ff', '#f85149', '#2ea043', '#bc8cff', '#e3b341', '#39d353', '#f0883e', '#79c0ff']
        for i in range(k_val):
            ax.scatter(
                X_mall[clusters == i, 0], 
                X_mall[clusters == i, 1], 
                s=70, 
                c=colors[i % len(colors)], 
                label=f'Cluster {i + 1}', 
                alpha=0.85
            )
            
        # Centroids
        ax.scatter(
            kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s=220, 
            c='yellow', 
            marker='X', 
            edgecolor='black', 
            label='Centroids'
        )
        
        ax.set_xlabel('Annual Income (k$)', color='white')
        ax.set_ylabel('Spending Score (1-100)', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
        ax.grid(color='#30363d', linestyle='--', alpha=0.5)
        st.pyplot(fig)


# ---------------------------------------------------------
# View 3: Student Pass/Fail Predictor
# ---------------------------------------------------------
elif selected_app == "🎓 Student Pass/Fail Predictor":
    st.markdown('<div class="main-header">🎓 Student Academic Pass/Fail Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Supervised Learning • Binary Logistic Regression with Feature Scaling</div>', unsafe_allow_html=True)
    
    model, scaler, acc, features = load_and_train_pass_fail_model()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Algorithm", "Logistic Regression")
    c2.metric("Test Accuracy", f"{acc * 100:.1f}%")
    c3.metric("Regularization", "L2 Penalty (Standard)")
    
    st.markdown("---")
    st.subheader("📚 Enter Student Engagement Metrics")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        att = st.slider("Lecture Attendance Percentage (%)", 0.0, 100.0, 82.5)
        hw = st.slider("Homework Completion Rate (%)", 0.0, 100.0, 78.0)
    with col_in2:
        mid = st.slider("Midterm Examination Score (0 - 100)", 0.0, 100.0, 68.0)
        hrs = st.slider("Independent Study Hours / Week", 0.0, 30.0, 12.0)
        
    scaled_input = scaler.transform([[att, hw, mid, hrs]])
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]
    
    st.markdown("---")
    res_box1, res_box2 = st.columns([1, 2])
    with res_box1:
        if pred == 1:
            st.success(f"🎉 **Predicted Outcome: PASS**\n\nProbability: **{prob * 100:.1f}%**")
        else:
            st.error(f"⚠️ **Predicted Outcome: RISK OF FAIL**\n\nPassing Probability: **{prob * 100:.1f}%**")
    with res_box2:
        st.progress(prob, text=f"Academic Confidence Index: {prob * 100:.1f}%")


# ---------------------------------------------------------
# View 4: Portfolio & Architecture Overview
# ---------------------------------------------------------
else:
    st.markdown('<div class="main-header">ℹ️ Machine Learning Portfolio & Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Engineering Specifications, Algorithms & Mathematical Foundations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🏗️ Repository Architecture
    This repository is engineered to showcase full-cycle Machine Learning:
    
    1. **Supervised Learning**:
       - `Diabetes-Prediction`: Binary disease classification using high-dimensional patient demographic and biochemical markers.
       - `Fraud-Detection-XGBoost`: Advanced gradient boosting algorithm applied to synthetic financial transaction fraud detection.
       - `Flight-Price-Prediction`: Continuous ticket price regression modeling utilizing temporal and categorical airline features.
       - `Student-Performance-Prediction`: Multivariate academic performance regression.
       - `Pass-Fail-Classification`: Regularized binary logistic regression.
    
    2. **Unsupervised Learning**:
       - `Customer-Segmentation-KMeans`: Unsupervised clustering utilizing Euclidean distance & centroid optimization.
       - `Dimensionality-Reduction-PCA-tSNE`: High-dimensional feature compression and manifold learning via Principal Component Analysis and t-Distributed Stochastic Neighbor Embedding.
    
    ---
    ### 👨‍💻 Connect with Omkar Mote
    - **GitHub**: [github.com/omkar333333](https://github.com/omkar333333)
    - **Location**: Balewadi, Pune, India
    """)

st.markdown("---")
st.caption("Developed with ❤️ by Omkar Mote | Powered by Streamlit & Scikit-Learn")
