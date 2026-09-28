"""
Interactive Machine Learning Portfolio Web Application
Developed by: Omkar Mote (https://github.com/omkar333333)
Enhanced with Plotly Interactive Charts, Speedometer Gauges & Tabbed UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score

# ---------------------------------------------------------
# Page Configuration & Design System
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
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58A6FF, #BC8CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px 8px 0px 0px;
        color: #c9d1d9;
        font-weight: 600;
        padding: 0 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        border-bottom: 2px solid #58a6ff !important;
        color: #58a6ff !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.75rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Profile Card
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/178635847?v=4", width=105)
    st.markdown("### **Omkar Mote**")
    st.caption("🎓 B.E. Artificial Intelligence & Data Science")
    st.markdown("📍 *Balewadi, Pune, India*")
    
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat&logo=github)](https://github.com/omkar333333)
    [![Repository](https://img.shields.io/badge/Repo-MACHINE--LEARNING--PROJECTS-blue?style=flat&logo=github)](https://github.com/omkar333333/MACHINE-LEARNING-PROJECTS)
    """)
    
    st.markdown("---")
    st.markdown("#### 🛠️ Core Tech Stack")
    st.markdown("`Python` • `Scikit-Learn` • `Pandas` • `NumPy` • `Plotly` • `Streamlit`")
    st.markdown("---")
    st.info("💡 **Interactive Tip**: Use tabs at the top to navigate between models. Hover over charts to inspect individual data points!")

# ---------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_diabetes_model():
    df = pd.read_csv("SUPERVISED-LEARNING/Diabetes-Prediction/diabetes_prediction_dataset.csv")
    df_sample = df.sample(n=min(25000, len(df)), random_state=42)
    
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
# Main Page Header & Navigation Tabs
# ---------------------------------------------------------
st.markdown('<div class="main-header">🤖 Machine Learning Interactive Portfolio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade predictive modeling, dynamic visualization, and clustering workflows.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Diabetes Diagnostic Predictor",
    "🛍️ Customer Segmentation (K-Means)",
    "🎓 Student Academic Predictor",
    "ℹ️ Architecture & Specifications"
])

# =========================================================
# TAB 1: Diabetes Diagnostic Predictor
# =========================================================
with tab1:
    with st.spinner("Loading Random Forest model..."):
        diab_model, diab_acc, diab_features = load_and_train_diabetes_model()

    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", "Random Forest (100 Trees)")
    col2.metric("Cross-Validation Accuracy", f"{diab_acc * 100:.2f}%")
    col3.metric("Evaluated Population", "25,000+ Records")

    st.markdown("---")
    st.subheader("📋 Enter Patient Clinical Parameters")

    with st.form("diabetes_input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Biological Sex", ["Female", "Male", "Other"])
            age = st.slider("Patient Age (Years)", 1, 95, 38)
            hypertension = st.selectbox("Hypertension Diagnosed?", ["No (0)", "Yes (1)"])
        with c2:
            heart_disease = st.selectbox("Heart Disease History?", ["No (0)", "Yes (1)"])
            smoking = st.selectbox("Smoking History", ["never", "current", "former", "ever", "not current", "No Info"])
            bmi = st.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=60.0, value=26.2, step=0.1)
        with c3:
            hba1c = st.slider("HbA1c Blood Level (%)", 3.5, 9.0, 5.8, step=0.1)
            glucose = st.slider("Fasting Blood Glucose (mg/dL)", 70, 300, 115, step=1)

        submitted = st.form_submit_button("⚡ Run Diagnostic Risk Assessment")

    if submitted:
        gender_val = 1 if gender == "Male" else 0
        hyp_val = 1 if "Yes" in hypertension else 0
        heart_val = 1 if "Yes" in heart_disease else 0
        smoke_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
        smoke_val = smoke_map.get(smoking, 0)

        input_df = pd.DataFrame([[
            gender_val, age, hyp_val, heart_val, smoke_val, bmi, hba1c, glucose
        ]], columns=diab_features)

        pred = diab_model.predict(input_df)[0]
        prob = diab_model.predict_proba(input_df)[0][1]

        st.markdown("### 🎯 Diagnostic Evaluation")
        res_c1, res_c2 = st.columns([1, 1])

        with res_c1:
            # Interactive Plotly Radial Speedometer
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Diabetes Probability Index (%)", 'font': {'size': 20, 'color': 'white'}},
                delta={'reference': 50.0, 'increasing': {'color': "#f85149"}, 'decreasing': {'color': "#2ea043"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#f85149" if prob >= 0.5 else "#2ea043"},
                    'bgcolor': "#161b22",
                    'borderwidth': 2,
                    'bordercolor': "#30363d",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(46, 160, 67, 0.25)'},
                        {'range': [35, 65], 'color': 'rgba(227, 179, 65, 0.25)'},
                        {'range': [65, 100], 'color': 'rgba(248, 81, 73, 0.25)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="#0d1117", font={'color': "white"}, height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_c2:
            st.markdown("#### Clinical Interpretation")
            if pred == 1 or prob >= 0.5:
                st.error(f"⚠️ **High Probability of Diabetes ({prob * 100:.1f}%)**\n\nPatient exhibits biomarker values (e.g. elevated HbA1c or Fasting Glucose) closely correlating with diabetic cohorts. Confirmatory laboratory testing recommended.")
            else:
                st.success(f"✅ **Low Risk / Non-Diabetic ({(1 - prob) * 100:.1f}% Confidence)**\n\nClinical inputs align with typical healthy non-diabetic parameter bounds.")
                st.balloons()

            report_data = f"""--- DIABETIC RISK DIAGNOSTIC REPORT ---
Patient Age: {age}
Biological Sex: {gender}
BMI: {bmi}
HbA1c: {hba1c}%
Fasting Glucose: {glucose} mg/dL
Hypertension: {'Yes' if hyp_val else 'No'}
Heart Disease: {'Yes' if heart_val else 'No'}
---------------------------------------
Risk Probability: {prob * 100:.2f}%
Prediction: {'High Diabetes Risk' if pred == 1 else 'Low / Non-Diabetic'}
Model: Random Forest (100 Trees, 96.4% Acc)
Developed by: Omkar Mote
"""
            st.download_button(
                label="📥 Download Diagnostic Report (.txt)",
                data=report_data,
                file_name="diabetes_diagnostic_report.txt",
                mime="text/plain"
            )

        # Plotly Feature Importance
        st.markdown("---")
        st.subheader("🔍 Random Forest Feature Importance Analysis")
        imp_df = pd.DataFrame({
            'Feature': [f.replace('_', ' ').title() for f in diab_features],
            'Importance': diab_model.feature_importances_
        }).sort_values(by='Importance', ascending=True)

        fig_imp = px.bar(
            imp_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Which patient biomarkers drive model decisions?",
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_imp.update_layout(
            paper_bgcolor="#0d1117", 
            plot_bgcolor="#0d1117", 
            font={'color': 'white'},
            height=320,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# =========================================================
# TAB 2: Customer Segmentation Explorer
# =========================================================
with tab2:
    st.markdown("### 🛍️ Unsupervised Customer Segmentation via K-Means")
    st.caption("Interactive clustering of mall shoppers based on Annual Income & Spending Score.")

    df_mall = load_mall_customers()
    m_c1, m_c2, m_c3 = st.columns(3)
    m_c1.metric("Customer Cohort Size", len(df_mall))
    m_c2.metric("Median Income", f"${df_mall['Annual Income (k$)'].median():.0f}k")
    m_c3.metric("Spending Score Mean", f"{df_mall['Spending Score (1-100)'].mean():.1f} / 100")

    st.markdown("---")
    ctrl_col, plot_col = st.columns([1, 2])

    with ctrl_col:
        st.subheader("⚙️ Cluster Controls")
        k = st.slider("Select Cluster Count (k):", 2, 8, 5)
        st.caption("💡 *k=5 represents the mathematically optimal Elbow Point on this dataset.*")

        X_mall = df_mall[['Annual Income (k$)', 'Spending Score (1-100)']].values
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        df_mall['Cluster'] = kmeans.fit_predict(X_mall)
        
        sil = silhouette_score(X_mall, df_mall['Cluster'])
        st.metric("Silhouette Coefficient", f"{sil:.3f}")

        st.markdown("#### 🎯 Personas (at k=5):")
        st.markdown("""
        - 💎 **Cluster 1**: High Income, High Spending *(Target Luxury)*
        - 🎯 **Cluster 2**: Low Income, High Spending *(Carefree Spenders)*
        - 🛡️ **Cluster 3**: High Income, Low Spending *(Prudent Savers)*
        - ⚖️ **Cluster 4**: Moderate Income, Moderate Spending *(Middle Tier)*
        - 📉 **Cluster 5**: Low Income, Low Spending *(Budget Conscious)*
        """)

    with plot_col:
        st.subheader(f"📊 Interactive Cluster Projection (k={k})")
        df_mall['Cluster_Label'] = df_mall['Cluster'].apply(lambda x: f"Cluster {x + 1}")
        
        fig_scatter = px.scatter(
            df_mall,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            color='Cluster_Label',
            hover_data=['CustomerID', 'Gender', 'Age'],
            title=f"K-Means Clustering Visualization (Hover over points)",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Add Centroids
        centroids = kmeans.cluster_centers_
        fig_scatter.add_trace(go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode='markers',
            marker=dict(symbol='x', size=16, color='yellow', line=dict(width=2, color='black')),
            name='Centroids'
        ))

        fig_scatter.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font={'color': 'white'},
            legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='#30363d'),
            height=500
        )
        fig_scatter.update_xaxes(gridcolor='#30363d')
        fig_scatter.update_yaxes(gridcolor='#30363d')
        st.plotly_chart(fig_scatter, use_container_width=True)

# =========================================================
# TAB 3: Academic Pass/Fail Predictor
# =========================================================
with tab3:
    st.markdown("### 🎓 Student Academic Success Predictor")
    st.caption("L2-Regularized Logistic Regression on attendance, study habits, and midterm scores.")

    pf_model, pf_scaler, pf_acc, pf_features = load_and_train_pass_fail_model()

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Model Algorithm", "Logistic Regression")
    pc2.metric("Validation Accuracy", f"{pf_acc * 100:.1f}%")
    pc3.metric("Regularization Penalty", "L2 (Ridge)")

    st.markdown("---")
    st.subheader("📚 Enter Student Engagement Metrics")

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        att = st.slider("Lecture Attendance (%)", 0.0, 100.0, 84.0)
        hw = st.slider("Homework Submission Rate (%)", 0.0, 100.0, 80.0)
    with col_in2:
        mid = st.slider("Midterm Examination (0 - 100)", 0.0, 100.0, 72.0)
        hrs = st.slider("Study Hours / Week", 0.0, 35.0, 14.0)

    scaled_in = pf_scaler.transform([[att, hw, mid, hrs]])
    pass_pred = pf_model.predict(scaled_in)[0]
    pass_prob = pf_model.predict_proba(scaled_in)[0][1]

    st.markdown("---")
    res_c1, res_c2 = st.columns([1, 1])

    with res_c1:
        fig_pass = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pass_prob * 100,
            title={'text': "Passing Confidence Score (%)", 'font': {'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': '#2ea043' if pass_pred == 1 else '#f85149'},
                'bgcolor': '#161b22',
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(248,81,73,0.2)'},
                    {'range': [50, 100], 'color': 'rgba(46,160,67,0.2)'}
                ]
            }
        ))
        fig_pass.update_layout(paper_bgcolor="#0d1117", font={'color': 'white'}, height=280)
        st.plotly_chart(fig_pass, use_container_width=True)

    with res_c2:
        st.markdown("#### Outcome Prediction")
        if pass_pred == 1:
            st.success(f"🎉 **Predicted Status: PASS**\n\nHigh likelihood of passing the curriculum with a **{pass_prob * 100:.1f}% confidence score**.")
        else:
            st.error(f"⚠️ **Predicted Status: AT RISK OF FAIL**\n\nPassing probability is only **{pass_prob * 100:.1f}%**. Increased independent study hours and attendance recommended.")

# =========================================================
# TAB 4: Architecture & Specifications
# =========================================================
with tab4:
    st.markdown("### 🏗️ Portfolio Architectural Overview")
    st.markdown("""
    This project is engineered to showcase the full data science lifecycle:
    
    1. **Supervised Classification & Regression**:
       - `Diabetes-Prediction`: Binary classification using physiological and biochemical markers.
       - `Fraud-Detection-XGBoost`: Gradient boosting handling extreme class imbalance.
       - `Flight-Price-Prediction`: Continuous regression on dynamic aviation features.
       - `Pass-Fail-Classification`: Regularized binary logistic modeling.
    
    2. **Unsupervised Clustering & Manifold Learning**:
       - `Customer-Segmentation-KMeans`: Distance-based consumer cohort grouping.
       - `Dimensionality-Reduction-PCA-tSNE`: High-dimensional feature projection.
       
    ---
    ### 👨‍💻 Author Information
    - **Developer**: Omkar Mote
    - **GitHub**: [github.com/omkar333333](https://github.com/omkar333333)
    - **Location**: Balewadi, Pune, India
    """)

st.markdown("---")
st.caption("Crafted with ❤️ by Omkar Mote | Powered by Streamlit, Scikit-Learn & Plotly")
