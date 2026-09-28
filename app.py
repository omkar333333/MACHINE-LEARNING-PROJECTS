"""
Interactive Machine Learning Portfolio Web Application
Developed by: Omkar Mote (https://github.com/omkar333333)
Features:
1. Clinical Diagnostics & "What-If" Sensitivity Simulator
2. Model Benchmark Arena (Side-by-Side Algorithm Battle)
3. Flight Ticket Fare Forecaster (Regression)
4. Customer Segmentation Explorer (K-Means Clustering)
5. Interactive Exploratory Data Analysis (EDA) Lab
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, silhouette_score, r2_score, mean_squared_error
)

# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Omkar Mote | ML Portfolio & Lab",
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
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8b949e;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px 8px 0px 0px;
        color: #c9d1d9;
        font-weight: 600;
        padding: 0 18px;
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
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Profile & Details
# ---------------------------------------------------------
with st.sidebar:
    st.image("assets/avatar.jpg", width=105)
    st.markdown("### **Omkar Mote**")
    st.caption("🎓 B.E. Artificial Intelligence & Data Science")
    st.markdown("📍 *Balewadi, Pune, India*")
    
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat&logo=github)](https://github.com/omkar333333)
    [![Repo](https://img.shields.io/badge/Repo-MACHINE--LEARNING--PROJECTS-blue?style=flat&logo=github)](https://github.com/omkar333333/MACHINE-LEARNING-PROJECTS)
    """)
    
    st.markdown("---")
    st.markdown("#### ⚡ Portfolio Features")
    st.markdown("""
    - 🩺 **Clinical Diagnostics & What-If**
    - ⚔️ **Model Benchmark Arena**
    - ✈️ **Flight Price Forecaster**
    - 🛍️ **Customer Segmentation**
    - 📊 **Exploratory Data Analysis Lab**
    """)
    st.markdown("---")
    st.caption("Built with Streamlit, Scikit-Learn, Plotly & Pandas")

# ---------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_diabetes_data():
    df = pd.read_csv("SUPERVISED-LEARNING/Diabetes-Prediction/diabetes_prediction_dataset.csv")
    df_sample = df.sample(n=min(20000, len(df)), random_state=42).copy()
    df_sample['gender'] = df_sample['gender'].map({'Female': 0, 'Male': 1, 'Other': 0}).fillna(0)
    smoke_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
    df_sample['smoking_history'] = df_sample['smoking_history'].map(smoke_map).fillna(0)
    features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    return df_sample, features

@st.cache_resource
def train_diabetes_rf():
    df, features = load_diabetes_data()
    X = df[features]
    y = df['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, features

@st.cache_resource
def load_mall_customers():
    return pd.read_csv("UNSUPERVISED-LEARNING/Customer-Segmentation-KMeans/Mall_Customers.csv")

@st.cache_resource
def load_flight_model():
    df = pd.read_csv("SUPERVISED-LEARNING/Flight-Price-Prediction/airlines_flights_data.csv", nrows=15000)
    features_cat = ['airline', 'source_city', 'destination_city', 'stops', 'class']
    features_num = ['duration', 'days_left']
    
    df_encoded = pd.get_dummies(df[features_cat + features_num], drop_first=True)
    X = df_encoded
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=40, max_depth=10, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    r2 = r2_score(y_test, reg.predict(X_test))
    
    return reg, r2, list(X.columns), df

# ---------------------------------------------------------
# Header & Navigation Tabs
# ---------------------------------------------------------
st.markdown('<div class="main-header">🤖 Machine Learning Intelligence Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade workflows covering Classification, Regression, Clustering, Benchmarking & Sensitivity Modeling.</div>', unsafe_allow_html=True)

tab_diab, tab_bench, tab_flight, tab_mall, tab_eda, tab_info = st.tabs([
    "🩺 Clinical & What-If",
    "⚔️ Benchmark Arena",
    "✈️ Flight Forecaster",
    "🛍️ Customer Segmentation",
    "📊 EDA Lab",
    "ℹ️ Architecture"
])

# =========================================================
# TAB 1: Clinical Diagnostics & What-If Sensitivity
# =========================================================
with tab_diab:
    st.markdown("### 🩺 Diabetes Diagnostics, Batch Inference & Sensitivity Analysis")
    st.caption("Random Forest binary classification with real-time what-if parameter simulation.")
    
    rf_model, rf_acc, feat_names = train_diabetes_rf()
    
    c_sub1, c_sub2 = st.tabs(["⚡ Patient Diagnostic & What-If", "📂 Batch CSV Scoring"])
    
    with c_sub1:
        with st.form("clinical_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Biological Sex", ["Female", "Male", "Other"])
                age = st.slider("Age (Years)", 1, 95, 45)
                hypertension = st.selectbox("Hypertension?", ["No (0)", "Yes (1)"])
            with c2:
                heart_disease = st.selectbox("Heart Disease History?", ["No (0)", "Yes (1)"])
                smoking = st.selectbox("Smoking History", ["never", "current", "former", "ever", "not current", "No Info"])
                bmi = st.number_input("Body Mass Index (BMI)", 12.0, 55.0, 27.5, step=0.1)
            with c3:
                hba1c = st.slider("HbA1c Blood Level (%)", 3.5, 9.0, 6.2, step=0.1)
                glucose = st.slider("Fasting Blood Glucose (mg/dL)", 70, 300, 130, step=1)
                
            run_btn = st.form_submit_button("⚡ Run Diagnostic Evaluation")
            
        gender_val = 1 if gender == "Male" else 0
        hyp_val = 1 if "Yes" in hypertension else 0
        heart_val = 1 if "Yes" in heart_disease else 0
        smoke_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
        smoke_val = smoke_map.get(smoking, 0)
        
        base_df = pd.DataFrame([[gender_val, age, hyp_val, heart_val, smoke_val, bmi, hba1c, glucose]], columns=feat_names)
        base_prob = rf_model.predict_proba(base_df)[0][1]
        
        st.markdown("---")
        st.subheader("🎯 Primary Diagnostic Outcome")
        
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            fig_gauge1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=base_prob * 100,
                title={'text': "Baseline Risk Score (%)", 'font': {'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': '#f85149' if base_prob >= 0.5 else '#2ea043'},
                    'bgcolor': '#161b22',
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(46, 160, 67, 0.25)'},
                        {'range': [35, 65], 'color': 'rgba(227, 179, 65, 0.25)'},
                        {'range': [65, 100], 'color': 'rgba(248, 81, 73, 0.25)'}
                    ]
                }
            ))
            fig_gauge1.update_layout(paper_bgcolor="#0d1117", font={'color': 'white'}, height=260)
            st.plotly_chart(fig_gauge1, use_container_width=True)
            
        with res_col2:
            st.markdown("#### Clinical Summary")
            if base_prob >= 0.5:
                st.error(f"⚠️ **High Risk of Diabetes: {base_prob * 100:.1f}% Probability**")
                st.write("Elevated biochemical markers detected. Lifestyle and pharmacological intervention recommended.")
            else:
                st.success(f"✅ **Low Risk / Non-Diabetic: {(1 - base_prob) * 100:.1f}% Confidence**")
                st.write("Diagnostic parameters lie within standard healthy physiological thresholds.")
                
        # "What-If" Sensitivity Simulator
        st.markdown("---")
        st.subheader("🎛️ 'What-If' Lifestyle Sensitivity Simulator")
        st.caption("Adjust hypothetical changes to observe real-time risk reduction or escalation:")
        
        wi1, wi2 = st.columns(2)
        with wi1:
            delta_glucose = st.slider("Hypothetical Glucose Adjustment (mg/dL):", -50, 50, -25)
        with wi2:
            delta_bmi = st.slider("Hypothetical BMI Change:", -8.0, 8.0, -3.0, step=0.5)
            
        hypo_glucose = max(70, min(300, glucose + delta_glucose))
        hypo_bmi = max(12.0, min(55.0, bmi + delta_bmi))
        
        hypo_df = pd.DataFrame([[gender_val, age, hyp_val, heart_val, smoke_val, hypo_bmi, hba1c, hypo_glucose]], columns=feat_names)
        hypo_prob = rf_model.predict_proba(hypo_df)[0][1]
        diff_prob = (hypo_prob - base_prob) * 100
        
        wi_col1, wi_col2 = st.columns([1, 1])
        with wi_col1:
            fig_gauge2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=hypo_prob * 100,
                delta={'reference': base_prob * 100, 'increasing': {'color': "#f85149"}, 'decreasing': {'color': "#2ea043"}},
                title={'text': "Simulated Post-Intervention Risk (%)", 'font': {'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': '#f85149' if hypo_prob >= 0.5 else '#2ea043'},
                    'bgcolor': '#161b22',
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(46, 160, 67, 0.25)'},
                        {'range': [35, 65], 'color': 'rgba(227, 179, 65, 0.25)'},
                        {'range': [65, 100], 'color': 'rgba(248, 81, 73, 0.25)'}
                    ]
                }
            ))
            fig_gauge2.update_layout(paper_bgcolor="#0d1117", font={'color': 'white'}, height=260)
            st.plotly_chart(fig_gauge2, use_container_width=True)
            
        with wi_col2:
            st.markdown("#### Scenario Impact Analysis")
            if diff_prob < 0:
                st.success(f"📉 **Risk Reduced by {abs(diff_prob):.1f}%!**\n\nLowering fasting glucose to {hypo_glucose} mg/dL and BMI to {hypo_bmi:.1f} yields a meaningful reduction in diabetic vulnerability.")
            else:
                st.warning(f"📈 **Risk Increased by {diff_prob:.1f}%!**\n\nElevating biomarker parameters increases calculated health risk.")

    with c_sub2:
        st.subheader("📂 Batch Patient Scoring via CSV Upload")
        st.write("Upload a `.csv` with patient data to evaluate multiple cohorts in bulk.")
        
        sample_template = pd.DataFrame({
            'gender': ['Male', 'Female', 'Female', 'Male'],
            'age': [52, 28, 64, 41],
            'hypertension': [0, 0, 1, 0],
            'heart_disease': [0, 0, 1, 0],
            'smoking_history': ['never', 'current', 'former', 'never'],
            'bmi': [28.4, 21.3, 33.1, 25.0],
            'HbA1c_level': [6.5, 4.8, 7.2, 5.5],
            'blood_glucose_level': [155, 90, 180, 110]
        })
        
        st.download_button(
            "📥 Download Sample CSV Template",
            data=sample_template.to_csv(index=False),
            file_name="sample_patients_template.csv",
            mime="text/csv"
        )
        
        uploaded_csv = st.file_uploader("Upload CSV file:", type=["csv"])
        
        if uploaded_csv is not None or st.button("🚀 Score Sample Cohort Dataset"):
            df_batch = pd.read_csv(uploaded_csv) if uploaded_csv is not None else sample_template.copy()
            
            # Preprocess batch
            df_b_encoded = df_batch.copy()
            df_b_encoded['gender'] = df_b_encoded['gender'].map({'Female': 0, 'Male': 1, 'Other': 0}).fillna(0)
            smoke_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
            df_b_encoded['smoking_history'] = df_b_encoded['smoking_history'].map(smoke_map).fillna(0)
            
            probs = rf_model.predict_proba(df_b_encoded[feat_names])[:, 1]
            df_batch['Diabetes_Probability_%'] = np.round(probs * 100, 2)
            df_batch['Predicted_Diagnosis'] = np.where(probs >= 0.5, 'Diabetic Risk', 'Non-Diabetic')
            
            b_c1, b_c2 = st.columns([2, 1])
            with b_c1:
                st.dataframe(df_batch, use_container_width=True)
            with b_c2:
                fig_pie = px.pie(
                    df_batch, names='Predicted_Diagnosis', 
                    title="Cohort Risk Distribution",
                    color_discrete_sequence=['#2ea043', '#f85149']
                )
                fig_pie.update_layout(paper_bgcolor="#0d1117", font={'color': 'white'}, height=280)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            st.download_button(
                "📥 Download Scored Batch CSV",
                data=df_batch.to_csv(index=False),
                file_name="scored_diabetes_predictions.csv",
                mime="text/csv"
            )

# =========================================================
# TAB 2: Model Benchmark Arena (Side-by-Side Comparison)
# =========================================================
with tab_bench:
    st.markdown("### ⚔️ Model Benchmark & Comparison Arena")
    st.caption("Train and compare 4 algorithms side-by-side on identical test folds.")
    
    df_raw, feat_b = load_diabetes_data()
    df_sub = df_raw.sample(n=6000, random_state=42)
    X_b = df_sub[feat_b]
    y_b = df_sub['diabetes']
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size=0.25, random_state=42, stratify=y_b)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    
    results = []
    cms = {}
    
    with st.spinner("Benchmarking algorithms in parallel..."):
        for name, clf in models.items():
            t0 = time.time()
            clf.fit(X_train_b, y_train_b)
            train_time_ms = (time.time() - t0) * 1000
            
            y_pred = clf.predict(X_test_b)
            acc = accuracy_score(y_test_b, y_pred)
            prec = precision_score(y_test_b, y_pred, zero_division=0)
            rec = recall_score(y_test_b, y_pred, zero_division=0)
            f1 = f1_score(y_test_b, y_pred, zero_division=0)
            
            results.append({
                "Algorithm": name,
                "Accuracy (%)": round(acc * 100, 2),
                "Precision (%)": round(prec * 100, 2),
                "Recall (%)": round(rec * 100, 2),
                "F1-Score (%)": round(f1 * 100, 2),
                "Train Time (ms)": round(train_time_ms, 1)
            })
            cms[name] = confusion_matrix(y_test_b, y_pred)
            
    res_df = pd.DataFrame(results)
    
    # Leaderboard Cards
    best_acc = res_df.loc[res_df['Accuracy (%)'].idxmax()]
    best_f1 = res_df.loc[res_df['F1-Score (%)'].idxmax()]
    fastest = res_df.loc[res_df['Train Time (ms)'].idxmin()]
    
    l1, l2, l3 = st.columns(3)
    l1.metric("🏆 Top Accuracy", f"{best_acc['Algorithm']} ({best_acc['Accuracy (%)']}%)")
    l2.metric("🎯 Top F1-Score", f"{best_f1['Algorithm']} ({best_f1['F1-Score (%)']}%)")
    l3.metric("⚡ Fastest Training", f"{fastest['Algorithm']} ({fastest['Train Time (ms)']} ms)")
    
    st.markdown("---")
    st.subheader("📊 Comparative Performance Metrics")
    
    # Melt for Plotly grouped bar
    melted = res_df.melt(id_vars=["Algorithm"], value_vars=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"], var_name="Metric", value_name="Score (%)")
    fig_bar = px.bar(
        melted, x="Algorithm", y="Score (%)", color="Metric", barmode="group",
        title="Side-by-Side Algorithm Comparison",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_bar.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font={'color': 'white'}, height=360)
    fig_bar.update_yaxes(range=[0, 105], gridcolor='#30363d')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Confusion Matrix side-by-side
    st.markdown("---")
    st.subheader("🔍 Confusion Matrix Comparison (True vs Predicted)")
    
    cm_cols = st.columns(4)
    for idx, (m_name, cm_val) in enumerate(cms.items()):
        with cm_cols[idx]:
            st.markdown(f"**{m_name}**")
            fig_cm = px.imshow(
                cm_val, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Negative', 'Positive'], y=['Negative', 'Positive'],
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(paper_bgcolor="#0d1117", font={'color': 'white'}, height=260, coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

# =========================================================
# TAB 3: Flight Ticket Price Forecaster
# =========================================================
with tab_flight:
    st.markdown("### ✈️ Aviation Ticket Fare Forecaster")
    st.caption("Multivariate Random Forest Regression predicting domestic flight prices in India (₹ INR).")
    
    reg_flight, flight_r2, flight_cols, df_flight_raw = load_flight_model()
    
    fl_m1, fl_m2, fl_m3 = st.columns(3)
    fl_m1.metric("Regression Algorithm", "Random Forest Regressor")
    fl_m2.metric("Coefficient of Determination (R²)", f"{flight_r2:.3f}")
    fl_m3.metric("Training Sample", "15,000 Verified Flight Logs")
    
    st.markdown("---")
    st.subheader("🛫 Select Flight Routing & Class")
    
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        airline_in = st.selectbox("Airline Carrier", ['Vistara', 'Air_India', 'Indigo', 'SpiceJet', 'AirAsia', 'GO_FIRST'])
        class_in = st.selectbox("Travel Cabin Class", ['Economy', 'Business'])
    with fc2:
        source_in = st.selectbox("Source City", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
        dest_in = st.selectbox("Destination City", ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'], index=1)
    with fc3:
        stops_in = st.selectbox("Flight Stops", ['zero', 'one', 'two_or_more'])
        days_in = st.slider("Days Remaining to Departure", 1, 50, 15)
        duration_in = st.slider("Estimated Duration (Hours)", 1.0, 18.0, 2.5, step=0.5)
        
    if source_in == dest_in:
        st.warning("⚠️ Source and Destination cities cannot be identical.")
    else:
        # Construct inference vector
        input_dict = {col: 0 for col in flight_cols}
        if 'duration' in input_dict: input_dict['duration'] = duration_in
        if 'days_left' in input_dict: input_dict['days_left'] = days_in
        
        # Categorical dummies
        for col_name in [f"airline_{airline_in}", f"source_city_{source_in}", f"destination_city_{dest_in}", f"stops_{stops_in}", f"class_{class_in}"]:
            if col_name in input_dict:
                input_dict[col_name] = 1
                
        pred_price = reg_flight.predict(pd.DataFrame([input_dict]))[0]
        
        st.markdown("---")
        st.subheader("💵 Estimated Ticket Price")
        p_c1, p_c2 = st.columns([1, 2])
        with p_c1:
            st.metric("Estimated Fare (INR)", f"₹ {pred_price:,.0f}")
            st.caption(f"Estimated Range: ₹ {pred_price * 0.9:,.0f} - ₹ {pred_price * 1.1:,.0f}")
        with p_c2:
            st.info(f"✈️ **Routing**: {source_in} ➔ {dest_in} | **Carrier**: {airline_in} | **Cabin**: {class_in} | **Stops**: {stops_in}")

# =========================================================
# TAB 4: Customer Segmentation Explorer
# =========================================================
with tab_mall:
    st.markdown("### 🛍️ Unsupervised Customer Segmentation via K-Means")
    st.caption("Partition mall consumer profiles based on Annual Income & Spending Score.")
    
    df_m = load_mall_customers()
    
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.subheader("⚙️ Cluster Configuration")
        k_val = st.slider("Select k Clusters:", 2, 8, 5)
        
        X_m = df_m[['Annual Income (k$)', 'Spending Score (1-100)']].values
        km = KMeans(n_clusters=k_val, init='k-means++', random_state=42, n_init=10)
        df_m['Cluster'] = km.fit_predict(X_m)
        
        sil_score = silhouette_score(X_m, df_m['Cluster'])
        st.metric("Silhouette Score", f"{sil_score:.3f}")
        
        st.markdown("#### 🎯 Personas at k=5:")
        st.markdown("""
        - 💎 **Cluster 1**: High Income, High Spend *(Luxury)*
        - 🎯 **Cluster 2**: Low Income, High Spend *(Carefree)*
        - 🛡️ **Cluster 3**: High Income, Low Spend *(Savers)*
        - ⚖️ **Cluster 4**: Medium Income & Spend *(Middle Tier)*
        - 📉 **Cluster 5**: Low Income, Low Spend *(Budget)*
        """)
        
    with m_col2:
        df_m['Cluster_Name'] = df_m['Cluster'].apply(lambda x: f"Cluster {x + 1}")
        fig_cl = px.scatter(
            df_m, x='Annual Income (k$)', y='Spending Score (1-100)', color='Cluster_Name',
            hover_data=['CustomerID', 'Gender', 'Age'],
            title=f"K-Means Cluster Projection (k={k_val})",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        # Centroids
        fig_cl.add_trace(go.Scatter(
            x=km.cluster_centers_[:, 0], y=km.cluster_centers_[:, 1],
            mode='markers', marker=dict(symbol='x', size=16, color='yellow', line=dict(width=2, color='black')),
            name='Centroids'
        ))
        fig_cl.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font={'color': 'white'}, height=480)
        fig_cl.update_xaxes(gridcolor='#30363d')
        fig_cl.update_yaxes(gridcolor='#30363d')
        st.plotly_chart(fig_cl, use_container_width=True)

# =========================================================
# TAB 5: Interactive Exploratory Data Analysis (EDA) Lab
# =========================================================
with tab_eda:
    st.markdown("### 📊 Interactive Exploratory Data Analysis (EDA) Lab")
    st.caption("Inspect distributions, correlation matrices, and statistical dispersion across repositories.")
    
    dataset_choice = st.selectbox(
        "Select Repository Dataset to Profile:",
        ["🩺 Diabetes Clinical Dataset", "🛍️ Mall Customer Spending Dataset", "✈️ Flight Price Dataset"]
    )
    
    if "Diabetes" in dataset_choice:
        eda_df, _ = load_diabetes_data()
        num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
    elif "Mall" in dataset_choice:
        eda_df = load_mall_customers()
        num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    else:
        eda_df = pd.read_csv("SUPERVISED-LEARNING/Flight-Price-Prediction/airlines_flights_data.csv", nrows=5000)
        num_cols = ['duration', 'days_left', 'price']
        
    eda_m1, eda_m2, eda_m3 = st.columns(3)
    eda_m1.metric("Total Records", f"{len(eda_df):,}")
    eda_m2.metric("Total Columns", len(eda_df.columns))
    eda_m3.metric("Missing Values", eda_df.isnull().sum().sum())
    
    st.markdown("---")
    eda_c1, eda_c2 = st.columns([1, 1])
    
    with eda_c1:
        st.subheader("🔥 Correlation Matrix Heatmap")
        corr = eda_df[num_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale='Viridis',
            title="Pearson Correlation Heatmap"
        )
        fig_corr.update_layout(paper_bgcolor="#0d1117", font={'color': 'white'}, height=360)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with eda_c2:
        st.subheader("📈 Feature Distribution & Outlier Boxplot")
        chosen_num = st.selectbox("Select Feature to Plot:", num_cols)
        fig_dist = px.histogram(
            eda_df, x=chosen_num, marginal="box", 
            title=f"Distribution & Outlier Boxplot: {chosen_num}",
            color_discrete_sequence=['#58a6ff']
        )
        fig_dist.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font={'color': 'white'}, height=360)
        fig_dist.update_xaxes(gridcolor='#30363d')
        st.plotly_chart(fig_dist, use_container_width=True)
        
    st.markdown("---")
    st.subheader("📋 Descriptive Statistics Table")
    st.dataframe(eda_df[num_cols].describe(), use_container_width=True)

# =========================================================
# TAB 6: Architecture & About
# =========================================================
with tab_info:
    st.markdown("### 🏗️ Machine Learning Architecture & Engineering Standards")
    st.markdown("""
    This portfolio demonstrates end-to-end Machine Learning systems engineering:
    
    - **Data Pipeline**: Clean extraction, categorical encoding, handling missing data, outlier normalization.
    - **Comparative Evaluation**: Rigorous evaluation using Accuracy, Precision, Recall, F1, R², and Silhouette metrics.
    - **Explainability**: Marginal impact analysis with What-If parameter simulators.
    - **Deployment**: Production interactive dashboard built with Streamlit and Plotly.
    
    ---
    ### 👨‍💻 Connect with Developer
    - **Author**: Omkar Mote
    - **GitHub**: [github.com/omkar333333](https://github.com/omkar333333)
    - **Location**: Balewadi, Pune, India
    """)

st.markdown("---")
st.caption("Crafted with ❤️ by Omkar Mote | Powered by Streamlit, Scikit-Learn & Plotly")
