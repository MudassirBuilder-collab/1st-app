# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------
# STEP 1: Load & Clean Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\L4-PC23\Downloads\archive\patient.csv")
    columns = ["sex", "age", "pneumonia", "diabetes", "asthma", "outcome"]
    df = df[columns]
    # Binary conversion
    for col in df.columns:
        if col != "age":
            df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)
    return df

df = load_data()

# -----------------------------
# STEP 2: Train KNN Model with Scaling
# -----------------------------
X = df.drop(columns="outcome")
y = df["outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale age column
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled["age"] = scaler.fit_transform(X_train[["age"]])
X_test_scaled["age"] = scaler.transform(X_test[["age"]])

# Weighted KNN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# Streamlit Layout: Tabs
# -----------------------------
st.set_page_config(page_title="COVID KNN Predictor", layout="wide")
tabs = st.tabs(["🏠 Introduction", "🩺 Check Yourself"])

# -----------------------------
# Tab 1: Introduction / Model Info
# -----------------------------
with tabs[0]:
    st.header("📝 Data & Model Overview")
    
    st.subheader("1. Dataset Info")
    st.text(df.info())
    
    st.subheader("2. First 10 Rows of Original Data")
    st.dataframe(df.head(10))
    
    st.subheader("3. First 10 Rows After Binary Conversion")
    st.dataframe(df.head(10))  # Already binary converted
    
    st.subheader("4. KNN Model Description")
    st.write("""
        K-Nearest Neighbors (KNN) is a simple, supervised machine learning algorithm 
        used for classification. In this project, we use KNN to predict COVID outcomes 
        based on patient symptoms and demographic data. Weighted KNN with age scaling 
        is applied for better predictions.
    """)
    
    st.subheader("5. Model Accuracy")
    st.info(f"KNN Model Accuracy on Test Data: {round(accuracy*100,2)}%")
    
    st.subheader("6. Visualizations")
    
    # Outcome Distribution
    st.markdown("**Outcome Distribution**")
    fig1, ax1 = plt.subplots()
    counts = df["outcome"].value_counts().sort_index()
    bars = ax1.bar(["Negative","Positive"], counts, color=["green","red"])
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(bar.get_height()),
                 ha='center', va='bottom')
    ax1.set_ylabel("Number of Patients")
    ax1.set_title("COVID Outcome Distribution")
    st.pyplot(fig1)
    
    # Age Distribution
    st.markdown("**Age Distribution**")
    fig2, ax2 = plt.subplots()
    ax2.hist(df["age"], bins=15, color="skyblue", edgecolor="black")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Number of Patients")
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)
    
    # Correct vs Incorrect Predictions
    st.markdown("**Prediction Results (Correct vs Incorrect)**")
    correct = (y_test == y_pred).sum()
    incorrect = (y_test != y_pred).sum()
    fig3, ax3 = plt.subplots()
    bars = ax3.bar(["Correct","Incorrect"], [correct, incorrect], color=["green","red"])
    for bar in bars:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height(), str(bar.get_height()),
                 ha='center', va='bottom')
    ax3.set_ylabel("Number of Samples")
    ax3.set_title("KNN Prediction Results")
    st.pyplot(fig3)
    
    # Actual vs Predicted
    st.markdown("**Actual vs Predicted Outcomes**")
    actual_counts = y_test.value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    fig4, ax4 = plt.subplots()
    ax4.bar(["Actual Negative","Actual Positive"], actual_counts, alpha=0.7, color="blue")
    ax4.bar(["Pred Negative","Pred Positive"], pred_counts, alpha=0.7, color="orange")
    ax4.set_ylabel("Number of Samples")
    ax4.set_title("Actual vs Predicted Outcomes")
    st.pyplot(fig4)

# -----------------------------
# Tab 2: Check Yourself / Prediction Form
# -----------------------------
with tabs[1]:
    st.header("🩺 Check Yourself")
    st.write("Enter your details and symptoms to check your predicted COVID outcome.")
    
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.slider("Age", min_value=0, max_value=100, value=30)
    pneumonia = st.selectbox("Pneumonia", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    asthma = st.selectbox("Asthma", ["Yes", "No"])
    
    input_df = pd.DataFrame({
        "sex": [1 if sex=="Male" else 0],
        "age": [age],
        "pneumonia": [1 if pneumonia=="Yes" else 0],
        "diabetes": [1 if diabetes=="Yes" else 0],
        "asthma": [1 if asthma=="Yes" else 0]
    })
    
    # Scale age for user input
    input_df["age"] = scaler.transform(input_df[["age"]])
    
    if st.button("Predict"):
        result = knn.predict(input_df)[0]
        st.subheader("Prediction Result:")
        if result == 1:
            st.error("⚠️ Positive COVID Outcome. Consult a doctor immediately.")
        else:
            st.success("✅ Negative COVID Outcome. Stay safe!")
