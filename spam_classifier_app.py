# spam_classifier_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    column_names = [f"feature_{i}" for i in range(57)] + ["is_spam"]
    df = pd.read_csv(url, header=None, names=column_names)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("is_spam", axis=1)
    y = df["is_spam"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump(model, "spam_model.pkl")
    return model, acc, cm, cr

def predict_input(model, inputs):
    array = np.array(inputs).reshape(1, -1)
    pred = model.predict(array)[0]
    prob = model.predict_proba(array)[0][pred]
    return pred, prob

st.set_page_config(page_title="Spam Classifier", layout="wide")
st.title("📧 Spam Email Classifier — End-to-End Data Science App")

df = load_data()

tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🧠 Prediction", "🔍 Model Evaluation"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Target Variable Distribution")
    st.bar_chart(df['is_spam'].value_counts())

    st.subheader("Correlation with Target (Top 10)")
    corr = df.corr()['is_spam'].drop('is_spam').abs().sort_values(ascending=False).head(10)
    st.write(corr)

    st.subheader("Heatmap of Top Correlations")
    top_features = corr.index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df[top_features + ['is_spam']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tab2:
    st.subheader("Enter 57 Features (SpamBase Format)")
    user_input = []
    cols = st.columns(3)
    for i in range(57):
        val = cols[i % 3].number_input(f"Feature {i}", min_value=0.0, max_value=100.0, value=0.0, key=f"f{i}")
        user_input.append(val)

    if os.path.exists("spam_model.pkl"):
        model = joblib.load("spam_model.pkl")
    else:
        model, _, _, _ = train_model(df)

    if st.button("Predict Spam / Not Spam"):
        pred, prob = predict_input(model, user_input)
        label = "🟥 Spam" if pred == 1 else "🟩 Not Spam"
        st.write(f"### Result: **{label}**")
        st.write(f"Confidence: **{prob:.2%}**")

with tab3:
    st.subheader("Training Model...")
    model, acc, cm, cr = train_model(df)

    st.metric("🎯 Accuracy", f"{acc*100:.2f}%")
    
    st.subheader("📌 Confusion Matrix")
    cm_df = pd.DataFrame(cm, columns=["Predicted Not Spam", "Predicted Spam"], index=["Actual Not Spam", "Actual Spam"])
    st.dataframe(cm_df)

    st.subheader("📋 Classification Report")
    st.json(cr)
