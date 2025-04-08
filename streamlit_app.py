# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid
import os
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import LabelEncoder
from dtreeviz import model
from PIL import Image

st.set_page_config(page_title="ML Visualizer", layout="wide")
st.title("ML Model Trainer & Visualizer")

# Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
algorithm = st.selectbox("Choose Algorithm", ["Decision Tree", "Random Forest"])
criterion = st.selectbox("Select Criterion", ["gini", "entropy"])
epochs = st.slider("Number of Epochs (for learning curve)", 3, 10, 5)
row_index = st.number_input("Row index for Tree Visualization", min_value=0, value=0)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    X = pd.get_dummies(df.iloc[:, :-1])
    y_raw = df.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    feature_names = X.columns.tolist()
    class_names = [str(cls) for cls in le.classes_]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model
    if algorithm == "Decision Tree":
        clf = DecisionTreeClassifier(criterion=criterion)
    else:
        clf = RandomForestClassifier(criterion=criterion, n_estimators=10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {round(acc*100, 2)}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, epochs)
    )
    fig2, ax2 = plt.subplots()
    ax2.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    ax2.plot(train_sizes, test_scores.mean(axis=1), label="Test")
    ax2.set_title("Learning Curve")
    ax2.legend()
    st.pyplot(fig2)

    # Visualization
    if algorithm == "Decision Tree":
        if row_index >= len(X):
            row_index = 0
        viz = model(clf, X, y, feature_names=feature_names, class_names=class_names, target_name="target").view(
            x=X.iloc[row_index], precision=2)
        svg_path = f"tree_{uuid.uuid4()}.svg"
        viz.save(svg_path)
        st.subheader("Decision Tree Visualization")
        st.image(svg_path)
        os.remove(svg_path)
    else:
        st.subheader("Random Forest Feature Importance")
        fig3, ax3 = plt.subplots()
        ax3.barh(feature_names, clf.feature_importances_)
        ax3.set_xlabel("Importance")
        ax3.set_title("Feature Importances")
        st.pyplot(fig3)
