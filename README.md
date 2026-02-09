
# 🩺 Breast Cancer Detection using Machine Learning (Google Colab)

## 📌 Overview

This project implements a **Breast Cancer Detection system using Machine Learning** to classify tumors as **malignant** or **benign**.
The model is built using the **Naive Bayes (GaussianNB)** algorithm and the **Breast Cancer dataset from Scikit-learn**.

Early and accurate detection of breast cancer plays a crucial role in improving survival rates and reducing unnecessary treatments. This project demonstrates how machine learning can assist in intelligent healthcare systems.

---

## 🎯 Objective

* Build a machine learning model to classify breast cancer tumors
* Achieve high accuracy using a simple and efficient algorithm
* Understand the complete ML pipeline: data loading, training, testing, and evaluation

---

## 🚀 Features

* Uses a **well-known medical dataset** from Scikit-learn
* Binary classification: **Malignant vs Benign**
* Simple and interpretable **Naive Bayes classifier**
* High accuracy (~97%)
* Fully runnable on **Google Colab**

---

## 🛠️ Technologies Used

* **Python**
* **Google Colab**
* **Scikit-learn**
* **Naive Bayes (GaussianNB)**
* **NumPy**

---

## 📂 Dataset Information

* Dataset: **Breast Cancer Wisconsin Dataset**
* Source: `sklearn.datasets.load_breast_cancer`
* Classes:

  * `0` → Malignant
  * `1` → Benign
* Features: 30 numerical features related to tumor characteristics

---

## ▶️ How to Run on Google Colab

### Step 1: Open Google Colab

Visit 👉 [https://colab.research.google.com](https://colab.research.google.com)
Create a **New Notebook**

---

### Step 2: Import Required Libraries

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
```

---

### Step 3: Load the Dataset

```python
data = load_breast_cancer()
labels = data["target"]
features = data["data"]
```

---

### Step 4: Split the Dataset

```python
train, test, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
```

---

### Step 5: Train the Model

```python
gnb = GaussianNB()
gnb.fit(train, train_labels)
```

---

### Step 6: Make Predictions & Evaluate

```python
preds = gnb.predict(test)
print("Accuracy:", accuracy_score(test_labels, preds))
```
---

## 📊 Model Performance

* **Algorithm:** Gaussian Naive Bayes
* **Accuracy:** ~97%
* The model correctly classifies tumors in most test cases, making it reliable for basic medical classification tasks.
---

## 🧠 Why Naive Bayes?

* Works well for **binary classification**
* Fast and efficient
* Performs well even with limited data
* Commonly used in medical diagnosis problems

---

## 🎯 Use Cases

* Educational demonstration of ML in healthcare
* Beginner-friendly medical ML project
* Foundation for advanced cancer prediction systems
* Can be integrated into **Flask/Django** applications

---

## 🔮 Future Enhancements

* Add data visualization
* Compare with other algorithms (SVM, Random Forest, Logistic Regression)
* Deploy as a web app
* Add confusion matrix and ROC curve
* Integrate with Django or Flask backend

---
