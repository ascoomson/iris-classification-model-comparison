## 🌸 Iris Classification: Reproducible Model Comparison

This project presents a structured comparison of three supervised machine learning classifiers — Decision Tree, Random Forest, and Support Vector Machine (SVM) — using the Iris dataset.

The workflow emphasizes reproducibility, hyperparameter tuning with cross-validation, and robust model evaluation using multiple performance metrics. It demonstrates how different algorithms behave under a consistent pipeline, highlighting differences in generalisation, overfitting, and predictive performance.

> Built as part of a broader effort to develop reproducible and interpretable machine learning workflows.
---

## 🎯 Objective

To evaluate how different classification algorithms perform under a consistent and reproducible machine learning pipeline, with a focus on generalisation performance.

---

## ## 🌸 Dataset Overview

The Iris dataset contains measurements of three distinct flower species:

* *Iris setosa*
* *Iris versicolor*
* *Iris virginica*

Although the dataset is numerical (sepal length, sepal width, petal length, and petal width), the visual differences between species help explain why this classification task is relatively well-structured.


![iris_species_diagram](images/iris_species_diagram.webp)

*Illustration of Iris species highlighting sepal and petal structures. Source: Adapted from https://dev.to/vaib/hands-on-ai-for-beginners-classifying-iris-flowers-in-python-2j4h (original creator not identified).*

---

### 📏 Typical Feature Ranges (cm)

| Feature      | Setosa    | Versicolor | Virginica |
| ------------ | --------- | ---------- | --------- |
| Sepal Length | 4.3 – 5.8 | 4.9 – 7.0  | 4.9 – 7.9 |
| Sepal Width  | 2.3 – 4.4 | 2.0 – 3.4  | 2.2 – 3.8 |
| Petal Length | 1.0 – 1.9 | 3.0 – 5.1  | 4.5 – 6.9 |
| Petal Width  | 0.1 – 0.6 | 1.0 – 1.8  | 1.4 – 2.5 |

These feature ranges highlight why the classes are well-separated, particularly along petal dimensions, which strongly influence model performance.

---

## ⚙️ Methodology

* Dataset: Iris dataset (sklearn)
* Train/Test Split: 80/20 (stratified)
* Cross-validation: 5-fold (random_state=42)
* Hyperparameter tuning: GridSearchCV

### Models Evaluated:

* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

---

## 📊 Results

| Model         | CV Accuracy | Train Accuracy | Test Accuracy | F1 Score   |
| ------------- | ----------- | -------------- | ------------- | ---------- |
| SVM           | 0.9833      | 0.9833         | **1.0000**    | **1.0000** |
| Decision Tree | 0.9583      | 0.9833         | 0.9667        | 0.9666     |
| Random Forest | 0.9667      | 0.9667         | 0.9333        | 0.9333     |

---

## 🔍 Key Insights

* SVM achieved perfect test accuracy, indicating strong generalisation on this dataset
* Decision Tree exhibits slight overfitting, with higher training than test accuracy
* Random Forest provides a more stable performance but slightly lower predictive accuracy
* The dataset is highly separable, allowing decision boundaries to perform exceptionally well
  
---

## 📊 Outputs

* Accuracy comparison chart
* Confusion matrices for each model
* Exported Results (.csv)

---

## 🔁 Reproducibility

All experiments use a fixed random seed:

```python
random_state = 42
```
This ensures consistent and reproducible results across runs.

---

## 📁 Project Structure

```
iris-classification-model-comparison/
│
├── iris_model_comparison.py
├── README.md
├── requirements.txt
│
├── images/
│   └── iris_species_diagram.webp
│
├── outputs/
│   ├── iris_accuracy_comparison.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_svm.png
│   └── iris_model_results.csv
```
---

## 🛠️ Tech Stack

Python · NumPy · Pandas · Scikit-learn · Matplotlib

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python iris_model_comparison.py
```

---

## 📌 Note

The Iris dataset is simple, small and highly structured, making it easy for models to achieve near-perfect accuracy. As such, results may not be generalize to more complex, noisy real-world datasets.

---

## 💡Why not Deep Learning?

Given the small dataset size (150 samples) and the relatively simple, structured feature space, classical machine learning models are more appropriate for this task. Models such as SVM already achieve near-perfect performance, making more complex approaches like neural networks unnecessary and potentially prone to overfitting.

While deep learning models—particularly Convolutional Neural Networks (CNNs)—excel in image classification tasks, this project uses pre-engineered numerical features (sepal and petal measurements) rather than raw images. As a result, the underlying patterns are already well represented, and simpler models can effectively learn the decision boundaries.
