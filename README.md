## 🌸 Iris Classification: Model Comparison Study

This project compares the performance of three supervised machine learning models — Decision Tree, Random Forest, and Support Vector Machine (SVM) — on the Iris dataset.

---

## 🎯 Objective

To evaluate how different classification algorithms perform under a consistent and reproducible machine learning pipeline.

---

## ⚙️ Methodology

* Dataset: Iris dataset (sklearn)
* Train/Test Split: 80/20 (stratified)
* Cross-validation: 5-fold (random_state=42)
* Hyperparameter tuning: GridSearchCV

### Models:

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

* SVM achieved perfect test accuracy, indicating strong generalisation
* Decision Tree shows slight overfitting
* Random Forest is more stable but slightly less accurate

---

## 📊 Outputs

* Accuracy comparison chart
* Confusion matrices
* Results CSV

---

## 🔁 Reproducibility

All experiments use:

```python
random_state = 42
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

The Iris dataset is simple and highly separable, so results may not generalise to more complex real-world datasets.
