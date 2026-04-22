# iris_model_comparison.py

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# =========================================================
# 1. Reproducibility
# =========================================================
seed = 42
np.random.seed(seed)
random.seed(seed)

# =========================================================
# 2. Load dataset
# =========================================================
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# =========================================================
# 3. Train-test split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=seed,
    stratify=y
)

# =========================================================
# 4. Cross-validation strategy
# =========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# =========================================================
# 5. Hyperparameter tuning
# =========================================================
param_grids = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=seed),
        "params": {
            "max_depth": [2, 3, 4, 5, None]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=seed),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [2, 3, 4, 5, None]
        }
    },
    "SVM": {
        "model": SVC(),
        "params": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10]
        }
    }
}

best_models = {}
results = []

for model_name, config in param_grids.items():
    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[model_name] = best_model

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    test_precision = precision_score(y_test, y_pred_test, average="macro")
    test_recall = recall_score(y_test, y_pred_test, average="macro")
    test_f1 = f1_score(y_test, y_pred_test, average="macro")

    results.append({
        "Model": model_name,
        "Best Parameters": grid.best_params_,
        "CV Accuracy": grid.best_score_,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Test Precision (Macro)": test_precision,
        "Test Recall (Macro)": test_recall,
        "Test F1 (Macro)": test_f1
    })

    # Console output
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")
    print(f"Train Accuracy:   {train_acc:.4f}")
    print(f"Test Accuracy:    {test_acc:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

# =========================================================
# 6. Results table
# =========================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Test Accuracy", ascending=False)

print("\nFinal Comparison Table:")
print(results_df)

results_df.to_csv("iris_model_results.csv", index=False)

# =========================================================
# 7. Bar chart for model comparison
# =========================================================
mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

models_list = results_df["Model"].tolist()
train_values = (results_df["Train Accuracy"] * 100).tolist()
test_values = (results_df["Test Accuracy"] * 100).tolist()

x = np.arange(len(models_list))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

bars1 = ax.bar(x - width/2, train_values, width, label="Train Accuracy")
bars2 = ax.bar(x + width/2, test_values, width, label="Test Accuracy")

ax.set_xlabel("Models")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Iris Dataset: Train vs Test Accuracy by Model")
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.set_ylim(85, 102)

ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            f"{height:.1f}%",
            ha="center",
            va="bottom"
        )

fig.tight_layout()
fig.savefig("iris_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# 8. Confusion matrices
# =========================================================
for model_name, model in best_models.items():
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=class_names,
        cmap="Blues",
        ax=ax
    )
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()
    fig.savefig(f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.show()