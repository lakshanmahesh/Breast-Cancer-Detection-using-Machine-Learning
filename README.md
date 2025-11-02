# Breast-Cancer-Detection-using-Machine-Learning
This project focuses on detecting Breast Cancer using various machine learning algorithms.   The goal is to build an accurate classification model that can predict whether a tumor is benign or malignant based on cell features.

## 📊 Dataset
The dataset includes medical cell features such as:
- Clump Thickness  
- Uniformity of Cell Size  
- Uniformity of Cell Shape  
- Marginal Adhesion  
- Epithelial Cell Size  
- Bare Nuclei  
- Bland Chromatin  
- Normal Nucleoli  
- Mitoses  
- Class (Target Variable)

## ⚙️ Data Preprocessing
1. **Feature Scaling**  
   - Standardized features using `StandardScaler` from `sklearn.preprocessing`.

2. **Handling Imbalanced Data**  
   - Used `SMOTEENN` (from `imblearn.combine`) to balance the dataset.
   - Visualized data distribution before and after balancing.

## 🤖 Model Training
Applied multiple machine learning models and evaluated them using **cross-validation**.

| Model                      | Accuracy         |
|:---------------------------|:----------------:|
| K-Nearest Neighbors        | 0.9988           |
| Random Forest              | 0.9964           |
| Naive Bayes                | 0.9807           |
| Logistic Regression        | 0.9976           |
| Support Vector Machine     | 0.9976           |
| Gradient Boosting          | 0.9940           |

## 🔧 Hyperparameter Tuning
- Used `GridSearchCV` for hyperparameter optimization.
- **Best Model:** Achieved **100% accuracy (1.0)** after tuning.

## 🧩 Evaluation
**Confusion Matrix:**
   [[420 0]
   [ 0 409]]

## 🧰 Technologies Used

Python

pandas, numpy

scikit-learn

imbalanced-learn

matplotlib,

pickle

## 📈 Results & Insights

Feature scaling and SMOTEENN improved model stability.

KNN and SVM provided top-tier accuracy.

GridSearchCV fine-tuned the parameters for perfect predictio
