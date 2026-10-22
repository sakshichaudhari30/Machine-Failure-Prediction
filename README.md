# Machine Failure Prediction Using Machine Learning

## 🚀 Live Demo
Try the deployed application here:

**[Machine Failure Prediction App](https://machine-failure-prediction-mvjo67hg8mnzruyxghtk8i.streamlit.app/)**

## 📌 Project Overview
This project uses Machine Learning classification techniques to predict potential machine failures based on sensor readings. The goal is to identify possible failures early and support predictive maintenance.

The model uses machine operating parameters such as air temperature, process temperature, rotational speed, torque, and tool wear to predict whether a machine failure may occur.

## 🎯 Project Objectives
- Analyze machine sensor data.
- Perform data cleaning and exploratory data analysis (EDA).
- Handle class imbalance in the target variable.
- Train and compare classification models.
- Tune the best-performing model.
- Deploy the trained model using Streamlit.

## 📂 Dataset Information
The dataset contains machine operating conditions and failure indicators.

**Target variable:** `Machine failure`

- `0` — No machine failure recorded.
- `1` — Machine failure recorded.

### Features Used for Prediction
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Streamlit

## 🔄 Project Workflow
1. **Data Collection:** Load the machine failure dataset.
2. **Data Cleaning:** Check for missing values, duplicate records, and data quality issues.
3. **Exploratory Data Analysis (EDA):** Analyze sensor distributions, patterns, and failure counts using visualizations.
4. **Outlier Handling:** Apply the IQR method to identify and handle outliers.
5. **Feature Selection:** Select five numerical sensor features for model training.
6. **Data Preprocessing:** Apply the saved preprocessing steps, including scaling and imputation where configured.
7. **Handling Class Imbalance:** Use SMOTE on the training data to improve learning from the minority failure class.
8. **Model Training:** Train and compare classification algorithms.
9. **Hyperparameter Tuning:** Use RandomizedSearchCV to optimize the Gradient Boosting model.
10. **Model Evaluation:** Evaluate the model using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
11. **Model Saving:** Save the trained model and preprocessing pipeline using Joblib.
12. **Deployment:** Build an interactive Streamlit application for machine failure prediction.

## 🤖 Models Compared
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Gradient Boosting was selected for hyperparameter tuning based on the initial ROC-AUC comparison.

## 📊 Model Performance
The tuned Gradient Boosting model achieved the following evaluation results on the held-out validation/test split used during development:

| Metric | Score |
|---|---:|
| Accuracy | 90.27% |
| Precision | 10.75% |
| Recall | 77.89% |
| F1-Score | 18.90% |
| ROC-AUC | 0.9045 |

**Note:** The model identifies many failure cases, but its low precision means it also produces false alarms. These results should be considered when interpreting predictions.

## 🌐 Streamlit Application
The deployed application allows users to enter five machine sensor readings and receive a predicted result.

**[Open Machine Failure Prediction App](https://machine-failure-prediction-mvjo67hg8mnzruyxghtk8i.streamlit.app/)**

### Application Output
- **Potential Machine Failure Detected:** The model predicts class `1`.
- **No Machine Failure Predicted:** The model predicts class `0`.

These are model predictions, not guarantees of machine safety or failure.

## 📁 Project Files
- `app.py` — Streamlit application.
- `gradient_boosting_model.pkl` — Saved trained model.
- `preprocessing_pipeline.pkl` — Saved preprocessing pipeline.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.

## ▶️ Run the Project Locally

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application
```bash
streamlit run app.py
```

## 💡 Key Learnings
- Data cleaning and exploratory data analysis.
- Handling imbalanced classification datasets.
- Applying preprocessing techniques to sensor data.
- Comparing and tuning machine learning models.
- Evaluating classification performance beyond accuracy.
- Deploying a machine learning model using Streamlit.

## 🔮 Future Improvements
- Improve precision while maintaining useful failure recall.
- Analyze false positives and false negatives.
- Evaluate the model on additional unseen data with confirmed labels.
- Add prediction confidence and clearer input validation.
- Monitor model performance as new sensor data becomes available.

## 👩‍💻 Author
**Sakshi Chaudhari**

GitHub: [sakshichaudhari30](https://github.com/sakshichaudhari30)

---

*This project was developed for learning and demonstration purposes. Predictions should not replace professional machine maintenance or safety procedures.*
