#Import all necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gradio as gr

from imblearn.over_sampling import SMOTE
import xgboost as xgb

from skopt import BayesSearchCV
from skopt.space import Real, Integer

#Load Dataset
path = "C:\\Users\\aarju\\Downloads\\blood-train.csv"
df = pd.read_csv(path)

# Drop unnecessary column (ID)
df.drop(df.columns[0], axis=1, inplace=True)

# Outlier Identification
cols=['Months_since_last_donation', 'Number_of_donations', 'Total_volume_donated', 'Months_since_first_donation']
for col in cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for outlier:{col}")
    plt.show()

#Outlier Treatment using IQR

cols_with_outliers = ['Months_since_last_donation', 'Number_of_donations', 'Total_volume_donated']
for col in cols_with_outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound,
                       np.where(df[col] > upper_bound, upper_bound, df[col]))
#Show no outliers after IQR
for col in cols_with_outliers:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for Outlier after IQR:{col}")
    plt.show()


#Feature Engineering
df['donation_rate'] = df['Number_of_donations'] / (df['Months_since_first_donation'] + 1)
df['recency_inverse'] = 1 / (df['Months_since_last_donation'] + 1)


#Define Features and Target
X = df.drop('Made_donation_in_march_2007', axis=1)
y = df['Made_donation_in_march_2007']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


#Bayesian Hyperparameter Tuning for XGBoost
param_space = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'min_child_weight': Integer(1, 10)
}

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=1,  # SMOTE already balances, keep 1
    eval_metric='logloss',
    random_state=42
)

bayes_search = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=param_space,
    n_iter=30,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
    random_state=42
)

bayes_search.fit(x_train, y_train)
best_model = bayes_search.best_estimator_

print("\n✅ Best Parameters Found:")
print(bayes_search.best_params_)


#Predictions and Evaluation
y_pred = best_model.predict(x_test)
print("\n📊 XGBoost Model Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


#Deployment - Gradio
def predict_donation(months_last, num_donations, total_volume, months_first):
    donation_rate = num_donations / (months_first + 1)
    recency_inverse = 1 / (months_last + 1)

    features = np.array([months_last, num_donations, total_volume, months_first,
                         donation_rate, recency_inverse]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    pred = best_model.predict(features_scaled)

    return "Will Donate" if pred[0] == 1 else "Will Not Donate"


#Gradio interface
feature_names = ["Months_since_last_donation", "Number_of_donations",
                 "Total_volume_donated", "Months_since_first_donation"]

app = gr.Interface(
    fn=predict_donation,
    inputs=[gr.Number(label=col) for col in feature_names],
    outputs="text",
    title="Blood Donation Prediction",
    description="Predict if a person will donate blood in March 2007"
)

app.launch(share=True)
