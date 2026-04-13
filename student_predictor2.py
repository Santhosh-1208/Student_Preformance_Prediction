import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

data = pd.read_csv("student.csv")

data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

data['performance'] = data['performance'].map({
    'Poor': 0,
    'Average': 1,
    'Good': 2
})

X = data[['attendance', 'study_hours', 'internal_marks']]
y = data['performance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("\n🔹 Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred, zero_division=0))

print("\n🔹 XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred, zero_division=0))

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion_matrix(y_test, rf_pred, "Random Forest Confusion Matrix")
plot_confusion_matrix(y_test, xgb_pred, "XGBoost Confusion Matrix")

importances = rf_model.feature_importances_
features = X.columns

plt.figure()
plt.bar(features, importances)
plt.title("Feature Importance (Random Forest)")
plt.show()

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

joblib.dump(xgb_model, "student_model.pkl")

def predict_student(attendance, study_hours, internal_marks):
    input_data = pd.DataFrame([[attendance, study_hours, internal_marks]],
                              columns=X.columns)

    pred = xgb_model.predict(input_data)[0]
    labels = {0: "Poor", 1: "Average", 2: "Good"}
    return labels[pred]

def generate_suggestions(attendance, study_hours, internal_marks):
    suggestions = []

    if attendance < 75:
        suggestions.append(f"⚠️ Attendance is low ({attendance}%). Try to reach 80%+")

    if study_hours < 2:
        suggestions.append(f"📚 Study time is low ({study_hours} hrs). Increase to 2–4 hrs/day")

    if internal_marks < 65:
        suggestions.append(f"📝 Internal marks are low ({internal_marks}). Revise regularly and practice tests")

    if attendance > 85 and study_hours > 3 and internal_marks > 75:
        suggestions.append("🔥 Excellent performance! Keep it up!")

    if not suggestions:
        suggestions.append("👍 You are doing good. Minor improvements can boost performance")

    return suggestions

sample_data = X_test.head(5)

print("\n🎓 STUDENT PERFORMANCE PREDICTIONS (FROM DATASET)\n")

for i, row in sample_data.iterrows():
    attendance = row["attendance"]
    study_hours = row["study_hours"]
    internal_marks = row["internal_marks"]

    result = predict_student(attendance, study_hours, internal_marks)
    suggestions = generate_suggestions(attendance, study_hours, internal_marks)

    print(f"\n👤 Student {i}")
    print(f"Attendance: {attendance}%")
    print(f"Study Hours: {study_hours}")
    print(f"Internal Marks: {internal_marks}")

    print("🎯 Predicted Performance:", result)
    print("💡 Suggestions:")

    for s in suggestions:
        print("-", s)
import os

if not os.path.exists("static"):
    os.makedirs("static")

# Save accuracy report to text file
with open("static/report.txt", "w",encoding="utf-8") as f:
    f.write("Random Forest Accuracy:\n")
    f.write(str(accuracy_score(y_test, rf_pred)) + "\n")
    f.write(classification_report(y_test, rf_pred))
    
    f.write("\n\nXGBoost Accuracy:\n")
    f.write(str(accuracy_score(y_test, xgb_pred)) + "\n")
    f.write(classification_report(y_test, xgb_pred))


# Save sample student predictions to text file
with open("static/sample_predictions.txt", "w",encoding="utf-8") as f:
    sample_data = X_test.head(5)

    for i, row in sample_data.iterrows():
        attendance = row["attendance"]
        study_hours = row["study_hours"]
        internal_marks = row["internal_marks"]

        result = predict_student(attendance, study_hours, internal_marks)
        suggestions = generate_suggestions(attendance, study_hours, internal_marks)

        f.write(f"\nStudent {i}\n")
        f.write(f"Attendance: {attendance}%\n")
        f.write(f"Study Hours: {study_hours}\n")
        f.write(f"Internal Marks: {internal_marks}\n")
        f.write(f"Predicted Performance: {result}\n")
        f.write("Suggestions:\n")
        for s in suggestions:
            f.write(f"- {s}\n")

# Confusion Matrix (XGBoost)
cm = confusion_matrix(y_test, xgb_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("XGBoost Confusion Matrix")
plt.savefig("static/confusion_matrix.png")
plt.close()

# Feature Importance (Random Forest)
plt.figure()
plt.bar(X.columns, rf_model.feature_importances_)
plt.title("Feature Importance (Random Forest)")
plt.savefig("static/feature_importance.png")
plt.close()

# SHAP Summary Plot
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("static/shap_summary.png")
plt.close()
