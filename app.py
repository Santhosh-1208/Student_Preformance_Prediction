from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

model = None

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


def train_and_generate(csv_path):
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
    data['performance'] = data['performance'].map({'Poor':0,'Average':1,'Good':2})

    X = data[['attendance','study_hours','internal_marks']]
    y = data['performance']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4, eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # ==== Console Style Report ====
    report_text = ""
    report_text += f"🔹 Random Forest Accuracy: {accuracy_score(y_test, rf_pred)}\n"
    report_text += classification_report(y_test, rf_pred, zero_division=0)
    report_text += "\n\n"
    report_text += f"🔹 XGBoost Accuracy: {accuracy_score(y_test, xgb_pred)}\n"
    report_text += classification_report(y_test, xgb_pred, zero_division=0)
    report_text += "\n\n🎓 STUDENT PERFORMANCE PREDICTIONS (FROM DATASET)\n\n"

    sample_data = X_test.head(5)
    labels = {0:"Poor",1:"Average",2:"Good"}

    for i, row in sample_data.iterrows():
        pred = xgb_model.predict(pd.DataFrame([row]))[0]
        suggestions = generate_suggestions(
            row["attendance"], row["study_hours"], row["internal_marks"]
        )

        report_text += f"\n👤 Student {i}\n"
        report_text += f"Attendance: {row['attendance']}%\n"
        report_text += f"Study Hours: {row['study_hours']}\n"
        report_text += f"Internal Marks: {row['internal_marks']}\n"
        report_text += f"🎯 Predicted Performance: {labels[pred]}\n"
        report_text += "💡 Suggestions:\n"

        for s in suggestions:
            report_text += f"- {s}\n"

    # ==== Images ====
    cm = confusion_matrix(y_test, xgb_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("XGBoost Confusion Matrix")
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    plt.figure()
    plt.bar(X.columns, rf.feature_importances_)
    plt.title("Feature Importance (Random Forest)")
    plt.savefig("static/feature_importance.png")
    plt.close()

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("static/shap_summary.png")
    plt.close()

    return xgb_model, report_text


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    global model
    file = request.files['file']
    path = "student.csv"
    file.save(path)

    model, report = train_and_generate(path)
    return render_template("result.html", report=report)


@app.route('/predict', methods=['POST'])
def predict():
    attendance = float(request.form['attendance'])
    study_hours = float(request.form['study_hours'])
    internal_marks = float(request.form['internal_marks'])

    df = pd.DataFrame([[attendance, study_hours, internal_marks]],
                      columns=['attendance','study_hours','internal_marks'])

    pred = model.predict(df)[0]
    label = {0:"Poor",1:"Average",2:"Good"}[pred]

    return render_template("result.html", prediction=label)


if __name__ == '__main__':
    app.run(debug=True)