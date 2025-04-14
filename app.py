import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, session, send_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
import joblib

app = Flask(__name__)
app.secret_key = 'secret'

UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
MODEL_FOLDER = 'models'
STATIC_FOLDER = 'static'

for folder in [UPLOAD_FOLDER, REPORT_FOLDER, MODEL_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    graph_paths = []
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            session["data_path"] = filepath

            df = pd.read_csv(filepath)

            # Generate column-wise plots
            for col in df.columns:
                plt.figure(figsize=(6, 4))
                if df[col].dtype == 'object' or df[col].nunique() < 20:
                    df[col].value_counts().plot(kind='bar', color='skyblue')
                    plt.ylabel('Count')
                else:
                    df[col].plot(kind='hist', bins=20, color='salmon', edgecolor='black')
                    plt.ylabel('Frequency')
                plt.title(f'Distribution of {col}')
                plt.tight_layout()

                graph_file = f"{col}_plot.png".replace(" ", "_")
                graph_path = os.path.join(STATIC_FOLDER, graph_file)
                plt.savefig(graph_path)
                plt.close()
                graph_paths.append(graph_file)

            return render_template("index.html", filename=file.filename, graph_paths=graph_paths)

    return render_template("index.html", filename=None, graph_paths=graph_paths)


@app.route("/prepare", methods=["GET", "POST"])
def prepare():
    data_path = session.get("data_path")
    if not data_path:
        return redirect("/")
    df = pd.read_csv(data_path)

    heatmap_path = os.path.join(STATIC_FOLDER, "heatmap.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    columns = df.columns.tolist()
    return render_template("prepare.html", columns=columns, heatmap="heatmap.png")


@app.route("/train", methods=["POST"])
def train():
    data_path = session.get("data_path")
    if not data_path:
        return redirect("/")

    df = pd.read_csv(data_path)

    problem_type = request.form.get("problem_type")  # Get problem type (classification or regression)
    target_column = request.form.get("target")
    remove_columns = request.form.getlist("remove_cols")

    if target_column in remove_columns:
        remove_columns.remove(target_column)

    X = df.drop(columns=remove_columns + [target_column])
    y = df[target_column]

    # Encode categorical features
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode target if categorical (for classification only)
    if problem_type == "classification" and y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        joblib.dump(le_y, os.path.join(MODEL_FOLDER, "label_encoder_y.pkl"))

    # Impute and scale
    X = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X), columns=X.columns)
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models based on problem type
    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "K-Nearest Neighbors": KNeighborsClassifier()
        }
    elif problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Support Vector Regressor": SVR(),
            "XGBoost Regressor": XGBRegressor(),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor()
        }

    results = []
    best_model = None
    best_score = -float("inf") if problem_type == "regression" else 0
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Predictions for training and validation sets
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        if problem_type == "classification":
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            results.append({
                "Model": name,
                "Train Score": round(train_acc * 100, 2),
                "Validation Score": round(test_acc * 100, 2)
            })
            if test_acc > best_score:
                best_score = test_acc
                best_model = model
                best_model_name = name
        elif problem_type == "regression":
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            results.append({
                "Model": name,
                "Train Score": round(train_r2, 2),
                "Validation Score": round(test_r2, 2)
            })
            if test_r2 > best_score:
                best_score = test_r2
                best_model = model
                best_model_name = name

    # Save best model
    model_path = os.path.join(MODEL_FOLDER, "best_model.pkl")
    joblib.dump(best_model, model_path)

    return render_template("train.html", results=results, best_model=best_model_name, problem_type=problem_type)


@app.route("/download-model")
def download_model():
    return send_file(os.path.join(MODEL_FOLDER, "best_model.pkl"), as_attachment=True)
