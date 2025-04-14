# 🔮 AutoML Web App

A beginner-friendly and powerful web application for automated machine learning. This app allows users to upload datasets, explore them visually, handle missing data, select target variables, and train multiple ML models (classification or regression) — all through a simple interface.

Hosted at : https://automl-flask.onrender.com/
---

## 📸 Project Preview

<!-- Insert a screenshot or GIF of your app interface below -->
![image](https://github.com/user-attachments/assets/9e64f64f-5451-4eb8-9267-0cc6ad6bdf8e)
![image](https://github.com/user-attachments/assets/7a004bfc-8731-472a-ab23-7fe0abfc88c4)
![image](https://github.com/user-attachments/assets/1c56a55d-2d55-47f0-a12e-8ea1a7bcc86e)
![image](https://github.com/user-attachments/assets/6184c55c-03b4-42f5-93ae-a69ef5966a23)
![image](https://github.com/user-attachments/assets/9b74e2b4-f39d-4a3d-8c9f-c128cc56ac68)


---

## ✨ Features

- 📂 Upload your CSV dataset
- 📊 Automatic EDA: View graphs, data types, and missing value overview
- 🧪 Correlation heatmap for smart feature selection
- 🎯 Choose your target variable
- 📈 Select whether to perform Classification or Regression
- 🚀 Trains 5 models and shows both Training & Validation scores
- 🥇 Best model is auto-identified
- 💾 Download trained model and performance report (CSV)

---

## 🛠️ Tech Stack

- Python
- Flask
- scikit-learn
- XGBoost
- Pandas / NumPy
- Matplotlib / Seaborn
- Gunicorn (for production deployment)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
python app.py
