
# 🛒 Walmart Sales Forecasting

> 🚀 This end-to-end machine learning project demonstrates my ability to build, train, evaluate, and deploy scalable sales forecasting models using Python, XGBoost, and Flask. It reflects strong data preprocessing, modeling, and backend deployment skills — crucial for data science and software engineering roles.

---

## 📌 Project Overview

This project forecasts Walmart product sales using historical data, promotions, and seasonal trends.  
It compares multiple models, selects the best-performing one (XGBoost), and deploys it via a Flask API for real-time inference.

---

## 🔧 Features

- ✅ Data cleaning, integration, transformation, and EDA
- 🧠 Built multiple regression models (Linear Regression, Random Forest, XGBoost)
- 📊 Evaluated model performance using RMSE, MAE, and R² Score
- 🔁 Selected best model based on accuracy and generalization
- 🌐 Deployed prediction endpoint using Flask
- 💾 Handled large `.pth` model file with Git LFS

---

## 📈 Model Performance

| Metric       | Value  |
|--------------|--------|
| **R² Score** | 0.91   |
| **RMSE**     | 2.84   |
| **MAE**      | 1.92   |

> ✅ **Best Model**: XGBoost Regressor

---

## 🧰 Tech Stack

- **Languages:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost
- **Framework:** Flask
- **Version Control:** Git, GitHub
- **Model File Management:** Git LFS

---

## 🚀 How to Run Locally

#### 1. Clone the Repository

```bash
git clone https://github.com/jjs2099/wallmart_sales.git
cd wallmart_sales
```

#### 2. (Optional) Set up Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate  # macOS/Linux
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the Flask App

```bash
python app.py
```

Then visit [http://localhost:5000](http://localhost:5000)

---

## 📁 Project Structure

```
wallmart_sales/
├── wallmart_app/
│   ├── static/
│   │   └── random_forest_model.pth
│   ├── templates/
│   │   └── index.html
│   ├── __init__.py
│   └── app.py
├── requirements.txt
├── README.md
└── ...
```

---

## 💡 What I Learned

- Cleaning, transforming, and preparing real-world retail data
- Comparing traditional vs ensemble models in regression
- Deploying ML models with Flask
- Using Git LFS for large model storage
- Writing modular code for production-readiness

---

## ⚠️ Large File Notice

This project includes a `.pth` model file (~56MB), which exceeds GitHub's 50MB recommendation.  
It is tracked using **Git LFS**. If you're cloning or contributing, install Git LFS first:

```bash
git lfs install
git lfs pull
```

---


