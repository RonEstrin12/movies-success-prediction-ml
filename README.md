# 🎬 Movies Success Prediction – Machine Learning Final Project

This project predicts whether a movie will be a **financial success** or **financial failure** based only on features available **before the movie is released**.
The dataset used is **The Movies Dataset** from Kaggle (`movies_metadata.csv`).

We train and compare three supervised machine learning models:

- **Decision Tree**
- **Random Forest**
- **AdaBoost**

The project follows a clean, modular structure with separate components for:
data loading, feature engineering, EDA, model training, and evaluation.

## 📁 Project Structure

```
movies_success_project/
├── data/
│   └── movies_metadata.csv
├── report/
│   └── eda_plots/
│       ├── budget_hist.png
│       ├── popularity_hist.png
│       └── movies_per_year.png
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preparation.py
│   ├── eda.py
│   └── models.py
├── notebook.ipynb
├── main.py
├── README.md
└── requirements.txt
```

## 🧠 Features Used

### Numerical Features

- Budget
- Popularity
- Runtime
- Number of genres
- Release year
- Release month

### Language One‑Hot Encoding

- lang_en
- lang_hi
- lang_fr
- lang_ru
- lang_ja
- lang_other

### Target Variable

- **success** – 1 for a financially successful movie, otherwise 0.

---

## 📈 Exploratory Data Analysis (EDA) – Plots

All plots were automatically generated and saved under:

```
report/eda_plots/
```

### 🎞 Budget Distribution

![Budget](report/eda_plots/budget_hist.png)

### ⭐ Popularity Distribution

![Popularity](report/eda_plots/popularity_hist.png)

### 📅 Movies Per Year

![Movies Per Year](report/eda_plots/movies_per_year.png)

---

## 🤖 Machine Learning Models

| Model         | Strengths                      | Notes                 |
| ------------- | ------------------------------ | --------------------- |
| Decision Tree | Simple, interpretable          | Prone to overfitting  |
| Random Forest | Best accuracy, stable results  | Handles noise well    |
| AdaBoost      | Strong recall, robust to error | Sensitive to outliers |

---

## ▶️ How to Run

### 1) Install dependencies

```
pip install -r requirements.txt
```

### 2) Place the dataset

```
movies_success_project/data/movies_metadata.csv
```

### 3) Run the project

```
python main.py
```

### 4) Or open the notebook

```
notebook.ipynb
```

---

## 🚀 Future Improvements

- Add NLP features (overview, tagline)
- Hyperparameter tuning (GridSearchCV)
- Cross‑validation
- Use advanced models (XGBoost / LightGBM)
- Integrate additional datasets (credits, keywords, ratings)

---

## 👨‍💻 Authors

- **Ron Estrin** – 318375755
- **Leedan Bayley** – 209876457
