# 🏋️‍♂️ Fitness Analytics & Prediction

This project focuses on analyzing fitness-related data and building predictive models using machine learning. It includes regression, classification, and statistical analysis to understand and predict key metrics such as calories burned, body fat percentage, and workout types based on various user characteristics.

---

## 📦 Features

- **Linear Regression** to predict calories burned.
- **Random Forest Regression** to predict body fat percentage.
- **Random Forest Classification** to classify workout types.
- **Statistical tests** including:
  - Shapiro–Wilk test for normality,
  - t-test and Mann–Whitney U test for group comparisons,
  - Chi-squared test for dependency between categorical variables.
- Visualizations of predictions and model evaluation metrics.
- Clean and modular codebase, ready for expansion.

---

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone  <link-to-repo>
cd Exercise-recommendations
python -m venv .venv # Please remember to create virtualenv!
pip install -r requirements.txt
```

---

## 🚀 How to Run after installing dependencies


```python
python main.py
```

---

## 📁 Data Format

Your dataset should be a CSV file with the following columns:

```
Age, Gender, Weight (kg), Height (m), Max_BPM, Avg_BPM, Resting_BPM,
Session_Duration (hours), Calories_Burned, Workout_Type,
Fat_Percentage, Water_Intake (liters), Workout_Frequency (days/week),
Experience_Level, BMI
```

---