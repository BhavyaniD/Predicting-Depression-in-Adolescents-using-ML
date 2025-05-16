# 🧠 Predicting Depression in Adolescents using Machine Learning

This project investigates the ability of supervised machine learning models—**Logistic Regression** and **Random Forest Classifiers**—to predict **Major Depressive Episodes with Severe Impairment (MDESI)** in adolescents. The study is based on data from the **National Survey on Drug Use and Health (NSDUH)** and focuses on identifying key sociodemographic and psychosocial predictors of adolescent depression.

---

## 📌 Objective

To compare the performance and interpretability of logistic regression and random forest classifiers in identifying adolescents at risk of severe depression.  
The project also explores how different variables—like gender, age, parental involvement, and school experience—impact prediction.

---

## 📊 Dataset

- Source: **National Survey on Drug Use and Health (NSDUH), 2011–2017**
- Sample Size: ~4,500 adolescents aged 12–17
- Features:
  - Demographics (gender, age group, race)
  - Family structure (parental presence, siblings)
  - Parental involvement
  - School experience
  - Insurance status & household income

---

## 🧪 Methodology

1. **Preprocessing & Feature Engineering**
   - Converted categorical features
   - Balanced dataset for training
2. **Modeling Techniques**
   - Logistic Regression
   - Random Forest (with Out-of-Bag error tuning)
3. **Model Evaluation Metrics**
   - Accuracy
   - Recall
   - AUC-ROC
   - Confusion Matrix
4. **Threshold Optimization**
   - Compared performance at various decision thresholds (θ)

---

## 📈 Results Summary

| Metric                      | Logistic Regression | Random Forest |
|----------------------------|---------------------|---------------|
| Test Accuracy              | 68.46%              | 69.46%        |
| Test Recall                | 64.08%              | 67.89%        |
| AUC (Test)                 | 0.7659              | 0.7656        |
| Interpretability           | ✅ High             | ❌ Moderate   |
| Model Flexibility          | ❌ Limited          | ✅ High       |

- **Top Predictors:** Gender (female), age group (14–17), low parental involvement, negative school experience
- **Random Forest** yielded better sensitivity and generalization
- **Logistic Regression** provided clearer interpretability (odds ratios)

---

## 🔍 Key Takeaways

- **Random Forest** is more effective when recall is prioritized (minimizing false negatives).
- **Logistic Regression** is ideal for resource-constrained settings where model transparency is needed.
- Both models are viable tools for early mental health screening with ~70% accuracy.

---

## 📚 References

- [Springer: ML Algorithms for Depression Detection](https://link.springer.com/article/10.1007/s10916-020-01576-1)  
- [ScienceDirect: Depression Detection Challenges](https://www.sciencedirect.com/science/article/pii/S0169260720300460)  
- [IEEE: Deep Learning for Depression](https://ieeexplore.ieee.org/document/9206140)

---

## 👩‍💻 Author

**Bhavyani Dodda**  
MS Data Science – Rutgers University  
📧 bhavyani.dodda123@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/bhavyani-dodda-414ab6195)

---
