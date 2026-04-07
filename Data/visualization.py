import matplotlib.pyplot as plt
import numpy as np

models = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]

#Values to plot ordered as above
baseline = [0.005, 0.232, 0.022, 0.070]
smote = [0.838, 0.232, 0.016, 0.059]
smoteenn = [0.957, 0.508, 0.476, 0.497]


x = np.arange(len(models))
width = 0.25

plt.figure()

#Maybe Change Color Idek
plt.bar(x - width, baseline, width, label="Baseline")
plt.bar(x, smote, width, label="SMOTE")
plt.bar(x + width, smoteenn, width, label="SMOTEENN")


plt.xlabel("Model")
plt.ylabel("Recall") 
plt.title("Recall Comparison Across Models and Resampling Methods")
plt.xticks(x, models, rotation=20)
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()