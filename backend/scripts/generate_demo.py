import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Generating synthetic biased dataset...")

np.random.seed(42)
n_samples = 1500

# Synthetic Employee Retention features
age = np.random.randint(22, 65, n_samples)
salary = np.random.randint(35000, 150000, n_samples)
performance = np.random.uniform(1.0, 5.0, n_samples)

# Sensitive Attribute: Demographic Group (e.g. 0 vs 1)
# Group 0 is given materially lower salaries and promotion potentials, representing bias
group = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
salary[group == 0] = salary[group == 0] * 0.75 

# Target Variable (Attrition: 1 left the company, 0 stayed)
# Attrition is organically higher for lower salaries and lower performance
base_prob = 0.1
attrition_prob = base_prob + (salary < 50000) * 0.2 + (performance < 2.5) * 0.2 

# *BLATANT BIAS INJECTION*: 
# Make it artificially much higher if they belong to group 0 (Direct Discrimination)
attrition_prob += (group == 0) * 0.35

attrition = np.random.binomial(1, np.clip(attrition_prob, 0, 1))

df = pd.DataFrame({
    'Age': age,
    'Salary': salary,
    'Performance': performance,
    'Demographic_Group': group,
    'Attrition': attrition
})

output_csv = os.path.join(os.path.dirname(__file__), '..', '..', 'demo_dataset.csv')
df.to_csv(output_csv, index=False)
print(f"Dataset generated => {output_csv}")

print("Training Scikit-Learn Model...")
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Model learns the injected historical bias directly from the feature set
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

output_pkl = os.path.join(os.path.dirname(__file__), '..', '..', 'demo_model.pkl')
joblib.dump(model, output_pkl)
print(f"Model saved => {output_pkl}")
print("Demo Generation Complete!")
