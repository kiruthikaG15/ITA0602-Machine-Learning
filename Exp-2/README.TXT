
import pandas as pd
from itertools import product

# Create Dataset
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

print("Training Data:")
print(df)

X = df.iloc[:, :-1].values.tolist()
Y = df.iloc[:, -1].tolist()

# Generate Hypotheses
domains = [set(row[i] for row in X) for i in range(len(X[0]))]
choices = [sorted(d) + ['?'] for d in domains]
hypotheses = list(product(*choices))

# Check Hypothesis
def covers(h, x):
    return all(a == '?' or a == b for a, b in zip(h, x))

# Find Consistent Hypotheses
consistent = [
    h for h in hypotheses
    if all(covers(h, x) == (y == 'Yes')
           for x, y in zip(X, Y))
]

print("\nTotal Consistent Hypotheses:", len(consistent))

print("\nAll Consistent Hypotheses:")
for h in consistent:
    print(h)

