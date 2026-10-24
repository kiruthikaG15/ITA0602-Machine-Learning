import pandas as pd
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

columns = ['Sky', 'Temp', 'Humidity', 'Wind',
           'Water', 'Forecast', 'EnjoySport']

df = pd.DataFrame(data, columns=columns)

print("Training Dataset:")
print(df)
def find_s(df):
    hypothesis = None

    for index, row in df.iterrows():
        if row['EnjoySport'] == 'Yes':
            example = row.iloc[:-1].tolist()

            if hypothesis is None:
                hypothesis = example.copy()
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != example[i]:
                        hypothesis[i] = '?'

            print("\nStep", index + 1, ":", hypothesis)
    return hypothesis
final_hypothesis = find_s(df)

print("\nMost Specific Hypothesis:")
print(final_hypothesis)
