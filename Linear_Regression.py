import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/macbookair/Desktop/Python/ML/area_price_items.csv')

print(df.columns)

plt.xlabel('Year')
plt.ylabel('Percentage')
plt.scatter(df['Year'], df['Percentage'], color='green', marker='.')
plt.show()

new_df = df.drop('Percentage', axis='columns')
print(new_df)

model1 = linear_model.LinearRegression()
model1.fit(new_df, df['Percentage'])

Results = model1.predict([[2020]])
print(Results)