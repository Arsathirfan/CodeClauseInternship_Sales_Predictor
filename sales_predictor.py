import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\IRFAN\\Downloads\\archive (6)\\business.retailsales.csv")

prp = sklearn.preprocessing.LabelEncoder()

data['Product Type'] = prp.fit_transform(list(data["Product Type"]))

data['Product Type'].fillna(data['Product Type'].mean, inplace=True)

X = np.array(data.drop(['Total Net Sales'], axis=1))
y = np.array(data['Total Net Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

value = np.array([[1, 34, 14935, -594, -1609]])

predict = model.predict(value)

print("Predicted Total Net Sales:", predict)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.show()
