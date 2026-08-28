import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np

data = pd.read_csv("student.csv")

x=data[['Hours']]
y=data[['score']]

model = LinearRegression()

model.fit(x,y)

predicted_score = model.predict(x)

mae = mean_absolute_error(y,predicted_score)
mse = mean_squared_error(y,predicted_score)
rmse=np.sqrt(mse)

print("RMSE:",rmse)
print("MAE:",mae)
print("MSE:",mse)

new_predicition = model.predict([[7]])
print(f"predicted sscore for {new_predicition}hour",new_predicition)