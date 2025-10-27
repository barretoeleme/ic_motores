import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error



class LinearRegressionModel(nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LinearRegressionModel, self).__init__()
         self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)





train_data = pd.DataFrame()

train_data['hysteresis'] = pd.read_csv('../dataset/2D/hysteresis_all_scaled_train.csv')['total']
train_data['id'] = pd.read_csv('../dataset/2D/idiq_all_scaled_train.csv')['id']
train_data['iq'] = pd.read_csv('../dataset/2D/idiq_all_scaled_train.csv')['iq']
train_data['joule'] = pd.read_csv('../dataset/2D/joule_all_scaled_train.csv')['total']
train_data['speed'] = pd.read_csv('../dataset/2D/speed_all_scaled_train.csv')['N']
train_data['d1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d1']
train_data['d2'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d2']
train_data['d3'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d3']
train_data['d4'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d4']
train_data['d5'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d5']
train_data['d6'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d6']
train_data['d7'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d7']
train_data['d8'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d8']
train_data['d9'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d9']
train_data['r1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['r1']
train_data['t1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['t1']



test_data = pd.DataFrame()

test_data['hysteresis'] = pd.read_csv('../dataset/2D/hysteresis_all_scaled_test.csv')['total']
test_data['id'] = pd.read_csv('../dataset/2D/idiq_all_scaled_test.csv')['id']
test_data['iq'] = pd.read_csv('../dataset/2D/idiq_all_scaled_test.csv')['iq']
test_data['joule'] = pd.read_csv('../dataset/2D/joule_all_scaled_test.csv')['total']
test_data['speed'] = pd.read_csv('../dataset/2D/speed_all_scaled_test.csv')['N']
test_data['d1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d1']
test_data['d2'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d2']
test_data['d3'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d3']
test_data['d4'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d4']
test_data['d5'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d5']
test_data['d6'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d6']
test_data['d7'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d7']
test_data['d8'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d8']
test_data['d9'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d9']
test_data['r1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['r1']
test_data['t1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['t1']


input_dim = len(train_data.columns.drop('hysteresis', 'joule')) 


variable = 'hysteresis'

columns = ['hysteresis', 'joule']

X_train = train_data.drop(columns = columns)
y_train = train_data[variable]
X_test = test_data.drop(columns = columns)
y_test = test_data[variable]



model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Linear regression model results in {variable} loss for motor 2D")
print(f"Score: {r2_score(y_test, predictions)}")
print(f"Mean squared error: {mean_squared_error(y_test, predictions)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, predictions)}")



method = 'linear'



newindex = pd.Index([method], name = 'method')
newcolumns = pd.Index(['score', 'mse', 'mape'], name = 'metric')
results = pd.DataFrame(index = newindex,
                       columns = newcolumns)
results.score.linear = r2_score(y_test, predictions)
results.mse.linear = mean_squared_error(y_test, predictions)
results.mape.linear = mean_absolute_percentage_error(y_test, predictions)

results.to_csv(f"../results/2D/{variable}/results_lin_reg.csv")



newcolumns2 = pd.Index(['y_test', 'y_pred'], name = 'data')
data = pd.DataFrame(columns = newcolumns2)
data.y_test = y_test
data.y_pred = predictions

data.to_csv(f"../pred/2D/{variable}/pred_lin_reg.csv")







