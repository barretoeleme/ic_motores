import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error



train_data_2D = pd.DataFrame()

train_data_2D['hysteresis'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/hysteresis_all_scaled_train.csv')['total']
train_data_2D['id'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/idiq_all_scaled_train.csv')['id']
train_data_2D['iq'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/idiq_all_scaled_train.csv')['iq']
train_data_2D['joule'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/joule_all_scaled_train.csv')['total']
train_data_2D['speed'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/speed_all_scaled_train.csv')['N']
train_data_2D['d1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d1']
train_data_2D['d2'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d2']
train_data_2D['d3'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d3']
train_data_2D['d4'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d4']
train_data_2D['d5'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d5']
train_data_2D['d6'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d6']
train_data_2D['d7'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d7']
train_data_2D['d8'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d8']
train_data_2D['d9'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d9']
train_data_2D['r1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['r1']
train_data_2D['t1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['t1']



test_data_2D = pd.DataFrame()

test_data_2D['hysteresis'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/hysteresis_all_scaled_test.csv')['total']
test_data_2D['id'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/idiq_all_scaled_test.csv')['id']
test_data_2D['iq'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/idiq_all_scaled_test.csv')['iq']
test_data_2D['joule'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/joule_all_scaled_test.csv')['total']
test_data_2D['speed'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/speed_all_scaled_test.csv')['N']
test_data_2D['d1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d1']
test_data_2D['d2'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d2']
test_data_2D['d3'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d3']
test_data_2D['d4'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d4']
test_data_2D['d5'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d5']
test_data_2D['d6'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d6']
test_data_2D['d7'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d7']
test_data_2D['d8'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d8']
test_data_2D['d9'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d9']
test_data_2D['r1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['r1']
test_data_2D['t1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['t1']



variable = 'hysteresis'

columns = ['hysteresis', 'joule']

X_train = train_data_2D.drop(columns = columns)
y_train = train_data_2D[variable]
X_test = test_data_2D.drop(columns = columns)
y_test = test_data_2D[variable]



model_2D = LinearRegression()
model_2D.fit(X_train, y_train)

predictions = model_2D.predict(X_test)
print("Linear regression model results in hysteresis loss for motor 2D")
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

results.to_csv("/mnt/c/prog/IniciacaoCientifica/results/2D/hysteresis/results_lin_reg.csv")



newcolumns2 = pd.Index(['y_test', 'y_pred'], name = 'data')
data = pd.DataFrame(columns = newcolumns2)
data.y_test = y_test
data.y_pred = predictions

data.to_csv("/mnt/c/prog/IniciacaoCientifica/pred/2D/hysteresis/pred_lin_reg.csv")







