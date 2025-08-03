import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error



train_data_Nabla = pd.DataFrame()

train_data_Nabla['hysteresis'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/hysteresis_all_scaled_train.csv')['total']
train_data_Nabla['id'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/idiq_all_scaled_train.csv')['id']
train_data_Nabla['iq'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/idiq_all_scaled_train.csv')['iq']
train_data_Nabla['joule'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/joule_all_scaled_train.csv')['total']
train_data_Nabla['speed'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/speed_all_scaled_train.csv')['N']
train_data_Nabla['d1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d1']
train_data_Nabla['d2'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d2']
train_data_Nabla['d3'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d3']
train_data_Nabla['d4'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d4']
train_data_Nabla['d5'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d5']
train_data_Nabla['d6'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d6']
train_data_Nabla['d7'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d7']
train_data_Nabla['d8'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d8']



test_data_Nabla = pd.DataFrame()

test_data_Nabla['hysteresis'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/hysteresis_all_scaled_test.csv')['total']
test_data_Nabla['id'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/idiq_all_scaled_test.csv')['id']
test_data_Nabla['iq'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/idiq_all_scaled_test.csv')['iq']
test_data_Nabla['joule'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/joule_all_scaled_test.csv')['total']
test_data_Nabla['speed'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/speed_all_scaled_test.csv')['N']
test_data_Nabla['d1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d1']
test_data_Nabla['d2'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d2']
test_data_Nabla['d3'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d3']
test_data_Nabla['d4'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d4']
test_data_Nabla['d5'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d5']
test_data_Nabla['d6'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d6']
test_data_Nabla['d7'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d7']
test_data_Nabla['d8'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d8']



variable = 'joule'

columns = ['hysteresis', 'joule']

X_train = train_data_Nabla.drop(columns = columns)
y_train = train_data_Nabla[variable]
X_test = test_data_Nabla.drop(columns = columns)
y_test = test_data_Nabla[variable]



model_Nabla = CatBoostRegressor()
model_Nabla.fit(X_train, y_train)

predictions = model_Nabla.predict(X_test)
print("CatBoost model results in joule loss for motor Nabla")
print(f"Score: {r2_score(y_test, predictions)}")
print(f"Mean squared error: {mean_squared_error(y_test, predictions)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, predictions)}")



method = 'catboost'



newindex = pd.Index([method], name = 'method')
newcolumns = pd.Index(['score', 'mse', 'mape'], name = 'metric')
results = pd.DataFrame(index = newindex,
                       columns = newcolumns)
results.score.catboost = r2_score(y_test, predictions)
results.mse.catboost = mean_squared_error(y_test, predictions)
results.mape.catboost = mean_absolute_percentage_error(y_test, predictions)

results.to_csv("/mnt/c/prog/IniciacaoCientifica/results/Nabla/joule/results_catboost.csv")



newcolumns2 = pd.Index(['y_test', 'y_pred'], name = 'data')
data = pd.DataFrame(columns = newcolumns2)
data.y_test = y_test
data.y_pred = predictions

data.to_csv("/mnt/c/prog/IniciacaoCientifica/pred/Nabla/joule/pred_catboost.csv")







