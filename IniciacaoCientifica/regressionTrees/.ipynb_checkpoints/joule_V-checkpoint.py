import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error



train_data_V = pd.DataFrame()

train_data_V['hysteresis'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/hysteresis_all_scaled_train.csv')['total']
train_data_V['id'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/idiq_all_scaled_train.csv')['id']
train_data_V['iq'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/idiq_all_scaled_train.csv')['iq']
train_data_V['joule'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/joule_all_scaled_train.csv')['total']
train_data_V['speed'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/speed_all_scaled_train.csv')['N']
train_data_V['d1'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/xgeom_all_scaled_train.csv')['d1']
train_data_V['d2'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/xgeom_all_scaled_train.csv')['d2']
train_data_V['d3'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/xgeom_all_scaled_train.csv')['d3']
train_data_V['r1'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/xgeom_all_scaled_train.csv')['r1']
train_data_V['t1'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/xgeom_all_scaled_train.csv')['t1']



test_data_V = pd.DataFrame()

test_data_V['hysteresis'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/hysteresis_all_scaled_test.csv')['total']
test_data_V['id'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/idiq_all_scaled_test.csv')['id']
test_data_V['iq'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/idiq_all_scaled_test.csv')['iq']
test_data_V['joule'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/joule_all_scaled_test.csv')['total']
test_data_V['speed'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/speed_all_scaled_test.csv')['N']
test_data_V['d1'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/xgeom_all_scaled_test.csv')['d1']
test_data_V['d2'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/xgeom_all_scaled_test.csv')['d2']
test_data_V['d3'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/xgeom_all_scaled_test.csv')['d3']
test_data_V['r1'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/xgeom_all_scaled_test.csv')['r1']
test_data_V['t1'] = pd.read_csv('/mnt/c/prog/ic_motores/IniciacaoCientifica/dataset/V/xgeom_all_scaled_test.csv')['t1']



variable = 'joule'

columns = ['hysteresis', 'joule']

X_train = train_data_V.drop(columns = columns)
y_train = train_data_V[variable]
X_test = test_data_V.drop(columns = columns)
y_test = test_data_V[variable]



model_V = DecisionTreeRegressor()
model_V.fit(X_train, y_train)

predictions = model_V.predict(X_test)
print("Regression tree model results in joule loss for motor V")
print(f"Score: {r2_score(y_test, predictions)}")
print(f"Mean squared error: {mean_squared_error(y_test, predictions)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, predictions)}")



method = 'reg_tree'



newindex = pd.Index([method], name = 'method')
newcolumns = pd.Index(['score', 'mse', 'mape'], name = 'metric')
results = pd.DataFrame(index = newindex,
                       columns = newcolumns)
results.score.reg_tree = r2_score(y_test, predictions)
results.mse.reg_tree = mean_squared_error(y_test, predictions)
results.mape.reg_tree = mean_absolute_percentage_error(y_test, predictions)

results.to_csv("/mnt/c/prog/ic_motores/IniciacaoCientifica/results/V/joule/results_reg_tree.csv")



newcolumns2 = pd.Index(['y_test', 'y_pred'], name = 'data')
data = pd.DataFrame(columns = newcolumns2)
data.y_test = y_test
data.y_pred = predictions

data.to_csv("/mnt/c/prog/ic_motores/IniciacaoCientifica/pred/V/joule/pred_reg_tree.csv")







