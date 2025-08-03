import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

train_data = pd.DataFrame()

train_data['hysteresis'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/hysteresis_all_scaled_train.csv')['total']
train_data['id'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/idiq_all_scaled_train.csv')['id']
train_data['iq'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/idiq_all_scaled_train.csv')['iq']
train_data['joule'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/joule_all_scaled_train.csv')['total']
train_data['speed'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/speed_all_scaled_train.csv')['N']
train_data['d1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d1']
train_data['d2'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d2']
train_data['d3'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d3']
train_data['d4'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d4']
train_data['d5'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d5']
train_data['d6'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d6']
train_data['d7'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d7']
train_data['d8'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d8']
train_data['d9'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d9']
train_data['r1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['r1']
train_data['t1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['t1']

test_data = pd.DataFrame()

test_data['hysteresis'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/hysteresis_all_scaled_test.csv')['total']
test_data['id'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/idiq_all_scaled_test.csv')['id']
test_data['iq'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/idiq_all_scaled_test.csv')['iq']
test_data['joule'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/joule_all_scaled_test.csv')['total']
test_data['speed'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/speed_all_scaled_test.csv')['N']
test_data['d1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d1']
test_data['d2'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d2']
test_data['d3'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d3']
test_data['d4'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d4']
test_data['d5'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d5']
test_data['d6'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d6']
test_data['d7'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d7']
test_data['d8'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d8']
test_data['d9'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d9']
test_data['r1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['r1']
test_data['t1'] = pd.read_csv('/mnt/c/prog/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['t1']

variable = 'joule'

columns = ['hysteresis', 'joule']

X_train = train_data.drop(columns = columns)
y_train = train_data[variable]

X_test = test_data.drop(columns = columns)
y_test = test_data[variable]

parameters = {
    'n_estimators' : np.random.poisson(lam = 5, size = 100),
    'learning_rate' : np.random.uniform(size = 100),
    'max_depth' : np.random.poisson(lam = 5, size = 100),
    'reg_alpha' : np.random.uniform(size = 100),
    'reg_lambda' : np.random.uniform(size = 100)
}

rand_search = RandomizedSearchCV(XGBRegressor(random_state = 42), parameters,
                                 cv = 5, n_iter = 200, random_state = 42, n_jobs = -1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
y_pred = rand_search.best_estimator_.predict(X_test)


print("Results in Randomized Search for XGBoost in joule loss prediction in motor 2D")
print(f"Coefficient of determination: {r2_score(y_test, y_pred)}")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"Mean absolute percentage error: {mean_absolute_percentage_error(y_test, y_pred)}")



method = 'xgb_rand'

newindex = pd.Index([method], name = 'method')
newcolumns = pd.Index(['score', 'mse', 'mape'], name = 'metric')
results = pd.DataFrame(index = newindex,
                       columns = newcolumns)
results.score.xgb_rand = r2_score(y_test, y_pred)
results.mse.xgb_rand = mean_squared_error(y_test, y_pred)
results.mape.xgb_rand = mean_absolute_percentage_error(y_test, y_pred)

results.to_csv("/mnt/c/prog/IniciacaoCientifica/results/2D/joule/results_xgb_rand.csv")


newcolumns2 = pd.Index(['y_test', 'y_pred'], name = 'data')
data = pd.DataFrame(columns = newcolumns2)
data.y_test = y_test
data.y_pred = y_pred

data.to_csv("/mnt/c/prog/IniciacaoCientifica/pred/2D/joule/pred_xgb_rand.csv")

