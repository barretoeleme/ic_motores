import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

train_data = pd.DataFrame()

train_data['hysteresis'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/hysteresis_all_scaled_train.csv')['total']
train_data['id'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/idiq_all_scaled_train.csv')['id']
train_data['iq'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/idiq_all_scaled_train.csv')['iq']
train_data['joule'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/joule_all_scaled_train.csv')['total']
train_data['speed'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/speed_all_scaled_train.csv')['N']
train_data['d1'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d1']
train_data['d2'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d2']
train_data['d3'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d3']
train_data['d4'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d4']
train_data['d5'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d5']
train_data['d6'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d6']
train_data['d7'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d7']
train_data['d8'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d8']
train_data['d9'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['d9']
train_data['r1'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['r1']
train_data['t1'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_train.csv')['t1']

test_data = pd.DataFrame()

test_data['hysteresis'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/hysteresis_all_scaled_test.csv')['total']
test_data['id'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/idiq_all_scaled_test.csv')['id']
test_data['iq'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/idiq_all_scaled_test.csv')['iq']
test_data['joule'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/joule_all_scaled_test.csv')['total']
test_data['speed'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/speed_all_scaled_test.csv')['N']
test_data['d1'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d1']
test_data['d2'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d2']
test_data['d3'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d3']
test_data['d4'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d4']
test_data['d5'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d5']
test_data['d6'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d6']
test_data['d7'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d7']
test_data['d8'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d8']
test_data['d9'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['d9']
test_data['r1'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['r1']
test_data['t1'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/2D/xgeom_all_scaled_test.csv')['t1']

variable = 'hysteresis'

columns = ['hysteresis', 'joule']

X_train = train_data.drop(columns = columns)
y_train = train_data[variable]

X_test = test_data.drop(columns = columns)
y_test = test_data[variable]

# parameters = {
#     'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 1],
#     'max_depth' : range(3, 21, 3),
#     'gamma' : [i/10.0 for i in range(0, 5)],
#     # 'colsample_bytree' : [i/10.0 for i in range(3, 10)],
#     'reg_alpha' : [1e-5, 1e-2, 0.1, 1, 10, 100],
#     'reg_lambda' : [1e-5, 1e-2, 0.1, 1, 10, 100]
# }
# scoring = ['neg_mean_absolute_percentage_error']
# # kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 0)

# rand_search = RandomizedSearchCV(estimator = XGBRegressor(), 
#                                  param_distributions = parameters,
#                                  scoring = scoring,
#                                  refit = False, 
#                                  # cv = kfold,
#                                  n_iter = 1,
#                                  n_jobs = -1,
#                                  verbose = 0)
# rand_search.fit(X_train, y_train)
# rand_search.get_params_
# y_pred = rand_search.best_estimator_.predict(X_test)



parameters = {
    'n_estimators' : [50],
    'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 1],
    'gamma' : [i/10.0 for i in range(0, 5)],
    'max_depth' : range(3, 21, 3),
    'reg_alpha' : [0, 0.5, 1, 5],
    'reg_lambda' : [0, 0.5, 1, 5]
}

rand_search = RandomizedSearchCV(estimator = XGBRegressor(random_state = 0), 
                                 param_distributions = parameters,
                                 scoring = 'neg_mean_absolute_percentage_error',
                                 refit = True,
                                 cv = 5, 
                                 n_iter = 1, 
                                 random_state = 0, 
                                 n_jobs = -1,
                                 verbose = 0)
rand_search.fit(X_train, y_train)
y_pred = rand_search.best_estimator_.predict(X_test)



print("Results in Randomized Search for XGBoost in hysteresis loss prediction in motor 2D")
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

results.to_csv("~/ic_motores/IniciacaoCientifica/results/2D/hysteresis/results_xgb_rand.csv")


newcolumns2 = pd.Index(['y_test', 'y_pred'], name = 'data')
data = pd.DataFrame(columns = newcolumns2)
data.y_test = y_test
data.y_pred = y_pred

data.to_csv("~/ic_motores/IniciacaoCientifica/pred/2D/hysteresis/pred_xgb_rand.csv")

