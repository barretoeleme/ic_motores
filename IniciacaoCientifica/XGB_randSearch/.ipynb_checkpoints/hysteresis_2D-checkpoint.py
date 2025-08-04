import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold

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



X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Define hyperparameter search space
param_distributions = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'gamma': [0, 0.1, 0.5, 1],
    'max_depth': range(3, 10),
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0.5, 1, 2]
}

# Use RepeatedKFold for more robust CV
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=0)

# RandomizedSearchCV with RepeatedKFold
rand_search = RandomizedSearchCV(
    estimator=XGBRegressor(verbosity=1, random_state=0),
    param_distributions=param_distributions,
    scoring='neg_mean_absolute_error',
    refit=True,
    cv=cv,
    n_iter=30,
    random_state=0,
    n_jobs=-1,
    verbose=2  # More detailed output
)

# Fit with randomized search (no early stopping here yet)
rand_search.fit(X_train_main, y_train_main)

# Print best parameters and score
print("Best Parameters:", rand_search.best_params_)
print("Best CV Score:", rand_search.best_score_)

# Retrieve and refit best estimator using early stopping
best_model = rand_search.best_estimator_
best_model.fit(
    X_train_main, y_train_main,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    early_stopping_rounds=10,
    verbose=True
)

# Predict
y_pred = best_model.predict(X_test)



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

