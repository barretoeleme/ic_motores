import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, train_test_split

train_data = pd.DataFrame()

train_data['hysteresis'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/hysteresis_all_scaled_train.csv')['total']
train_data['id'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/idiq_all_scaled_train.csv')['id']
train_data['iq'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/idiq_all_scaled_train.csv')['iq']
train_data['joule'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/joule_all_scaled_train.csv')['total']
train_data['speed'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/speed_all_scaled_train.csv')['N']
train_data['d1'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d1']
train_data['d2'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d2']
train_data['d3'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d3']
train_data['d4'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d4']
train_data['d5'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d5']
train_data['d6'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d6']
train_data['d7'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d7']
train_data['d8'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_train.csv')['d8']


test_data = pd.DataFrame()

test_data['hysteresis'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/hysteresis_all_scaled_test.csv')['total']
test_data['id'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/idiq_all_scaled_test.csv')['id']
test_data['iq'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/idiq_all_scaled_test.csv')['iq']
test_data['joule'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/joule_all_scaled_test.csv')['total']
test_data['speed'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/speed_all_scaled_test.csv')['N']
test_data['d1'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d1']
test_data['d2'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d2']
test_data['d3'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d3']
test_data['d4'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d4']
test_data['d5'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d5']
test_data['d6'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d6']
test_data['d7'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d7']
test_data['d8'] = pd.read_csv('~/ic_motores/IniciacaoCientifica/dataset/Nabla/xgeom_all_scaled_test.csv')['d8']

variable = 'joule'

columns = ['hysteresis', 'joule']

X_train = train_data.drop(columns = columns)
y_train = train_data[variable]

X_test = test_data.drop(columns = columns)
y_test = test_data[variable]


X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Define hyperparameter space for CatBoost
param_distributions = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'random_strength': [1, 5, 10],
    'bagging_temperature': [0.1, 0.5, 1]
}

# Use RepeatedKFold
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=0)

# RandomizedSearchCV with CatBoost
rand_search = RandomizedSearchCV(
    estimator=CatBoostRegressor(
        silent=True,      # turn off internal logging
        random_state=0
    ),
    param_distributions=param_distributions,
    scoring='neg_mean_absolute_percentage_error',
    refit=True,
    cv=cv,
    n_iter=30,
    random_state=0,
    n_jobs=-1,
    verbose=2
)

# Fit search
rand_search.fit(X_train_main, y_train_main)

# Print best params
print("Best Parameters:", rand_search.best_params_)
print("Best CV Score:", rand_search.best_score_)

# Refit best model with early stopping
best_model = rand_search.best_estimator_

best_model.fit(
    X_train_main, y_train_main,
    eval_set=(X_val, y_val),
    early_stopping_rounds=10,
    use_best_model=True,
    verbose=100  # Show progress every 100 iterations
)

# Predict
y_pred = best_model.predict(X_test)



print("Results in Randomized Search for CatBoost in joule loss prediction in motor Nabla")
print(f"Coefficient of determination: {r2_score(y_test, y_pred)}")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"Mean absolute percentage error: {mean_absolute_percentage_error(y_test, y_pred)}")



method = 'cat_rand'

newindex = pd.Index([method], name = 'method')
newcolumns = pd.Index(['score', 'mse', 'mape'], name = 'metric')
results = pd.DataFrame(index = newindex,
                       columns = newcolumns)
results.score.cat_rand = r2_score(y_test, y_pred)
results.mse.cat_rand = mean_squared_error(y_test, y_pred)
results.mape.cat_rand = mean_absolute_percentage_error(y_test, y_pred)

results.to_csv("~/ic_motores/IniciacaoCientifica/results/Nabla/joule/results_cat_rand.csv")


newcolumns2 = pd.Index(['y_test', 'y_pred'], name = 'data')
data = pd.DataFrame(columns = newcolumns2)
data.y_test = y_test
data.y_pred = y_pred

data.to_csv("~/ic_motores/IniciacaoCientifica/pred/Nabla/joule/pred_cat_rand.csv")

