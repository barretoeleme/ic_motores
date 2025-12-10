import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, train_test_split



MOTOR = "2D"
PATH = f"../dataset/{MOTOR}/"
TRAIN_FILE = "_all_scaled_train.csv"
TEST_FILE = "_all_scaled_test.csv"

train_data = pd.DataFrame()

train_data = pd.concat([train_data, pd.read_csv(f'{PATH}idiq{TRAIN_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
train_data['speed'] = pd.read_csv(f'{PATH}speed{TRAIN_FILE}')['N']
train_data = pd.concat([train_data, pd.read_csv(f'{PATH}xgeom{TRAIN_FILE}').drop(columns = "Unnamed: 0")], axis = 1)

train_data['hysteresis'] = pd.read_csv(f'{PATH}hysteresis{TRAIN_FILE}')['total']
train_data['joule'] = pd.read_csv(f'{PATH}joule{TRAIN_FILE}')['total']


test_data = pd.DataFrame()

test_data = pd.concat([test_data, pd.read_csv(f'{PATH}idiq{TEST_FILE}').drop(columns = "Unnamed: 0")], axis = 1)

test_data['speed'] = pd.read_csv(f'{PATH}speed{TEST_FILE}')['N']
test_data = pd.concat([test_data, pd.read_csv(f'{PATH}xgeom{TEST_FILE}').drop(columns = "Unnamed: 0")], axis = 1)

test_data['hysteresis'] = pd.read_csv(f'{PATH}hysteresis{TEST_FILE}')['total']
test_data['joule'] = pd.read_csv(f'{PATH}joule{TEST_FILE}')['total']



variable = 'hysteresis'

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
    verbose=0
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
    verbose=0  # Show progress every 100 iterations
)

# Predict
y_pred = best_model.predict(X_test)

print(f"Category Boost with Randomized Search model results in {variable} loss for motor {MOTOR}")
print(f"Score: {r2_score(y_test, y_pred)}")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred)}")



method = "cat_rand"


newindex = pd.Index([method], name = 'method')
newcolumns = pd.Index(['score', 'mse', 'mape'], name = 'metric')
results = pd.DataFrame(index = newindex,
                       columns = newcolumns)
results.loc[method, "score"] = r2_score(y_test, y_pred)
results.loc[method, "mse"] = mean_squared_error(y_test, y_pred)
results.loc[method, "mape"] = mean_absolute_percentage_error(y_test, y_pred)

results.to_csv(f"../results/{MOTOR}/{variable}/results_{method}.csv")



newcolumns2 = pd.Index(['y_test', 'y_pred'], name = 'data')
data = pd.DataFrame(columns = newcolumns2)
data.y_test = y_test
data.y_pred = y_pred

data.to_csv(f"../pred/{MOTOR}/{variable}/pred_{method}.csv")

