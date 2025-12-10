import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


MOTOR = "V"
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



model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Linear regression model results in {variable} loss for motor {MOTOR}")
print(f"Score: {r2_score(y_test, y_pred)}")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred)}")



method = "reg_tree"


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