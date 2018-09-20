import pandas as pd
import numpy as np
import operator
from timeit import default_timer as timer
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reports.plots import feature_importance_plot

## data parameter
window_size = 10
normalise = True
test_ratio = 0.2

data = pd.read_csv('data/nasdaq100_padding.csv') # (40560, 82), index = range(0,40560), last columns is 'NDX'

from data.data_processor import DataForModel
data_processing = DataForModel(data, test_ratio)
start = timer()
X_train, Y_train = data_processing.get_train_batch(window_size, normalise) #(32438, 9, 82) (32438, 1)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2])) #(32438, 738)
x_test, y_test = data_processing.get_test_batch(window_size, normalise)  # (8102, 9, 82)(8102, 1)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2])) # (8102, 738)
elapsed_time = timer()-start
print('processing train and test data by batch method took %d s' %elapsed_time) # 20s

start=timer()
xgb_params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',  # regression task
    'subsample': 0.80,  # 80% of data to grow trees and prevent overfitting
    'colsample_bytree': 0.85,  # 85% of features used
    'eta': 0.1,
    'max_depth': 10,
    'seed': 42}
boosting_iterations = 100
dtrain = xgb.DMatrix(X_train, Y_train)
dtest = xgb.DMatrix(x_test, y_test)
watchlist = [(dtrain, 'train'), (dtest, 'validate')]
xgb_model = xgb.train(xgb_params, dtrain, boosting_iterations, evals=watchlist, verbose_eval=True)
importance = xgb_model.get_fscore() #637
importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
feature_importance_plot(importance_sorted, 'feature_importance')

elapsed_time = timer() - start
y_pred_test = xgb_model.predict(xgb.DMatrix(x_test))
print('explained_variance_score:', explained_variance_score(y_pred_test, y_test))
print('final mse: ', mean_squared_error(y_pred_test,y_test))
print('xgboost training took {time}s'.format(time=elapsed_time))
print('y_pred_test: ', y_pred_test)

y_pred_train = xgb_model.predict(xgb.DMatrix(X_train))
#y_pred = np.concatenate((y_pred_train, y_pred_test))
y_true = np.concatenate((Y_train, y_test))
plt.figure()
plt.plot(range(1,1+len(y_true)), y_true, label='true')
plt.plot(range(1, len(y_pred_train)+1), y_pred_train, label = 'predicted_train')
plt.plot(range(len(y_pred_train)+1, len(y_true)+1), y_pred_test, label = 'predicted_test' )
plt.legend()
plt.savefig('reports/y_xgb.png')
plt.show()


'''
## grid search
x_train, x_val, y_train, y_val = train_test_split(X_train,Y_train, test_size=test_ratio, random_state=42)
dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_val,y_val)
dtest = xgb.DMatrix(x_test, y_test)
watchlist = [(dtrain, 'train'), (dval, 'validate')]

# Grid Search
params_sk = {
    'objective': 'reg:linear',
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'seed': 42}

skrg = XGBRegressor(**params_sk)
skrg.fit(x_train,y_train)

params_grid = {"n_estimators": st.randint(100, 500),
                               "colsample_bytree": st.beta(10, 1),
                               "subsample": st.beta(10, 1),
                               "gamma": st.uniform(0, 10),
                               'reg_alpha': st.expon(0, 50),
                               "min_child_weight": st.expon(0, 50),
                              "learning_rate": st.uniform(0.06, 0.12),
               'max_depth': st.randint(6, 30)
               }

search_sk = RandomizedSearchCV(skrg, params_grid, cv=5, random_state=1, n_iter=20)  # 5 fold cross validation
'''



'''
## batch training
batch_size = 1000
train_gen = data_processing.generate_train_batch(window_size, batch_size, normalise)

model_gb = None
mse = []
for batch in train_gen:
    x = batch[0]
    x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
    y = batch[1]
    model_gb = xgb.train(xgb_params, dtrain = xgb.DMatrix(x,y), xgb_model = model_gb)
    y_pred_temp = model_gb.predict(xgb.DMatrix(x_test))
    mse.append(mean_squared_error(y_pred_temp, y_test))
'''


'''
from predictionmodel.lstm_pytorch import LSTM
model = LSTM(data_processing.data_train.shape[1], hidden_size=3)
model.train(train_gen, configs['training']['epochs'])

x_test, y_test = data_processing.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    
plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    # plot_results(predictions, y_test)

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.savefig('reports/prediction_result.png')
    plt.show()
    '''
