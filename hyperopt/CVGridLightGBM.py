import optuna  # pip install optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
rand_seed = 21

param_grid = {
"n_estimators": [200, 300, 400, 500, 600, 700, 800, 900, 1000],
"learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
"max_depth": [8, 12, 15, 17, 20],
}

cat_cols_sig = ['OCCP', 'SCHL', 'ST', 'JWTRNS', 'DRAT', 'COW', 'SEX', \
       'RELSHIPP', 'POBP', 'ENG', 'MAR', 'RAC1P']
numeric_cols = ['WKHP', 'AGEP', 'PINCP']

df_all = pd.read_pickle("./data_frames/dense_acs_mm_notoh.pkl")
df_train, df_test = train_test_split(df_all, test_size=0.2, random_state = rand_seed)

X_train = df_train.drop(['PINCP'], axis=1, inplace = False)
y_train = df_train['PINCP']

grid = GridSearchCV(estimator = lgbm.LGBMRegressor(random_state=rand_seed), 
                    param_grid = param_grid, scoring='r2', cv=5, n_jobs = 4, verbose = 10)

fit_params = {'categorical_feature': cat_cols_sig}
grid.fit(X_train, y_train, **fit_params)

print("Best: %f using %s" % (grid.best_score_, grid.best_params_))

#Best: 0.572793 using {'learning_rate': 0.05, 'max_depth': 12, 'n_estimators': 1000}

'''
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 97/225] END learning_rate=0.1, max_depth=8, n_estimators=800;, score=0.571 total time=  32.5s
[CV 3/5; 98/225] START learning_rate=0.1, max_depth=8, n_estimators=900.........
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 97/225] END learning_rate=0.1, max_depth=8, n_estimators=800;, score=0.573 total time=  31.9s
[CV 4/5; 98/225] START learning_rate=0.1, max_depth=8, n_estimators=900.........
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 98/225] END learning_rate=0.1, max_depth=8, n_estimators=900;, score=0.572 total time=  39.7s
[CV 5/5; 98/225] START learning_rate=0.1, max_depth=8, n_estimators=900.........
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 98/225] END learning_rate=0.1, max_depth=8, n_estimators=900;, score=0.572 total time=  40.0s
[CV 1/5; 99/225] START learning_rate=0.1, max_depth=8, n_estimators=1000........
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 98/225] END learning_rate=0.1, max_depth=8, n_estimators=900;, score=0.569 total time=  39.5s
[CV 2/5; 99/225] START learning_rate=0.1, max_depth=8, n_estimators=1000........
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 98/225] END learning_rate=0.1, max_depth=8, n_estimators=900;, score=0.571 total time=  41.4s
[CV 3/5; 99/225] START learning_rate=0.1, max_depth=8, n_estimators=1000........
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 98/225] END learning_rate=0.1, max_depth=8, n_estimators=900;, score=0.573 total time=  35.8s
[CV 4/5; 99/225] START learning_rate=0.1, max_depth=8, n_estimators=1000........
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 99/225] END learning_rate=0.1, max_depth=8, n_estimators=1000;, score=0.572 total time=  41.0s
[CV 5/5; 99/225] START learning_rate=0.1, max_depth=8, n_estimators=1000........
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 99/225] END learning_rate=0.1, max_depth=8, n_estimators=1000;, score=0.572 total time=  40.8s
[CV 1/5; 100/225] START learning_rate=0.1, max_depth=12, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 99/225] END learning_rate=0.1, max_depth=8, n_estimators=1000;, score=0.569 total time=  40.1s
[CV 2/5; 100/225] START learning_rate=0.1, max_depth=12, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 100/225] END learning_rate=0.1, max_depth=12, n_estimators=200;, score=0.571 total time=  12.3s
[CV 3/5; 100/225] START learning_rate=0.1, max_depth=12, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 100/225] END learning_rate=0.1, max_depth=12, n_estimators=200;, score=0.571 total time=  11.1s
[CV 4/5; 100/225] START learning_rate=0.1, max_depth=12, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 100/225] END learning_rate=0.1, max_depth=12, n_estimators=200;, score=0.568 total time=  12.6s
[CV 5/5; 100/225] START learning_rate=0.1, max_depth=12, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 99/225] END learning_rate=0.1, max_depth=8, n_estimators=1000;, score=0.571 total time=  41.6s
[CV 1/5; 101/225] START learning_rate=0.1, max_depth=12, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 100/225] END learning_rate=0.1, max_depth=12, n_estimators=200;, score=0.569 total time=  10.9s
[CV 2/5; 101/225] START learning_rate=0.1, max_depth=12, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 100/225] END learning_rate=0.1, max_depth=12, n_estimators=200;, score=0.571 total time=  11.1s
[CV 3/5; 101/225] START learning_rate=0.1, max_depth=12, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 99/225] END learning_rate=0.1, max_depth=8, n_estimators=1000;, score=0.572 total time=  45.6s
[CV 4/5; 101/225] START learning_rate=0.1, max_depth=12, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 101/225] END learning_rate=0.1, max_depth=12, n_estimators=300;, score=0.572 total time=  13.2s
[CV 5/5; 101/225] START learning_rate=0.1, max_depth=12, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 101/225] END learning_rate=0.1, max_depth=12, n_estimators=300;, score=0.569 total time=  13.2s
[CV 1/5; 102/225] START learning_rate=0.1, max_depth=12, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 101/225] END learning_rate=0.1, max_depth=12, n_estimators=300;, score=0.570 total time=  13.8s
[CV 2/5; 102/225] START learning_rate=0.1, max_depth=12, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 101/225] END learning_rate=0.1, max_depth=12, n_estimators=300;, score=0.572 total time=  18.0s
[CV 3/5; 102/225] START learning_rate=0.1, max_depth=12, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 101/225] END learning_rate=0.1, max_depth=12, n_estimators=300;, score=0.572 total time=  15.5s
[CV 4/5; 102/225] START learning_rate=0.1, max_depth=12, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 102/225] END learning_rate=0.1, max_depth=12, n_estimators=400;, score=0.572 total time=  18.3s
[CV 5/5; 102/225] START learning_rate=0.1, max_depth=12, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 102/225] END learning_rate=0.1, max_depth=12, n_estimators=400;, score=0.569 total time=  18.7s
[CV 1/5; 103/225] START learning_rate=0.1, max_depth=12, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 102/225] END learning_rate=0.1, max_depth=12, n_estimators=400;, score=0.573 total time=  20.4s
[CV 2/5; 103/225] START learning_rate=0.1, max_depth=12, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 102/225] END learning_rate=0.1, max_depth=12, n_estimators=400;, score=0.571 total time=  17.3s
[CV 3/5; 103/225] START learning_rate=0.1, max_depth=12, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 102/225] END learning_rate=0.1, max_depth=12, n_estimators=400;, score=0.573 total time=  17.7s
[CV 4/5; 103/225] START learning_rate=0.1, max_depth=12, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 103/225] END learning_rate=0.1, max_depth=12, n_estimators=500;, score=0.573 total time=  22.7s
[CV 5/5; 103/225] START learning_rate=0.1, max_depth=12, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 103/225] END learning_rate=0.1, max_depth=12, n_estimators=500;, score=0.570 total time=  21.2s
[CV 1/5; 104/225] START learning_rate=0.1, max_depth=12, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 103/225] END learning_rate=0.1, max_depth=12, n_estimators=500;, score=0.573 total time=  23.0s
[CV 2/5; 104/225] START learning_rate=0.1, max_depth=12, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 103/225] END learning_rate=0.1, max_depth=12, n_estimators=500;, score=0.571 total time=  20.1s
[CV 3/5; 104/225] START learning_rate=0.1, max_depth=12, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 103/225] END learning_rate=0.1, max_depth=12, n_estimators=500;, score=0.573 total time=  22.7s
[CV 4/5; 104/225] START learning_rate=0.1, max_depth=12, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 104/225] END learning_rate=0.1, max_depth=12, n_estimators=600;, score=0.572 total time=  22.8s
[CV 5/5; 104/225] START learning_rate=0.1, max_depth=12, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 104/225] END learning_rate=0.1, max_depth=12, n_estimators=600;, score=0.573 total time=  24.5s
[CV 1/5; 105/225] START learning_rate=0.1, max_depth=12, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 104/225] END learning_rate=0.1, max_depth=12, n_estimators=600;, score=0.570 total time=  29.3s
[CV 2/5; 105/225] START learning_rate=0.1, max_depth=12, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 104/225] END learning_rate=0.1, max_depth=12, n_estimators=600;, score=0.573 total time=  25.1s
[CV 3/5; 105/225] START learning_rate=0.1, max_depth=12, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 104/225] END learning_rate=0.1, max_depth=12, n_estimators=600;, score=0.571 total time=  27.9s
[CV 4/5; 105/225] START learning_rate=0.1, max_depth=12, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 105/225] END learning_rate=0.1, max_depth=12, n_estimators=700;, score=0.572 total time=  28.4s
[CV 5/5; 105/225] START learning_rate=0.1, max_depth=12, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 105/225] END learning_rate=0.1, max_depth=12, n_estimators=700;, score=0.573 total time=  27.6s
[CV 1/5; 106/225] START learning_rate=0.1, max_depth=12, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 105/225] END learning_rate=0.1, max_depth=12, n_estimators=700;, score=0.570 total time=  28.2s
[CV 2/5; 106/225] START learning_rate=0.1, max_depth=12, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 105/225] END learning_rate=0.1, max_depth=12, n_estimators=700;, score=0.571 total time=  29.0s
[CV 3/5; 106/225] START learning_rate=0.1, max_depth=12, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 105/225] END learning_rate=0.1, max_depth=12, n_estimators=700;, score=0.573 total time=  30.1s
[CV 4/5; 106/225] START learning_rate=0.1, max_depth=12, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 106/225] END learning_rate=0.1, max_depth=12, n_estimators=800;, score=0.572 total time=  32.9s
[CV 5/5; 106/225] START learning_rate=0.1, max_depth=12, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 106/225] END learning_rate=0.1, max_depth=12, n_estimators=800;, score=0.573 total time=  31.9s
[CV 1/5; 107/225] START learning_rate=0.1, max_depth=12, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 106/225] END learning_rate=0.1, max_depth=12, n_estimators=800;, score=0.570 total time=  35.5s
[CV 2/5; 107/225] START learning_rate=0.1, max_depth=12, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 106/225] END learning_rate=0.1, max_depth=12, n_estimators=800;, score=0.571 total time=  34.5s
[CV 3/5; 107/225] START learning_rate=0.1, max_depth=12, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 106/225] END learning_rate=0.1, max_depth=12, n_estimators=800;, score=0.573 total time=  31.2s
[CV 4/5; 107/225] START learning_rate=0.1, max_depth=12, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 107/225] END learning_rate=0.1, max_depth=12, n_estimators=900;, score=0.572 total time=  36.4s
[CV 5/5; 107/225] START learning_rate=0.1, max_depth=12, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 107/225] END learning_rate=0.1, max_depth=12, n_estimators=900;, score=0.573 total time=  35.4s
[CV 1/5; 108/225] START learning_rate=0.1, max_depth=12, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 107/225] END learning_rate=0.1, max_depth=12, n_estimators=900;, score=0.569 total time=  34.3s
[CV 2/5; 108/225] START learning_rate=0.1, max_depth=12, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 107/225] END learning_rate=0.1, max_depth=12, n_estimators=900;, score=0.571 total time=  36.5s
[CV 3/5; 108/225] START learning_rate=0.1, max_depth=12, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 107/225] END learning_rate=0.1, max_depth=12, n_estimators=900;, score=0.573 total time=  39.8s
[CV 4/5; 108/225] START learning_rate=0.1, max_depth=12, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 108/225] END learning_rate=0.1, max_depth=12, n_estimators=1000;, score=0.572 total time=  38.5s
[CV 5/5; 108/225] START learning_rate=0.1, max_depth=12, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 108/225] END learning_rate=0.1, max_depth=12, n_estimators=1000;, score=0.572 total time=  42.3s
[CV 1/5; 109/225] START learning_rate=0.1, max_depth=15, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 109/225] END learning_rate=0.1, max_depth=15, n_estimators=200;, score=0.571 total time=  12.2s
[CV 2/5; 109/225] START learning_rate=0.1, max_depth=15, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 108/225] END learning_rate=0.1, max_depth=12, n_estimators=1000;, score=0.569 total time=  38.8s
[CV 3/5; 109/225] START learning_rate=0.1, max_depth=15, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 109/225] END learning_rate=0.1, max_depth=15, n_estimators=200;, score=0.572 total time=  11.4s
[CV 4/5; 109/225] START learning_rate=0.1, max_depth=15, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 109/225] END learning_rate=0.1, max_depth=15, n_estimators=200;, score=0.568 total time=  11.5s
[CV 5/5; 109/225] START learning_rate=0.1, max_depth=15, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 108/225] END learning_rate=0.1, max_depth=12, n_estimators=1000;, score=0.570 total time=  39.8s
[CV 1/5; 110/225] START learning_rate=0.1, max_depth=15, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 109/225] END learning_rate=0.1, max_depth=15, n_estimators=200;, score=0.569 total time=  11.7s
[CV 2/5; 110/225] START learning_rate=0.1, max_depth=15, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 109/225] END learning_rate=0.1, max_depth=15, n_estimators=200;, score=0.571 total time=  11.0s
[CV 3/5; 110/225] START learning_rate=0.1, max_depth=15, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 108/225] END learning_rate=0.1, max_depth=12, n_estimators=1000;, score=0.573 total time=  42.0s
[CV 4/5; 110/225] START learning_rate=0.1, max_depth=15, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 110/225] END learning_rate=0.1, max_depth=15, n_estimators=300;, score=0.571 total time=  13.4s
[CV 5/5; 110/225] START learning_rate=0.1, max_depth=15, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 110/225] END learning_rate=0.1, max_depth=15, n_estimators=300;, score=0.572 total time=  14.5s
[CV 1/5; 111/225] START learning_rate=0.1, max_depth=15, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 110/225] END learning_rate=0.1, max_depth=15, n_estimators=300;, score=0.569 total time=  16.4s
[CV 2/5; 111/225] START learning_rate=0.1, max_depth=15, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 110/225] END learning_rate=0.1, max_depth=15, n_estimators=300;, score=0.570 total time=  16.3s
[CV 3/5; 111/225] START learning_rate=0.1, max_depth=15, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 110/225] END learning_rate=0.1, max_depth=15, n_estimators=300;, score=0.572 total time=  13.6s
[CV 4/5; 111/225] START learning_rate=0.1, max_depth=15, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 111/225] END learning_rate=0.1, max_depth=15, n_estimators=400;, score=0.572 total time=  15.8s
[CV 5/5; 111/225] START learning_rate=0.1, max_depth=15, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 111/225] END learning_rate=0.1, max_depth=15, n_estimators=400;, score=0.573 total time=  16.6s
[CV 1/5; 112/225] START learning_rate=0.1, max_depth=15, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 111/225] END learning_rate=0.1, max_depth=15, n_estimators=400;, score=0.570 total time=  17.9s
[CV 2/5; 112/225] START learning_rate=0.1, max_depth=15, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 111/225] END learning_rate=0.1, max_depth=15, n_estimators=400;, score=0.571 total time=  20.7s
[CV 3/5; 112/225] START learning_rate=0.1, max_depth=15, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 111/225] END learning_rate=0.1, max_depth=15, n_estimators=400;, score=0.573 total time=  17.6s
[CV 4/5; 112/225] START learning_rate=0.1, max_depth=15, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 112/225] END learning_rate=0.1, max_depth=15, n_estimators=500;, score=0.572 total time=  19.2s
[CV 5/5; 112/225] START learning_rate=0.1, max_depth=15, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 112/225] END learning_rate=0.1, max_depth=15, n_estimators=500;, score=0.573 total time=  19.4s
[CV 1/5; 113/225] START learning_rate=0.1, max_depth=15, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 112/225] END learning_rate=0.1, max_depth=15, n_estimators=500;, score=0.570 total time=  20.6s
[CV 2/5; 113/225] START learning_rate=0.1, max_depth=15, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 112/225] END learning_rate=0.1, max_depth=15, n_estimators=500;, score=0.571 total time=  20.2s
[CV 3/5; 113/225] START learning_rate=0.1, max_depth=15, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 112/225] END learning_rate=0.1, max_depth=15, n_estimators=500;, score=0.573 total time=  21.5s
[CV 4/5; 113/225] START learning_rate=0.1, max_depth=15, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 113/225] END learning_rate=0.1, max_depth=15, n_estimators=600;, score=0.572 total time=  24.1s
[CV 5/5; 113/225] START learning_rate=0.1, max_depth=15, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 113/225] END learning_rate=0.1, max_depth=15, n_estimators=600;, score=0.573 total time=  27.6s
[CV 1/5; 114/225] START learning_rate=0.1, max_depth=15, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 113/225] END learning_rate=0.1, max_depth=15, n_estimators=600;, score=0.570 total time=  24.9s
[CV 2/5; 114/225] START learning_rate=0.1, max_depth=15, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 113/225] END learning_rate=0.1, max_depth=15, n_estimators=600;, score=0.571 total time=  23.6s
[CV 3/5; 114/225] START learning_rate=0.1, max_depth=15, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 113/225] END learning_rate=0.1, max_depth=15, n_estimators=600;, score=0.573 total time=  21.9s
[CV 4/5; 114/225] START learning_rate=0.1, max_depth=15, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 114/225] END learning_rate=0.1, max_depth=15, n_estimators=700;, score=0.572 total time=  28.3s
[CV 5/5; 114/225] START learning_rate=0.1, max_depth=15, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 114/225] END learning_rate=0.1, max_depth=15, n_estimators=700;, score=0.573 total time=  31.6s
[CV 1/5; 115/225] START learning_rate=0.1, max_depth=15, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 114/225] END learning_rate=0.1, max_depth=15, n_estimators=700;, score=0.570 total time=  30.9s
[CV 2/5; 115/225] START learning_rate=0.1, max_depth=15, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 114/225] END learning_rate=0.1, max_depth=15, n_estimators=700;, score=0.571 total time=  28.9s
[CV 3/5; 115/225] START learning_rate=0.1, max_depth=15, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 114/225] END learning_rate=0.1, max_depth=15, n_estimators=700;, score=0.573 total time=  26.5s
[CV 4/5; 115/225] START learning_rate=0.1, max_depth=15, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 115/225] END learning_rate=0.1, max_depth=15, n_estimators=800;, score=0.572 total time=  31.3s
[CV 5/5; 115/225] START learning_rate=0.1, max_depth=15, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 115/225] END learning_rate=0.1, max_depth=15, n_estimators=800;, score=0.573 total time=  32.5s
[CV 1/5; 116/225] START learning_rate=0.1, max_depth=15, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 115/225] END learning_rate=0.1, max_depth=15, n_estimators=800;, score=0.570 total time=  32.0s
[CV 2/5; 116/225] START learning_rate=0.1, max_depth=15, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 115/225] END learning_rate=0.1, max_depth=15, n_estimators=800;, score=0.571 total time=  30.1s
[CV 3/5; 116/225] START learning_rate=0.1, max_depth=15, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 115/225] END learning_rate=0.1, max_depth=15, n_estimators=800;, score=0.573 total time=  30.5s
[CV 4/5; 116/225] START learning_rate=0.1, max_depth=15, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 116/225] END learning_rate=0.1, max_depth=15, n_estimators=900;, score=0.572 total time=  35.3s
[CV 5/5; 116/225] START learning_rate=0.1, max_depth=15, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 116/225] END learning_rate=0.1, max_depth=15, n_estimators=900;, score=0.572 total time=  35.1s
[CV 1/5; 117/225] START learning_rate=0.1, max_depth=15, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 116/225] END learning_rate=0.1, max_depth=15, n_estimators=900;, score=0.570 total time=  35.8s
[CV 2/5; 117/225] START learning_rate=0.1, max_depth=15, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 116/225] END learning_rate=0.1, max_depth=15, n_estimators=900;, score=0.571 total time=  32.7s
[CV 3/5; 117/225] START learning_rate=0.1, max_depth=15, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 117/225] END learning_rate=0.1, max_depth=15, n_estimators=1000;, score=0.572 total time=  36.9s
[CV 4/5; 117/225] START learning_rate=0.1, max_depth=15, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 116/225] END learning_rate=0.1, max_depth=15, n_estimators=900;, score=0.573 total time=  40.1s
[CV 5/5; 117/225] START learning_rate=0.1, max_depth=15, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 117/225] END learning_rate=0.1, max_depth=15, n_estimators=1000;, score=0.572 total time=  36.5s
[CV 1/5; 118/225] START learning_rate=0.1, max_depth=17, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 118/225] END learning_rate=0.1, max_depth=17, n_estimators=200;, score=0.571 total time=  11.1s
[CV 2/5; 118/225] START learning_rate=0.1, max_depth=17, n_estimators=200.......
[CV 3/5; 117/225] END learning_rate=0.1, max_depth=15, n_estimators=1000;, score=0.570 total time=  39.2s
[CV 3/5; 118/225] START learning_rate=0.1, max_depth=17, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 118/225] END learning_rate=0.1, max_depth=17, n_estimators=200;, score=0.572 total time=  11.0s
[CV 4/5; 118/225] START learning_rate=0.1, max_depth=17, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 118/225] END learning_rate=0.1, max_depth=17, n_estimators=200;, score=0.568 total time=  11.0s
[CV 5/5; 118/225] START learning_rate=0.1, max_depth=17, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 117/225] END learning_rate=0.1, max_depth=15, n_estimators=1000;, score=0.572 total time=  38.3s
[CV 1/5; 119/225] START learning_rate=0.1, max_depth=17, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 117/225] END learning_rate=0.1, max_depth=15, n_estimators=1000;, score=0.571 total time=  42.7s
[CV 2/5; 119/225] START learning_rate=0.1, max_depth=17, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 118/225] END learning_rate=0.1, max_depth=17, n_estimators=200;, score=0.570 total time=  10.1s
[CV 3/5; 119/225] START learning_rate=0.1, max_depth=17, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 118/225] END learning_rate=0.1, max_depth=17, n_estimators=200;, score=0.571 total time=  10.2s
[CV 4/5; 119/225] START learning_rate=0.1, max_depth=17, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 119/225] END learning_rate=0.1, max_depth=17, n_estimators=300;, score=0.572 total time=  13.1s
[CV 5/5; 119/225] START learning_rate=0.1, max_depth=17, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 119/225] END learning_rate=0.1, max_depth=17, n_estimators=300;, score=0.572 total time=  13.8s
[CV 1/5; 120/225] START learning_rate=0.1, max_depth=17, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 119/225] END learning_rate=0.1, max_depth=17, n_estimators=300;, score=0.569 total time=  14.5s
[CV 2/5; 120/225] START learning_rate=0.1, max_depth=17, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 119/225] END learning_rate=0.1, max_depth=17, n_estimators=300;, score=0.571 total time=  18.0s
[CV 3/5; 120/225] START learning_rate=0.1, max_depth=17, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 119/225] END learning_rate=0.1, max_depth=17, n_estimators=300;, score=0.572 total time=  16.7s
[CV 4/5; 120/225] START learning_rate=0.1, max_depth=17, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 120/225] END learning_rate=0.1, max_depth=17, n_estimators=400;, score=0.572 total time=  18.8s
[CV 5/5; 120/225] START learning_rate=0.1, max_depth=17, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 120/225] END learning_rate=0.1, max_depth=17, n_estimators=400;, score=0.573 total time=  19.6s
[CV 1/5; 121/225] START learning_rate=0.1, max_depth=17, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 120/225] END learning_rate=0.1, max_depth=17, n_estimators=400;, score=0.569 total time=  20.1s
[CV 2/5; 121/225] START learning_rate=0.1, max_depth=17, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 120/225] END learning_rate=0.1, max_depth=17, n_estimators=400;, score=0.571 total time=  19.6s
[CV 3/5; 121/225] START learning_rate=0.1, max_depth=17, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 120/225] END learning_rate=0.1, max_depth=17, n_estimators=400;, score=0.573 total time=  16.7s
[CV 4/5; 121/225] START learning_rate=0.1, max_depth=17, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 121/225] END learning_rate=0.1, max_depth=17, n_estimators=500;, score=0.572 total time=  20.0s
[CV 5/5; 121/225] START learning_rate=0.1, max_depth=17, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 121/225] END learning_rate=0.1, max_depth=17, n_estimators=500;, score=0.573 total time=  19.6s
[CV 1/5; 122/225] START learning_rate=0.1, max_depth=17, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 121/225] END learning_rate=0.1, max_depth=17, n_estimators=500;, score=0.569 total time=  20.6s
[CV 2/5; 122/225] START learning_rate=0.1, max_depth=17, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 121/225] END learning_rate=0.1, max_depth=17, n_estimators=500;, score=0.571 total time=  26.6s
[CV 3/5; 122/225] START learning_rate=0.1, max_depth=17, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 121/225] END learning_rate=0.1, max_depth=17, n_estimators=500;, score=0.572 total time=  23.4s
[CV 4/5; 122/225] START learning_rate=0.1, max_depth=17, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 122/225] END learning_rate=0.1, max_depth=17, n_estimators=600;, score=0.572 total time=  24.7s
[CV 5/5; 122/225] START learning_rate=0.1, max_depth=17, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 122/225] END learning_rate=0.1, max_depth=17, n_estimators=600;, score=0.573 total time=  27.1s
[CV 1/5; 123/225] START learning_rate=0.1, max_depth=17, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 122/225] END learning_rate=0.1, max_depth=17, n_estimators=600;, score=0.570 total time=  29.0s
[CV 2/5; 123/225] START learning_rate=0.1, max_depth=17, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 122/225] END learning_rate=0.1, max_depth=17, n_estimators=600;, score=0.571 total time=  26.9s
[CV 3/5; 123/225] START learning_rate=0.1, max_depth=17, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 122/225] END learning_rate=0.1, max_depth=17, n_estimators=600;, score=0.572 total time=  24.6s
[CV 4/5; 123/225] START learning_rate=0.1, max_depth=17, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 123/225] END learning_rate=0.1, max_depth=17, n_estimators=700;, score=0.572 total time=  27.9s
[CV 5/5; 123/225] START learning_rate=0.1, max_depth=17, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 123/225] END learning_rate=0.1, max_depth=17, n_estimators=700;, score=0.573 total time=  26.6s
[CV 1/5; 124/225] START learning_rate=0.1, max_depth=17, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 123/225] END learning_rate=0.1, max_depth=17, n_estimators=700;, score=0.571 total time=  29.1s
[CV 2/5; 124/225] START learning_rate=0.1, max_depth=17, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 123/225] END learning_rate=0.1, max_depth=17, n_estimators=700;, score=0.570 total time=  34.3s
[CV 3/5; 124/225] START learning_rate=0.1, max_depth=17, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 123/225] END learning_rate=0.1, max_depth=17, n_estimators=700;, score=0.573 total time=  25.2s
[CV 4/5; 124/225] START learning_rate=0.1, max_depth=17, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 124/225] END learning_rate=0.1, max_depth=17, n_estimators=800;, score=0.572 total time=  34.4s
[CV 5/5; 124/225] START learning_rate=0.1, max_depth=17, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 124/225] END learning_rate=0.1, max_depth=17, n_estimators=800;, score=0.573 total time=  38.9s
[CV 1/5; 125/225] START learning_rate=0.1, max_depth=17, n_estimators=900.......
[CV 3/5; 124/225] END learning_rate=0.1, max_depth=17, n_estimators=800;, score=0.570 total time=  36.6s
[CV 2/5; 125/225] START learning_rate=0.1, max_depth=17, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 124/225] END learning_rate=0.1, max_depth=17, n_estimators=800;, score=0.572 total time=  37.0s
[CV 3/5; 125/225] START learning_rate=0.1, max_depth=17, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 124/225] END learning_rate=0.1, max_depth=17, n_estimators=800;, score=0.572 total time=  32.6s
[CV 4/5; 125/225] START learning_rate=0.1, max_depth=17, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 125/225] END learning_rate=0.1, max_depth=17, n_estimators=900;, score=0.572 total time=  37.1s
[CV 5/5; 125/225] START learning_rate=0.1, max_depth=17, n_estimators=900.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 125/225] END learning_rate=0.1, max_depth=17, n_estimators=900;, score=0.573 total time=  38.7s
[CV 1/5; 126/225] START learning_rate=0.1, max_depth=17, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 125/225] END learning_rate=0.1, max_depth=17, n_estimators=900;, score=0.570 total time=  39.6s
[CV 2/5; 126/225] START learning_rate=0.1, max_depth=17, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 125/225] END learning_rate=0.1, max_depth=17, n_estimators=900;, score=0.571 total time=  41.4s
[CV 3/5; 126/225] START learning_rate=0.1, max_depth=17, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 125/225] END learning_rate=0.1, max_depth=17, n_estimators=900;, score=0.572 total time=  36.2s
[CV 4/5; 126/225] START learning_rate=0.1, max_depth=17, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 126/225] END learning_rate=0.1, max_depth=17, n_estimators=1000;, score=0.572 total time=  46.6s
[CV 5/5; 126/225] START learning_rate=0.1, max_depth=17, n_estimators=1000......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 126/225] END learning_rate=0.1, max_depth=17, n_estimators=1000;, score=0.573 total time=  45.1s
[CV 1/5; 127/225] START learning_rate=0.1, max_depth=20, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 127/225] END learning_rate=0.1, max_depth=20, n_estimators=200;, score=0.571 total time=  11.1s
[CV 2/5; 127/225] START learning_rate=0.1, max_depth=20, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 126/225] END learning_rate=0.1, max_depth=17, n_estimators=1000;, score=0.569 total time=  41.8s
[CV 3/5; 127/225] START learning_rate=0.1, max_depth=20, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 127/225] END learning_rate=0.1, max_depth=20, n_estimators=200;, score=0.572 total time=  12.2s
[CV 4/5; 127/225] START learning_rate=0.1, max_depth=20, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 126/225] END learning_rate=0.1, max_depth=17, n_estimators=1000;, score=0.571 total time=  42.3s
[CV 5/5; 127/225] START learning_rate=0.1, max_depth=20, n_estimators=200.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 127/225] END learning_rate=0.1, max_depth=20, n_estimators=200;, score=0.568 total time=  15.0s
[CV 1/5; 128/225] START learning_rate=0.1, max_depth=20, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 127/225] END learning_rate=0.1, max_depth=20, n_estimators=200;, score=0.570 total time=  12.2s
[CV 2/5; 128/225] START learning_rate=0.1, max_depth=20, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 127/225] END learning_rate=0.1, max_depth=20, n_estimators=200;, score=0.571 total time=  13.4s
[CV 3/5; 128/225] START learning_rate=0.1, max_depth=20, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 126/225] END learning_rate=0.1, max_depth=17, n_estimators=1000;, score=0.572 total time=  48.1s
[CV 4/5; 128/225] START learning_rate=0.1, max_depth=20, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 128/225] END learning_rate=0.1, max_depth=20, n_estimators=300;, score=0.572 total time=  16.2s
[CV 5/5; 128/225] START learning_rate=0.1, max_depth=20, n_estimators=300.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 128/225] END learning_rate=0.1, max_depth=20, n_estimators=300;, score=0.572 total time=  13.2s
[CV 1/5; 129/225] START learning_rate=0.1, max_depth=20, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 128/225] END learning_rate=0.1, max_depth=20, n_estimators=300;, score=0.569 total time=  13.1s
[CV 2/5; 129/225] START learning_rate=0.1, max_depth=20, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 128/225] END learning_rate=0.1, max_depth=20, n_estimators=300;, score=0.571 total time=  15.8s
[CV 3/5; 129/225] START learning_rate=0.1, max_depth=20, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 128/225] END learning_rate=0.1, max_depth=20, n_estimators=300;, score=0.572 total time=  15.9s
[CV 4/5; 129/225] START learning_rate=0.1, max_depth=20, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 129/225] END learning_rate=0.1, max_depth=20, n_estimators=400;, score=0.572 total time=  17.6s
[CV 5/5; 129/225] START learning_rate=0.1, max_depth=20, n_estimators=400.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 129/225] END learning_rate=0.1, max_depth=20, n_estimators=400;, score=0.573 total time=  19.7s
[CV 1/5; 130/225] START learning_rate=0.1, max_depth=20, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 129/225] END learning_rate=0.1, max_depth=20, n_estimators=400;, score=0.571 total time=  21.2s
[CV 2/5; 130/225] START learning_rate=0.1, max_depth=20, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 129/225] END learning_rate=0.1, max_depth=20, n_estimators=400;, score=0.569 total time=  22.7s
[CV 3/5; 130/225] START learning_rate=0.1, max_depth=20, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 129/225] END learning_rate=0.1, max_depth=20, n_estimators=400;, score=0.573 total time=  17.6s
[CV 4/5; 130/225] START learning_rate=0.1, max_depth=20, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 130/225] END learning_rate=0.1, max_depth=20, n_estimators=500;, score=0.572 total time=  22.5s
[CV 5/5; 130/225] START learning_rate=0.1, max_depth=20, n_estimators=500.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 130/225] END learning_rate=0.1, max_depth=20, n_estimators=500;, score=0.573 total time=  21.2s
[CV 1/5; 131/225] START learning_rate=0.1, max_depth=20, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 130/225] END learning_rate=0.1, max_depth=20, n_estimators=500;, score=0.569 total time=  24.9s
[CV 2/5; 131/225] START learning_rate=0.1, max_depth=20, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 130/225] END learning_rate=0.1, max_depth=20, n_estimators=500;, score=0.571 total time=  21.3s
[CV 3/5; 131/225] START learning_rate=0.1, max_depth=20, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 130/225] END learning_rate=0.1, max_depth=20, n_estimators=500;, score=0.572 total time=  21.3s
[CV 4/5; 131/225] START learning_rate=0.1, max_depth=20, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 131/225] END learning_rate=0.1, max_depth=20, n_estimators=600;, score=0.572 total time=  30.3s
[CV 5/5; 131/225] START learning_rate=0.1, max_depth=20, n_estimators=600.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 131/225] END learning_rate=0.1, max_depth=20, n_estimators=600;, score=0.570 total time=  29.6s
[CV 1/5; 132/225] START learning_rate=0.1, max_depth=20, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 131/225] END learning_rate=0.1, max_depth=20, n_estimators=600;, score=0.573 total time=  30.9s
[CV 2/5; 132/225] START learning_rate=0.1, max_depth=20, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 4/5; 131/225] END learning_rate=0.1, max_depth=20, n_estimators=600;, score=0.571 total time=  25.6s
[CV 3/5; 132/225] START learning_rate=0.1, max_depth=20, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 5/5; 131/225] END learning_rate=0.1, max_depth=20, n_estimators=600;, score=0.572 total time=  27.5s
[CV 4/5; 132/225] START learning_rate=0.1, max_depth=20, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 2/5; 132/225] END learning_rate=0.1, max_depth=20, n_estimators=700;, score=0.573 total time=  29.9s
[CV 5/5; 132/225] START learning_rate=0.1, max_depth=20, n_estimators=700.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 1/5; 132/225] END learning_rate=0.1, max_depth=20, n_estimators=700;, score=0.572 total time=  31.7s
[CV 1/5; 133/225] START learning_rate=0.1, max_depth=20, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
[CV 3/5; 132/225] END learning_rate=0.1, max_depth=20, n_estimators=700;, score=0.570 total time=  31.4s
[CV 2/5; 133/225] START learning_rate=0.1, max_depth=20, n_estimators=800.......
/home/jupyter-kriach/.conda/envs/biasbounties/lib/python3.10/site-packages/lightgbm/basic.py:2065: UserWarning: Using categorical_feature in Dataset.
  _log_warning('Using categorical_feature in Dataset.')
    
'''

