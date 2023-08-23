import pandas as pd
import numpy as np
import random
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from utils import ordinal_encoder

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

cat_cols_sig = ['OCCP', 'SCHL', 'ST', 'JWTRNS', 'DRAT', 'COW', 'SEX', \
       'RELSHIPP', 'POBP', 'ENG', 'MAR', 'RAC1P'] # significant features from the earlier analysis
numeric_cols = ['WKHP', 'AGEP', 'PINCP']
print(len(cat_cols_sig), len(numeric_cols))

df_all = pd.read_pickle("./data_frames/dense_acs_mm_notoh.pkl")
df_all = ordinal_encoder(df_all, cat_cols_sig)

from sklearn.model_selection import train_test_split
rand_seed = 21
df_train, df_test = train_test_split(df_all, test_size=0.2, random_state = rand_seed)

y_train = df_train['PINCP']
X_train = df_train.drop(['PINCP'], axis=1, inplace = False)

y_test = df_test['PINCP']
X_test = df_test.drop(['PINCP'], axis=1, inplace = False)

# max_depth = np.arange(1, 20)
max_depth = [4, 6, 8]
learning_rate = [0.01, 0.05, 0.1]
n_estimators = [10, 100, 200]

param_grid = {'max_depth': max_depth,
              'learning_rate': learning_rate,
              'n_estimators': n_estimators
            }


gbr = GradientBoostingRegressor()
rs = RandomizedSearchCV(gbr, param_grid, n_iter = 20, scoring = 'r2', n_jobs=4, random_state = rand_seed, verbose = 10)
res = rs.fit(X_train, y_train)
print("Best: %f using %s" % (res.best_score_, res.best_params_))
