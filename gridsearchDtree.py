import pandas as pd
import numpy as np
import random
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from utils import ordinal_encoder

from sklearn import tree
from sklearn.model_selection import GridSearchCV

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

max_depth = np.arange(1, 20)
splitter = ['best', 'random']

# grid_dt = {'max_depth': max_depth,
#                'splitter': splitter,
#                'min_samples_leaf': min_samples_leaf,
#                'min_samples_split': min_samples_split}
grid_dt = {'max_depth': max_depth,
               'splitter': splitter
          }

dtree_reg = tree.DecisionTreeRegressor()
grid_search = GridSearchCV(estimator = dtree_reg, param_grid = grid_dt, n_jobs=2, verbose=10)
grid_result = grid_search.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Best: 0.493419 using {'max_depth': 13, 'splitter': 'best'}