import sys
sys.path.append("/home/jupyter-kriach/md_github_lat/multigroupcode/multigroup-code/")
import pandas as pd
import numpy as np
from bilevel.utils import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


numeric_feat = ['hours-per-week', 'age', 'capital-gain', 'capital-loss', 'education-num']
cat_feat = ['workclass', 'education', 'marital-status', 'relationship', 'race', 'sex', 'native-country', 'occupation']
numerics_all = numeric_feat + ['income']

print(numerics_all)
print(cat_feat)

# encoding
adult_loc = '/home/jupyter-kriach/md_github_lat/multigroupcode/multigroup-code/data_small/adult_reconstruction.csv'
df_adult = pd.read_csv(adult_loc)
df_adult = numeric_scaler(df_adult, numerics_all) #minmax scale
df_adult = ordinal_encoder(df_adult, cat_feat) #label encode
df_adult = df_adult * 1.0 # bool to float

# train,test split
from sklearn.model_selection import train_test_split
random_seed = 21
X_train, X_test, y_train, y_test = train_test_split(df_adult.drop('income', axis=1),  \
                                                    df_adult['income'], test_size = 0.2, \
                                                    shuffle=True, random_state = random_seed)

# grid search for ``best'' gbr
param_grid = {
"max_depth" : np.arange(3, 20), \
"learning_rate" : [0.01, 0.05, 0.1], \
"n_estimators" : np.arange(100, 1100, 200)
}
gbr = GradientBoostingRegressor()
grid_search = GridSearchCV(estimator = gbr, 
                    param_grid = param_grid, 
                    scoring='r2',
                    cv=5,
                    n_jobs = 4, 
                    verbose = 10)
grid_result = grid_search.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Best: 0.693028 using {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 700}