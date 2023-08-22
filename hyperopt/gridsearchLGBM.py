import optuna  # pip install optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import train_test_split
import lightgbm as lgbm

rand_seed = 21
def objective(trial, X, y):
    param_grid = {
        "device_type": trial.suggest_categorical("device_type", ['gpu']),
        # "n_estimators": trial.suggest_categorical("n_estimators", \
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step = 100)
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, step = 0.01),
        # "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 8, 15),
        # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        # "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        # "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        # "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        # "bagging_fraction": trial.suggest_float(
            # "bagging_fraction", 0.2, 0.95, step=0.1
        # ),
        # "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        # "feature_fraction": trial.suggest_float(
            # "feature_fraction", 0.2, 0.95, step=0.1
        # ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=rand_seed)
    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = lgbm.LGBMRegressor(objective="regression", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=cat_cols_sig
        )
        r2_score = model.score(X_test, y_test)
        cv_scores[idx] = r2_score
    return np.mean(cv_scores)

cat_cols_sig = ['OCCP', 'SCHL', 'ST', 'JWTRNS', 'DRAT', 'COW', 'SEX', \
       'RELSHIPP', 'POBP', 'ENG', 'MAR', 'RAC1P']
numeric_cols = ['WKHP', 'AGEP', 'PINCP']

df_all = pd.read_pickle("./data_frames/dense_acs_mm_notoh.pkl")
df_train, df_test = train_test_split(df_all, test_size=0.2, random_state = rand_seed)

X = df_train.drop(['PINCP'], axis=1, inplace = False).to_numpy()
y = df_train['PINCP'].to_numpy()

print("here", type(X), type(y))

optuna.logging.set_verbosity(optuna.logging.WARNING)
# study = optuna.create_study(direction="minimize", study_name="LGBM Regressor")
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
func = lambda trial: objective(trial, X, y)
study.optimize(func)

print(f"\tBest value (rmse): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")