# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

# others
from scipy import stats

# MLFLOW PARAMETERS
MLFLOW_URI = "https://mlflow.lewagon.co/"
CUSTOMURI = ""
myname = "VictorBnnt"
EXPERIMENT_NAME = f"[FR] [Paris] [{myname}] NYCTaxiFare"


# training parameters
######################################################
# LinearRegression model
######################################################
grid_lr = {# 'model__kernel': ["rbf", "sigmoid"],#['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
           # 'model__C': stats.loguniform(1, 2),
           # 'gamma': 'auto',
           # 'model__degree': stats.randint(1, 3),
           "preprocessor__scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()]
           }
#
params_lr = {"random_grid_search": grid_lr,
             "model": LinearRegression()}
######################################################

######################################################
# RandomForestRegressor model
######################################################
grid_rfr = {'model__n_estimators': stats.randint(1, 300),
            'model__max_depth': stats.randint(1, 300),
            'model__max_samples': stats.randint(1, 300),
            "preprocessor__scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()]
            }
#
params_rfr = {"random_grid_search": grid_rfr,
              "model": RandomForestRegressor()}
######################################################

######################################################
# GradientBoostingRegressor model
######################################################
grid_gbr = {#'model__loss': ["ls", "lad", "huber", "quantile"],
            'model__learning_rate': stats.loguniform(0.001, 10),
            'model__n_estimators': stats.randint(1, 300),
            "preprocessor__scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()]
           }
#
params_gbr = {"random_grid_search": grid_gbr,
              "model": GradientBoostingRegressor()}
######################################################

######################################################
# AdaBoostRegressor model
######################################################
grid_abr = {'model__learning_rate': stats.loguniform(0.001, 10),
            'model__n_estimators': stats.randint(1, 300),
            "preprocessor__scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()]
            }
#
params_abr = {"random_grid_search": grid_abr,
              "model": AdaBoostRegressor()}
######################################################

######################################################
# SVR model
######################################################
grid_SVR = {'model__C': stats.loguniform(0.001, 10),
            'model__epsilon': stats.loguniform(0.001, 10),
            'model__gamma': stats.loguniform(0.001, 10)
            #"preprocessor__scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()]
            }
#
params_SVR = {"random_grid_search": grid_SVR,
              "model": SVR()}
######################################################
