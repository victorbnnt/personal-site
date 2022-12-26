# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# models
from sklearn.linear_model import LogisticRegression

# others
from scipy import stats

# MLFLOW PARAMETERS
CUSTOMURI = ""
myname = "VictorBnnt"
EXPERIMENT_NAME = f"[FR] [Paris] [{myname}] RainInAustralia"

# DATASET CLEANING
badly_named = {"AliceSprings": "Alice Springs",
               "BadgerysCreek": "Badgerys Creek",
               "CoffsHarbour": "Coffs Harbour",
               "GoldCoast": "Gold Coast",
               "MelbourneAirport": "Melbourne Airport",
               "MountGambier": "Mount Gambier",
               "MountGinini": "Mount Ginini",
               "NorahHead": "Norah Head",
               "NorfolkIsland": "Norfolk Island",
               "PearceRAAF": "Pearce RAAF",
               "PerthAirport": "Perth Airport",
               "SalmonGums": "Salmon Gums",
               "SydneyAirport": "Sydney Airport",
               "WaggaWagga": "Wagga Wagga"}

# training parameters
######################################################
# LogisticRegression model
######################################################
grid_lr = {'classifier__penalty': ["l1", "l2"], # "elasticnet", "none",
           'classifier__solver': ["liblinear", "sag", "saga"], # "newton-cg", "lbfgs",
           'classifier__class_weight': ["balanced", "none"],
           'classifier__C': stats.loguniform(0.1, 10),
           'classifier__tol': stats.loguniform(0.0001, 2),
           # 'classifier__l1_ratio': stats.loguniform(0.01, 1),
           'preprocessor__preproc__numeric__imputer__strategy': ["mean", "median"],
           "preprocessor__scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()]
           }
#
params_lr = {"random_grid_search": grid_lr,
             "model": LogisticRegression()}
######################################################
