# imports
import argparse
import subprocess
from termcolor import colored
from NYCTaxiFare.data import get_data, clean_data
from NYCTaxiFare.encoders import TimeFeaturesEncoder, DistanceTransformer
from NYCTaxiFare.parameters import *
from NYCTaxiFare.utils import custom_rmse

# pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer

# joblib
import joblib


# Update to change parameters to test
params = params_SVR


class Trainer():

    def __init__(self, X, y, params=params):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.rmse = make_scorer(custom_rmse, greater_is_better=False)
        self.model = None
        self.X_test = None
        self.params = params
        self.X = X
        self.y = y
        self.baseline_rmse = None
        self.optimized_rmse = None
        self.experiment_name = EXPERIMENT_NAME


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # TIME PIPELINE
        pipe_time = Pipeline([
            ('time_features_create', TimeFeaturesEncoder('pickup_datetime')),
            ('time_features_ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        # COMBINATION OF DISTANCE AND TIME PIPELINE
        preprocessor = ColumnTransformer([
            ('distance', DistanceTransformer(), ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']),
            ('time', pipe_time, ['pickup_datetime'])
        ], remainder='passthrough')

        # PREPROCESSOR PIPELINE
        pipe_prepro = Pipeline([
            ('dist_and_time', preprocessor),
            ('scaler', MinMaxScaler())
        ])

        # FULL PIPELINE
        self.pipeline = Pipeline([
            ('preprocessor', pipe_prepro),
            ('model', self.params["model"])
        ])


    def cross_validate_baseline(self, cv=20):
        """ compute model baseline on rmsle """

        # custom scorer definition Root Mean Squared Log Error
        # rmsle = make_scorer(lambda *x: mean_squared_log_error(*x)**0.5)

        # launching crossvalidation scoring
        baseline = cross_validate(self.pipeline,
                                  self.X,
                                  self.y,
                                  scoring=self.rmse,
                                  cv=cv)
        self.baseline_rmse = -round(baseline["test_score"].mean(), 3)

        # ### PRINT RESULTS ON TERMINAL
        print("Baseline " + type(self.params["model"]).__name__ + " model rmse: " +
              str(self.baseline_rmse))

    def run(self):
        """ looking for best parameters for the model and training """

        self.model = RandomizedSearchCV(self.pipeline,
                                        self.params["random_grid_search"],
                                        scoring=self.rmse,
                                        n_iter=100,
                                        cv=5,
                                        n_jobs=-1)
        self.model.fit(self.X, self.y)
        self.optimized_rmse = -round(self.model.best_score_, 3)
        print("Tuned " + type(self.params["model"]).__name__ + " model best rmse: " +
              str(round(self.optimized_rmse, 3)))

        # ### PRINT BEST PARAMETERS
        print("\n####################################\nBest parameters:")
        for k, v in self.model.best_params_.items():
            print(k, colored(v, "green"))
        print("####################################\n")

    def save_model(self, model_name):
        """ Save the model into a .joblib format """
        joblib.dump(self.model, model_name + ".joblib")
        print(colored("Trained model saved locally under " + model_name + ".joblib", "green"))



# terminal parameter definition
parser = argparse.ArgumentParser(description='NYCTaxiFare trainer')
parser.add_argument('-m', action="store",
                    dest="modelname",
                    help='.joblib model name - default: model',
                    default="model")

if __name__ == "__main__":
    # getting optionnal arguments otherwise default
    results = parser.parse_args()

    # get data
    data, _ = get_data(nrows=1000)

    # clean data
    data = clean_data(data)

    # set X and y
    X = data.drop(["fare_amount"], axis=1)
    y = data["fare_amount"]

    # define trainer
    trainer = Trainer(X, y)
    trainer.set_pipeline()

    # get best accuracy
    trainer.cross_validate_baseline()
    trainer.run()

    # saving trained model and moving it to models folder
    trainer.save_model(model_name=results.modelname)
    subprocess.run(["mv", results.modelname + ".joblib", "models"])
