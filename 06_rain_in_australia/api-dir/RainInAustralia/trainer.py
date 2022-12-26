# imports
import argparse
import subprocess
from termcolor import colored
from RainInAustralia.data import get_data, clean_data
from RainInAustralia.parameters import *

# pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# mlflow
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
import mlflow

# joblib
import joblib


# Update to change parameters to test
params = params_lr


class Trainer():

    def __init__(self, X, y, params=params):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.model = None
        self.params = params
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # NUMERIC PIPELINE
        pipe_numeric = Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        # PIPELINE FOR BINARY FEATURES
        pipe_binary = Pipeline([
            ('encoder', OneHotEncoder(sparse=False, drop='if_binary'))
        ])

        # PIPELINE FOR MULTICLASS FEATURES
        pipe_multiclass = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False, handle_unknown="ignore"))
        ])

        # IMPUTE AND ENCODE PIPELINE
        impute_and_encode = ColumnTransformer([
            ('numeric', pipe_numeric, make_column_selector(dtype_include="float64")),
            ('binary', pipe_binary, make_column_selector(dtype_include="int64")),
            ('multiclass', pipe_multiclass, make_column_selector(dtype_include="object"))])

        # PREPROCESSOR PIPELINE
        preprocessor = Pipeline([("preproc", impute_and_encode),
                                 ("scaler", StandardScaler())])

        # FULL PIPELINE
        self.pipeline = Pipeline([
                      ("preprocessor", preprocessor),
                      ('classifier', LogisticRegression(max_iter=10000))
              ])


    def cross_validate_baseline(self, cv=20):
        """ compute model baseline on accuracy and recall """

        # launching crossvalidation scoring
        y_pred = cross_val_predict(estimator=self.pipeline,
                                   X=self.X,
                                   y=self.y,
                                   cv=cv)
        self.baseline_scores = {"accuracy": round(accuracy_score(y, y_pred)*100, 1),
                                "recall": round(recall_score(y, y_pred)*100, 1),
                                "precision": round(precision_score(y, y_pred)*100, 1)}

        # ### PRINT RESULTS ON TERMINAL
        print("Baseline accuracy with " + type(self.params["model"]).__name__ + " model is: " +
              str(self.baseline_scores["accuracy"])+"%")
        print("Baseline recall with " + type(self.params["model"]).__name__ + " model is: " +
              str(self.baseline_scores["recall"])+"%")
        print("Baseline precision with " + type(self.params["model"]).__name__ + " model is: " +
              str(self.baseline_scores["precision"])+"%")

        # ### MLFLOW RECORDS
        self.mlflow_log_metric("Baseline accuracy", self.baseline_scores["accuracy"])
        self.mlflow_log_metric("Baseline recall", self.baseline_scores["recall"])
        self.mlflow_log_metric("Baseline precision", self.baseline_scores["precision"])
        self.mlflow_log_param("Model", type(self.params["model"]).__name__)


    def run(self):
        """ looking for best parameters for the model and training """

        self.model = RandomizedSearchCV(self.pipeline,
                                        self.params["random_grid_search"],
                                        scoring="recall",
                                        n_iter=10,
                                        cv=5,
                                        n_jobs=-1)
        self.model.fit(self.X, self.y)
        self.optimized_recall = round(self.model.best_score_*100, 1)
        print("Tuned " + type(self.params["model"]).__name__ + " model best recall: " +
              str(round(self.optimized_recall, 1)))

        # ### PRINT BEST PARAMETERS
        print("\n####################################\nBest parameters:")
        for k, v in self.model.best_params_.items():
            print(k, colored(v, "green"))
        print("####################################\n")

        # ### MLFLOW RECORDS
        self.mlflow_log_metric("Optimized recall", self.optimized_recall)
        for k, v in self.model.best_params_.items():
            self.mlflow_log_param(k, v)



    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(CUSTOMURI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



    def save_model(self, model_name):
        """ Save the model into a .joblib format """
        joblib.dump(self.model, model_name + ".joblib")
        print(colored("Trained model saved locally under " + model_name + ".joblib", "green"))


# terminal parameter definition
parser = argparse.ArgumentParser(description='Rain in Australia trainer')
parser.add_argument('-m', action="store",
                    dest="modelname",
                    help='.joblib model name - default: model',
                    default="model")

if __name__ == "__main__":
    # getting optionnal arguments otherwise default
    results = parser.parse_args()

    # get data
    data = get_data()

    # clean data
    data = clean_data(data, reduced=True)
    print(data)

    # set X and y
    X = data.drop(["RainTomorrow"], axis=1)
    y = data["RainTomorrow"]

    # define trainer
    trainer = Trainer(X, y)
    trainer.set_pipeline()

    # get best accuracy and recall
    trainer.cross_validate_baseline(cv=5)
    trainer.run()

    # saving trained model and moving it to models folder
    trainer.save_model(model_name=results.modelname)
    subprocess.run(["mv", results.modelname + ".joblib", "models"])
