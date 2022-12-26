# imports
import os
import joblib
import argparse

PATH_TO_LOCAL_MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/models/"


def get_model(model):
    return joblib.load(os.path.join(PATH_TO_LOCAL_MODEL, model + ".joblib"))


# terminal parameter definition
parser = argparse.ArgumentParser(description='Prediction on Rain next day in Australia')
parser.add_argument('-m', action="store",
                    dest="modelname",
                    help='.joblib model to load for prediction - default: model',
                    default="model")


if __name__ == "__main__":

    # getting optionnal arguments otherwise default
    results = parser.parse_args()

    model = get_model(results.modelname)

    print("\nLoaded model parameters:")
    for k, v in model.best_params_.items():
        print(str(k) + ": " + str(v))
    print("\n")
