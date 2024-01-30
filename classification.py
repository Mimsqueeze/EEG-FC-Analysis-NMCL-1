# Imports
import numpy as np
import scipy.signal as ss
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

# ----------------------- author: Minsi Hu ---------------------------

# ----------------------- program description: -----------------------
#           Performs classification of high or low mental
#    workload on EEG features extracted from feature_extraction.py
# --------------------------------------------------------------------

# GLOBAL VARIABLES
FEATURE_TYPE = "COH" # Set for file names prefix ("PLI", "COH", "PLI_COH")

"""
The main() function

Reads features produced by feature_extraction.py, and performs 
classification of whether the EEG data represents high or low mental
workload. Saves the results of classifcation in the ./results directory
"""
def main():
  # Initialize dataframe to store features, where each column is a 5 second
  # interval and the rows contain the pli/coherence features
  data_hi = pd.DataFrame()
  data_lo = pd.DataFrame()

  print(f"Reading in features...")

  # Read in features for all subjects from 1 to 48
  for i in range(1, 49):
    # hi and lo keys
    key_hi = f"sub{i:02}_hi"
    key_lo = f"sub{i:02}_lo"

    df_hi = pd.read_csv(f"./features/{FEATURE_TYPE} {key_hi}.csv", 
                        engine="python")
    df_hi.rename(columns={"Unnamed: 0": "Time"}, inplace=True)
    df_hi.set_index("Time", inplace=True)
    data_hi = pd.concat([data_hi, df_hi])

    df_lo = pd.read_csv(f"./features/{FEATURE_TYPE} {key_lo}.csv", 
                        engine="python")
    df_lo.rename(columns={"Unnamed: 0": "Time"}, inplace=True)
    df_lo.set_index("Time", inplace=True)
    data_lo = pd.concat([data_lo, df_lo])

  print(f"Finished reading in features!")
  print(data_lo)
  print(data_hi)

  # Add labels to features dataframes, 1 means hi and 0 means lo
  data_hi["label"] = 1
  data_lo["label"] = 0

  # Combine the lo and hi features into a single dataframe
  data = pd.concat([data_hi, data_lo])

  # Shuffle the rows of the dataframe
  data = data.sample(frac = 1)

  # Create classifier

  # MODEL = "RFC n=10"
  # rfc = RandomForestClassifier(n_estimators=10, random_state=42)
  # clf = CalibratedClassifierCV(rfc)

  MODEL = "SVM"
  svm = SVC(kernel='linear', C=1.0, probability=True)
  clf = CalibratedClassifierCV(svm)

  # MODEL = "LOG"
  # log = LogisticRegression(max_iter=5000, random_state=42)
  # clf = CalibratedClassifierCV(log)

  # MODEL = "MLP n=250"
  # mlp = MLPClassifier(
  #   hidden_layer_sizes=(250,), max_iter=1000, random_state=42)
  # clf = CalibratedClassifierCV(mlp, method='sigmoid')

  # Create scoring metrics
  metrics = {'f1' : make_scorer(f1_score, average='weighted'),
            'accuracy' : make_scorer(accuracy_score),
            'neg_log_loss' : 'neg_log_loss'}
  
  # Split data into features and labels
  X = data.drop(columns=["label"])
  y = data["label"]

  print(f"Training and testing...")

  # Compute cross validation scores
  scores = cross_validate(clf, X, y, cv=10, scoring=metrics, 
                          return_estimator=True, return_train_score=True)

  print(f"Finished training and testing!")

  # Create a dataframe to store results to be saved into file
  df = pd.DataFrame()
  df["Classifier"] = [MODEL]
  df["Mean Fit Time"] = [np.mean(scores['fit_time'])]
  df["Mean Score Time"] = [np.mean(scores['score_time'])]
  df["Mean Train Accuracy"] = [np.mean(scores['train_accuracy'])]
  df["Mean Test Accuracy"] = [np.mean(scores['test_accuracy'])]
  df["Mean Train F1"] = [np.mean(scores['train_f1'])]
  df["Mean Test F1"] = [np.mean(scores['test_f1'])]
  df["Mean Train Neg Log Loss"] = [np.mean(scores['train_neg_log_loss'])]
  df["Mean Test Neg Log Loss"] = [np.mean(scores['test_neg_log_loss'])]
  
  # Transpose the dataframe to get metrics as rows
  df = df.T

  print(f"Storing results in ./results/{MODEL}.csv!")

  # Save the results to file
  df.to_csv(f"./results/{FEATURE_TYPE} {MODEL}.csv", header=False)

if __name__ == "__main__":
  main()
