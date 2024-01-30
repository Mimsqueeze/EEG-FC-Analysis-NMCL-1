# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------- author: Minsi Hu -----------------------

# ----------------------- program description: -----------------------
#            Plots the performance of different classifiers
#         in classifying high/low mental workload of EEG data
# --------------------------------------------------------------------

# GLOBAL VARIABLES
FEATURE_TYPE = "COH" # Set for file names prefix ("PLI", "COH", "PLI_COH")

"""
The main() function

Reads in the results data and plots the performance of different classifiers
based on different metrics
"""
def main():
  print("Plotting...")

  # Read the results data
  df1 = pd.read_csv(f"./results/{FEATURE_TYPE} SVM.csv", engine="python", 
                    header=0, index_col=0)
  df2 = pd.read_csv(f"./results/{FEATURE_TYPE} RFC n=50.csv", engine="python", 
                    header=0, index_col=0)
  df3 = pd.read_csv(f"./results/{FEATURE_TYPE} LOG.csv", engine="python", 
                    header=0, index_col=0)
  df4 = pd.read_csv(f"./results/{FEATURE_TYPE} MLP n=250.csv", engine="python",
                    header=0, index_col=0)

  # Merge the results into a single dataframe
  data = df1.merge(df2, left_index=True, right_index=True)
  data = data.merge(df3, left_index=True, right_index=True)
  data = data.merge(df4, left_index=True, right_index=True)

  # Convert Negative Log Loss to Log Loss
  data = data.rename(index={"Mean Train Neg Log Loss": "Mean Train Log Loss", 
                            "Mean Test Neg Log Loss": "Mean Test Log Loss"})
  data = data.apply(np.abs)

  # Round all numbers to 2
  data = data.apply(lambda x: np.round(x, 2))

  # Create result metrics array for different plots
  result_metrics = [["Mean Fit Time", "Mean Score Time"],
                    ["Mean Train Accuracy", "Mean Test Accuracy"],
                    ["Mean Train F1", "Mean Test F1"],
                    ["Mean Train Log Loss", "Mean Test Log Loss"]
                    ]
  
  # Make a different plot for each metric
  for metric in result_metrics:
    # Extract the wanted rows
    df = data.loc[metric]

    x = np.arange(len(df.columns)) # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    # Plot the bars
    for row_name in df.index.values:
      offset = width * multiplier
      rects = ax.bar(x + offset, df.iloc[multiplier], width, label=row_name)
      ax.bar_label(rects, padding=3)
      multiplier += 1

    # Add labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Value/Score of ML Metric')
    ax.set_title('Performance metrics for different classifiers')
    ax.set_xticks(x + (len(df.index.values)-1)*width/2, df.columns)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 1.15)

    # Save plot to file
    plt.savefig(f"./plots/{FEATURE_TYPE} {metric}.png", dpi=500, 
                bbox_inches="tight")
    plt.clf()

    print(f"Finished plotting {metric}!")

if __name__ == '__main__':
  main()