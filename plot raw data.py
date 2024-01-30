# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------- author: Minsi Hu -----------------------

# ----------------------- program description: -----------------------
#                         Plots raw EEG data
# --------------------------------------------------------------------

"""
The main() function

Reads in the EEG data, plots them, and saves it into the 
./raw data/ directory
"""
def main():
  print("Plotting raw data...")

  for i in range(1, 49):
    # hi and lo keys
    key_hi = f"sub{i:02}_hi"
    key_lo = f"sub{i:02}_lo"

    # Read in the data
    data_hi = pd.read_csv(
      f"./data/STEW Dataset/{key_hi}.txt", sep="   ", header=None, 
      engine="python")
    data_lo = pd.read_csv(
      f"./data/STEW Dataset/{key_lo}.txt", sep="   ", header=None, 
      engine="python")
    
    # Rename the columns
    column_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", 
                    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    data_hi.columns = column_names
    data_lo.columns = column_names

    # Plot the raw eeg data
    data_hi.plot(y=column_names, kind="line", legend=True)
    plt.title(f"Raw EEG Data for Subject {i} High Workload")
    plt.xlabel("Time (1/128s)")
    plt.ylabel("Postsynaptic potentials (mV)")
    plt.xlim(0, 19200)
    plt.legend(loc="upper left", ncols=7, fontsize="xx-small")
    plt.savefig(f"./raw data/{key_hi}.png", dpi=500, bbox_inches="tight")
    plt.clf()
    plt.close()

    data_lo.plot(y=column_names, kind="line", legend=True)
    plt.title(f"Raw EEG Data for Subject {i} Low Workload")
    plt.xlabel("Time (1/128s)")
    plt.ylabel("Postsynaptic potentials (mV)")
    plt.xlim(0, 19200)
    plt.legend(loc="upper left", ncols=7, fontsize="xx-small")
    plt.savefig(f"./raw data/{key_lo}.png", dpi=500, bbox_inches="tight")
    plt.clf()
    plt.close()


  print(f"Finished plotting raw data!")

if __name__ == "__main__":
  main()