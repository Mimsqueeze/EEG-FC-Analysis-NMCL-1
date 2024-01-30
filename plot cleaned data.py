# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss

# ----------------------- author: Minsi Hu -----------------------

# ----------------------- program description: -----------------------
#                         Plots cleaned EEG data
# --------------------------------------------------------------------

"""
The main() function

Reads in the EEG data, applys a butterworth filter, plots them, and
saves it into the ./cleaned data/ directory
"""
def main():
  print("Plotting cleaned data...")

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

    # Create butterworth filter: bandpass 1-50hz, 6th order, 128hz sampling
    sos = ss.butter(
      N=6, Wn=[1, 50], btype='bandpass', analog=False, output='sos', fs=128)
    
    # Apply the filter to clean the data
    data_hi = pd.DataFrame(ss.sosfilt(sos, data_hi))
    data_lo = pd.DataFrame(ss.sosfilt(sos, data_lo))

    # Rename the columns
    column_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", 
                    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    data_hi.columns = column_names
    data_lo.columns = column_names

    # Plot the cleaned eeg data
    data_hi.plot(y=column_names, kind="line", legend=True)
    plt.title(f"Cleaned EEG Data for Subject {i} High Workload")
    plt.xlabel("Time (1/128s)")
    plt.ylabel("Postsynaptic potentials (mV)")
    plt.xlim(0, 19200)
    plt.legend(loc="upper left", ncols=7, fontsize="xx-small")
    plt.savefig(f"./cleaned data/{key_hi}.png", dpi=500, bbox_inches="tight")
    plt.clf()
    plt.close()

    data_lo.plot(y=column_names, kind="line", legend=True)
    plt.title(f"Cleaned EEG Data for Subject {i} Low Workload")
    plt.xlabel("Time (1/128s)")
    plt.ylabel("Postsynaptic potentials (mV)")
    plt.xlim(0, 19200)
    plt.legend(loc="upper left", ncols=7, fontsize="xx-small")
    plt.savefig(f"./cleaned data/{key_lo}.png", dpi=500, bbox_inches="tight")
    plt.clf()
    plt.close()

  print(f"Finished plotting cleaned data!")

if __name__ == "__main__":
  main()