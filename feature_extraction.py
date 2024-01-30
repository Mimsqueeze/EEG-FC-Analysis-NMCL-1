# Imports
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as ss
import pandas as pd
import networkx as nx

# ------------------------- author: Minsi Hu -------------------------

# ----------------------- program description: -----------------------
#              Reads in EEG data and extracts features to 
#                      be used in classification
# --------------------------------------------------------------------

# GLOBAL VARIABLES
DISABLE_PLI = True # Set to true if you only want coherence features
DISABLE_COH = False # Set to true if you only want pli features
FEATURE_TYPE = "COH" # Set for file names prefix ("PLI", "COH", "PLI_COH")
SAVE_FEATURES = False # Set to true if you want to save the features to file

"""
The pli(num_sensors, data) function
  Computes PLI Connectivity

Parameters
  num_sensors : int
    number of sensors used for EEG
  data : numpy.ndarray
    the EEG data itself inside a numpy.ndarray, where each column is a 
    sensor, each row is a point in time, and each entry is a float value
    representing the postsynaptic potentials at a given time

Returns
  pli_m : numpy.ndarray
    The PLI connectivity/adjacency matrix for given data, where each row
    and column is a sensor and each entry is a float value for the pli 
    connectivity of two given sensors
  pli_v : numpy.ndarray
    The PLI connectivity vector for given data containing the unique
    information from the connectivity matrix
  pli_g : numpy.ndarray
    Computed graph features from the PLI connectivity matrix
"""
def pli(num_sensors, data):
  # Declare empty connectivity matrix
  pli_m = np.zeros([num_sensors, num_sensors], dtype=float)
  
  # Computing hilbert transform to extract complex components from data, 
  # giving us data points in the complex plane. This allows us to estimate 
  # phase angle by taking the inverse tangent of the complex component 
  # over the real component
  data_hilbert = np.imag(ss.hilbert(data))

  # Compute the instantaneous phase angle of data points in radians
  phase = np.arctan(data_hilbert/data)

  # Find number of data points which is the number of rows
  num_points = np.shape(data)[0]
  
  # Fill connectivity matrix by computing pli for every pair of sensors
  for i in range(num_sensors):
    for k in range(num_sensors):
      # Compute pli by calculating the phase difference of all points,
      # applying the signum function, and then computing the average of that
      pli_m[i,k] = np.abs(np.sum(np.sign(phase[:,i]-phase[:,k])))/num_points

  # Computing PLI connectivity vector to reduce dimensionality by only
  # keeping unique information
  pli_v = pli_m[np.triu_indices(pli_m.shape[0], k=1)]

  # Compute graph features
  pli_g = get_graph_features(pli_m)

  # Return the pli connectivity matrix, vector, and graph features
  return pli_m, pli_v, pli_g

"""
The coherence(num_sensors, data, f_min, f_max, sfreq) function
  Computes coherence statistic

Parameters
  num_sensors : int
    number of sensors used for EEG
  data : numpy.ndarray
    the EEG data itself inside a numpy.ndarray, where each column is a 
    sensor, each row is a point in time, and each entry is a float value
    representing the postsynaptic potentials at a given time
  f_min : float
  f_max : float
  sfreq : float
    sampling frequency of the data in hz

Returns
  coh_m : numpy.ndarray
    The coherence connectivity matrix for given data, where each row
    and column is a sensor and each entry is a float value for the coherence 
    connectivity of two given sensors
  coh_v : numpy.ndarray
    The coherence connectivity vector for given data containing the unique
    information from the connectivity matrix
  coh_g : numpy.ndarray
    Computed graph features from the coherence connectivity matrix
"""
def coherence(num_sensors, data, f_min, f_max, sfreq):
  # Declare empty connectivity_matrix
  coh_m = np.zeros([num_sensors, num_sensors], dtype=float)
  
  # Fill connectivity matrix by computing coherence for every pair of 
  # sensors
  for i in range(num_sensors):
    for k in range(num_sensors):
      # Use scipy coherence function to estimate coherence
      f, Cxy = ss.coherence(data[:,i], data[:,k], fs = sfreq)

      # Filter coherence values within a specific frequency range
      coh_m[i,k] = np.mean(Cxy[np.where((f >= f_min) & (f <= f_max))])
  
  # Computing coherence connectivity vector to reduce dimensionality by only
  # keeping unique information
  coh_v = coh_m[np.triu_indices(coh_m.shape[0], k=1)]

  # Compute graph features
  coh_g = get_graph_features(coh_m)

  # Return the coherence connectivity matrix, vector, and graph features
  return coh_m, coh_v, coh_g

"""
The get_graph_features(matrix) function
  Computes graph features from an adjacency matrix

Parameters
  matrix : numpy.ndarray
    adjacency matrix for the graph

Returns
  features : numpy.ndarray
    The vector containing the graph features, including the average shortest
    path length, the closeness centrality for each node, and the clustering
    coefficient for each node
"""
def get_graph_features(matrix):
  # Convert matrix to NetworkX graph
  G = nx.from_numpy_array(matrix)

  # Initialize array to be returned
  features = []

  # Compute average shortest path length and add it to the array
  features += [nx.average_shortest_path_length(G, weight='weight')]

  # Compute closeness centrality and clustering coefficient and add them to 
  # the tuple
  for i in range(0, 14):
    features += [nx.closeness_centrality(G, u=i, distance='weight')]
    features += [nx.clustering(G, nodes=i, weight='weight')]

  return np.array(features) 

"""
The plot_m() function
  Plots the adjacency matrix given in argument and saves it to file

Parameters
  matrix : numpy.ndarray
    adjacency matrix for the graph
  title : string
    title of the plot
  filename : string
    filename of the plot
"""
def plot_m(matrix, title, filename):
  cax = plt.matshow(matrix)
  plt.title(title)
  plt.colorbar(cax)
  plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
             ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", 
              "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"])
  plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
             ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", 
              "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"])
  plt.tick_params(axis='both', which='both', labelsize=6)
  plt.savefig(f"./matrices/{filename}.png", dpi=500, bbox_inches="tight")
  plt.clf()

"""
The main() function

Reads in the data, cleans the data, and extracts features such as pli and 
coherence to be used in classification in classification.py. Saves the 
extracted features into the ./features directory, where each row is a 
5 second time frame and each column is a feature value
"""
def main():
  # Add all the data from subject 1 to 48 to the dictionary
  for i in range(39, 40):
    print(f"Creating features from Subject {i}...")

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
    data_hi = ss.sosfilt(sos, data_hi)
    data_lo = ss.sosfilt(sos, data_lo)

    # Initialize dataframe to store result of feature creation, where each
    # column is a 5 second time frame and the rows are feature values that
    # come from combining the result of PLI and Coherence vectors
    df_hi = pd.DataFrame()
    df_lo = pd.DataFrame()

    # Compute PLI and Coherence for every 5 second interval of EEG
    # for t in range(0, 19200, 640):
    for t in range(0, 640, 640):
      # Get the 5 second interval starting at t to t+640, since our sampling
      # rate is 128hz (640 = 128*5)
      interval_hi = data_hi[t:t+640,:]
      interval_lo = data_lo[t:t+640,:]

      # Compute the PLI connectivity
      pli_m_hi, pli_v_hi, pli_g_hi = pli(14, interval_hi)
      pli_m_lo, pli_v_lo, pli_g_lo = pli(14, interval_lo)
      plot_m(pli_m_hi, "PLI High", f"pli_{key_hi}")
      plot_m(pli_m_lo, "PLI Low", f"pli_{key_lo}")

      # Compute the coherence connectivity for 4-8hz (theta)
      coh_m1_hi, coh_v1_hi, coh_g1_hi = coherence(14, interval_hi, 4, 8, 128)
      coh_m1_lo, coh_v1_lo, coh_g1_lo = coherence(14, interval_lo, 4, 8, 128)
      plot_m(coh_m1_hi, "Coherence 4-8 Hz (Theta) High", f"coh_1_{key_hi}")
      plot_m(coh_m1_lo, "Coherence 4-8 Hz (Theta) Low", f"coh_1_{key_lo}")

      # Compute the coherence connectivity for 8-11hz (low-alpha)
      coh_m2_hi, coh_v2_hi, coh_g2_hi = coherence(14, interval_hi, 8, 11, 128)
      coh_m2_lo, coh_v2_lo, coh_g2_lo = coherence(14, interval_lo, 8, 11, 128)
      plot_m(coh_m2_hi, "Coherence 8-11 Hz (Low-Alpha) High", f"coh_2_{key_hi}")
      plot_m(coh_m2_lo, "Coherence 8-11 Hz (Low-Alpha) Low", f"coh_2_{key_lo}")

      # Compute the coherence connectivity for 11-13hz (high-alpha)
      coh_m3_hi, coh_v3_hi, coh_g3_hi = coherence(14, interval_hi, 11, 13, 128)
      coh_m3_lo, coh_v3_lo, coh_g3_lo = coherence(14, interval_lo, 11, 13, 128)
      plot_m(coh_m3_hi, "Coherence 11-13 Hz (High-Alpha) High", f"coh_3_{key_hi}")
      plot_m(coh_m3_lo, "Coherence 11-13 Hz (High-Alpha) Low", f"coh_3_{key_lo}")

      # Compute the coherence connectivity for 13-30hz (beta)
      coh_m4_hi, coh_v4_hi, coh_g4_hi = coherence(14, interval_hi, 13, 30, 128)
      coh_m4_lo, coh_v4_lo, coh_g4_lo = coherence(14, interval_lo, 13, 30, 128)
      plot_m(coh_m4_hi, "Coherence 13-30 Hz (Beta) High", f"coh_4_{key_hi}")
      plot_m(coh_m4_lo, "Coherence 13-30 Hz (Beta) Low", f"coh_4_{key_lo}")

      # Compute the coherence connectivity for 30-40hz (gamma)
      coh_m5_hi, coh_v5_hi, coh_g5_hi = coherence(14, interval_hi, 30, 40, 128)
      coh_m5_lo, coh_v5_lo, coh_g5_lo = coherence(14, interval_lo, 30, 40, 128)
      plot_m(coh_m5_hi, "Coherence 30-40 Hz (Gamma) High", f"coh_5_{key_hi}")
      plot_m(coh_m5_lo, "Coherence 30-40 Hz (Gamma) Low", f"coh_5_{key_lo}")

      # Smush all connectivity vectors into a single vector
      combined_hi = None
      combined_lo = None
      if DISABLE_PLI:
        combined_hi = np.concatenate(
        (coh_v1_hi, coh_v2_hi, coh_v3_hi, coh_v4_hi, coh_v5_hi,
         coh_g1_hi, coh_g2_hi, coh_g3_hi, coh_g4_hi, coh_g5_hi), 
        axis=0)
        combined_lo = np.concatenate(
        (coh_v1_lo, coh_v2_lo, coh_v3_lo, coh_v4_lo, coh_v5_lo,
         coh_g1_lo, coh_g2_lo, coh_g3_lo, coh_g4_lo, coh_g5_lo), 
        axis=0)
      elif DISABLE_COH: 
        combined_hi = np.concatenate(
          (pli_v_hi, pli_g_hi), axis=0)
        combined_lo = np.concatenate(
          (pli_v_lo, pli_g_lo), axis=0)
      else:
        combined_hi = np.concatenate(
          (pli_v_hi, coh_v1_hi, coh_v2_hi, coh_v3_hi, coh_v4_hi, coh_v5_hi,
          pli_g_hi, coh_g1_hi, coh_g2_hi, coh_g3_hi, coh_g4_hi, coh_g5_hi), 
          axis=0)
        combined_lo = np.concatenate(
          (pli_v_lo, coh_v1_lo, coh_v2_lo, coh_v3_lo, coh_v4_lo, coh_v5_lo,
          pli_g_lo, coh_g1_lo, coh_g2_lo, coh_g3_lo, coh_g4_lo, coh_g5_lo), 
          axis=0)
      
      # Assign combined vector to dataframe column corresponding to the
      # time frame
      df_hi[f"S{i}: {5*t/640}-{5 + 5*t/640}"] = combined_hi
      df_lo[f"S{i}: {5*t/640}-{5 + 5*t/640}"] = combined_lo

    # Transpose the dataframes to get 5 second time frames as the rows and
    # each features values on the columns
    df_hi = df_hi.T
    df_lo = df_lo.T

    # Save the features to file
    if SAVE_FEATURES:
      print(f"Saving features from Subject {i}!")
      df_hi.to_csv(f"./features/{FEATURE_TYPE} {key_hi}.csv")
      df_lo.to_csv(f"./features/{FEATURE_TYPE} {key_lo}.csv")

if __name__ == "__main__":
  main()