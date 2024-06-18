# EEG-FC-Analysis-NMCL-1

This repository contains the implementation for a comparative analysis of two different methods of functional connectivity. Functional connectivity is a quantitative measure of how different regions of the brain interact with each other, which is to classify between two levels of mental workload. Specifically, we are evaluating the performance of two FC methods, namely phase lag index (PLI) and coherence, in mental workload classification of EEG data. 

* Note, the formal report for this project can be found in the repository's base directory, titled `nmcl-project1-report-minsi-hu.pdf`.

## Table of Contents
- [Installation and Usage](#Installation-and-Usage)
- [Methodology](#Methodology)
- [Results and Discussion](#Results-and-Discussion)
- [Credits and Acknowledgements](#Credits-and-Acknowledgements)

## Installation and Usage
To run the programs in the repository is simple. Simply clone the repository into your local directory and you can run the python files located in the repository. Here's a list of the files you can run and what they do:
| File name | Description |
| --- | --- |
| `classification.py` | Performs classification of high or low mental workload on EEG features extracted from feature_extraction.py |
|`feature_extraction.py` | Reads in EEG data and extracts features to be used in classification |
| `plot cleaned data.py` | Plots cleaned EEG data |
| `plot raw data.py` | Plots raw EEG data |
| `plot results` | Plots the performance of different classifiers in classifying high/low mental workload of EEG data |

## Methodology
The methodology behind the project is the following: read the STEW raw EEG dataset and clean/process the data. Then, compute PLI and coherence features and use those features to classify high vs. low mental workload. Finally, use those results to determine whether the PLI or coherence features were more effective for classification.

## Results and Discussion
We observe from the results that the coherence features were much more effective in classification between two levels of mental workload. Using coherence features alone yielded an “acceptable” accuracy of around 85% while using only PLI features gave an accuracy of around 65%. This result is consistent across the different ML algorithms we used, namely SVM, RFC, LOG, and MLP. 

However, there were many limitations of the methodology that likely influenced the results. The main limitation/inconsistency is that we did not extract frequency ranges for computing the PLI features as we did for the coherence features, which may help to explain the huge difference between their effectiveness. My next project will definitely take this into account, and compute PLI features for different frequency ranges. I will also look into more methods of functional connectivity, using convolutional neural networks for classification, experimenting with different time intervals, etc.

## Credits and Acknowledgements
Credits to Wei Lun Lim, Olga Sourina, Lipo Wang for the [Simultaneous Task EEG Workload Dataset](https://dx.doi.org/10.21227/44r8-ya50) (STEW Dataset). The STEW Dataset consists of raw EEG data from 48 male subjects under rest and workload conditions. The full dataset can be found in this repository inside of the data directory.

Credit to Muhammad Salman Kabir for [his implementation](https://github.com/5a7man/eeg_fConn) of different functional connectivity methods, which my implementations drew heavy inspiration from.

Special thanks to everyone in the [The Neuromotor Control and Learning (NMCL) Laboratory](https://sph.umd.edu/research-impact/laboratories-projects-and-programs/neuromotor-control-and-learning-laboratory) for their invaluable guidance. 

Finally, special thanks to Arya Teymourlouei for being an incredible mentor and friend, and helping me with this project.
