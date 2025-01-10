# Detecting Parkinson's Tremor using ML

This is a project under **NeuroTech UofT** that aims to detect Parkinson's tremor using machine learning. The first part of the project follows [this research paper][1] in preprocessing the public accelerometer and EEG data, then using simple ML tools such as KKN and SVM to classify it into "pre-tremor", "tremor" and "non-tremor" classes.

## Data Preprocessing

First, we use the accelerometer data to extract tremor offset times, which are labels to be used in the classification. This is implemented in `acc_preprocessor.py` Please refer to the paper for the complete algorithm:

![Accelerometer Preprocessing](resources/acc_preprocess.png)

The EEG data is first preprocessed by filtering and applying ICA. Then, we use the extracted timestamps for tremor onset and offset to label and segment the EEG data into "pre-tremor", "tremor" and "non-tremor" classes. The results are saved in the `processed/eeg_data` directory. This is implemented in `eeg_preprocessor.py`.

## Feature Extraction

After preprocessing the data, you can load the eeg_data from `processed/eeg_data` using the `EEGDataLoader` class in `eeg_loader.py`. This class contains all the feature extraction methods and provide methods to obtain the features and labels for classification directly.

The following features are then extracted from each segment:

![Feature Extraction](resources/features.png)

## Classification

Finally, we train various models using these features and labels...

[1]: https://bmcneurol.biomedcentral.com/articles/10.1186/s12883-023-03468-0
