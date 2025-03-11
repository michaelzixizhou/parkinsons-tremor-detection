import os
import multiprocessing, warnings
from preprocessors.eeg_preprocessor import EEGPreprocessor
from preprocessors.acc_preprocessor import AccelerometerPreprocessor

# Define paths to the directories
ACC_DATA_DIR = "data/accelerometer_data/"
EEG_DATA_DIR = "data/eeg_data/"
PROCESSED_ACC_DIR = "processed/accelerometer_data/"
PROCESSED_EEG_DIR = "processed/eeg_data/"

def get_matching_files(acc_data_dir, eeg_data_dir):
    """
    Function to get the matching accelerometer and EEG files.
    The data provided should already satisfy the requirement,
    but this function is added as a safety check.
    """
    acc_files = [f for f in os.listdir(acc_data_dir) if f.endswith(".pkl")]
    eeg_files = [f for f in os.listdir(eeg_data_dir) if f.endswith("-epo.fif")]

    # Extract IDs (before the first underscore) from filenames
    acc_ids = set(f.split("_")[0] for f in acc_files)
    eeg_ids = set(f.split("_")[0] for f in eeg_files)

    # Find the common IDs
    common_ids = acc_ids.intersection(eeg_ids)

    # Filter the files to only include those that have matching IDs
    matched_acc_files = [f for f in acc_files if f.split("_")[0] in common_ids]
    matched_eeg_files = [f for f in eeg_files if f.split("_")[0] in common_ids]

    return matched_acc_files, matched_eeg_files

def process_single_pair(acc_file, eeg_file, acc_data_dir, eeg_data_dir):
    """
    Function to process a single pair of accelerometer and EEG files.
    This function is used in parallel processing.

    Parameters:
    acc_file (str): The accelerometer data file to process.
    eeg_file (str): The EEG data file to process.
    acc_data_dir (str): Path to the directory containing accelerometer data files.
    eeg_data_dir (str): Path to the directory containing EEG data files.

    Returns:
    None
    """
    try:
        # Process Accelerometer data
        acc_path = os.path.join(acc_data_dir, acc_file)
        print(f"Processing Accelerometer file: {acc_file}")
        accelerometer_loader = AccelerometerPreprocessor(acc_path, frequency=100)
        timestamps = accelerometer_loader.preprocess_data()
        # accelerometer_loader.save_features()
        # accelerometer_loader.visualize_features()

        # If no timestamps found, skip EEG processing
        if len(timestamps) == 0:
            warnings.warn(f"No timestamps found in {acc_file}. Skipping EEG processing.")
            return

        # Process EEG data
        eeg_path = os.path.join(eeg_data_dir, eeg_file)
        print(f"Processing EEG file: {eeg_file}")
        eeg_loader = EEGPreprocessor(eeg_path)
        eeg_loader.preprocess()
        eeg_loader.segment_with_labels(timestamps, save=True)
        # eeg_loader.plot_epochs()
    except Exception as e:
        print(f"Error processing {acc_file} and {eeg_file}: {e}")

def process_data(acc_data_dir, eeg_data_dir, check_matching_files=True, remove_duplicates=True):
    """
    Function to process accelerometer and EEG data files in parallel.
    It will not do any visualizations. It only saves the processed EEG epochs files.
    You can then load the files and visualize them using the EEGLoader class.

    Parameters:
    acc_data_dir (str): Path to the directory containing accelerometer data files.
    eeg_data_dir (str): Path to the directory containing EEG data files.
    check_matching_files (bool): Whether to check for matching files. Default is True.
    remove_duplicates (bool): Whether to remove duplicates to avoid reprocessing files. Default is True.

    Returns:
    None
    """
    if check_matching_files:
        matched_acc_files, matched_eeg_files = get_matching_files(acc_data_dir, eeg_data_dir)
    else:
        matched_acc_files = [f for f in os.listdir(acc_data_dir) if f.endswith(".pkl")]
        matched_eeg_files = [f for f in os.listdir(eeg_data_dir) if f.endswith("-epo.fif")]

    if remove_duplicates:
        # We will scan the existing processed/eeg_data directory to avoid reprocessing those files
        existing_files = [f for f in os.listdir(eeg_data_dir) if f.endswith("-epo.fif")]
        matched_eeg_files = [f for f in matched_eeg_files if f not in existing_files]
        matched_acc_files = [f for f in matched_acc_files if f.replace(".pkl", "-epo.fif") not in existing_files]

    print(f"Found {len(matched_acc_files)} matching accelerometer files.")
    print(f"Found {len(matched_eeg_files)} matching EEG files.")

    # Use multiprocessing to process the files in parallel
    with multiprocessing.Pool() as pool:
        pool.starmap(process_single_pair, [(acc_file, eeg_file, acc_data_dir, eeg_data_dir)
                                          for acc_file, eeg_file in zip(matched_acc_files, matched_eeg_files)])

if __name__ == "__main__":
    # Run the batch processing in parallel
    process_data(ACC_DATA_DIR, EEG_DATA_DIR, check_matching_files=False)
