import mne
import re
import os
from collections import defaultdict
def load_edf(edf_path, output_dir):
    """
    Load an EDF file and return the data as a NumPy array.
    Args:
        path_edf (str): The path to the EDF file.
        output_dir (str): The output directory where the info file will be saved.
    Returns:
            numpy.ndarray: The data from the EDF file.
            
    """
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False,encoding="latin1")
    output_path = os.path.join(output_dir, os.path.basename(edf_path) + "_info.txt")  # Create a path for the output text file
    with open(output_path, 'w') as f:
        f.write("EDF File Information:\n")  # Write a header to the file
        f.write(str(raw.info))  # Save the info to a text file
        f.write("\n Channel Names:\n")  # Write a header for the channel names
        f.write(str(raw.ch_names))  # Save the channel names to the text file
    print(f"Edf info saved in {output_path}")  # Print a message indicating where the info was saved
    return raw.get_data(), raw.ch_names


def extract_electrodes(channel_names, path_edf,output_dir):
    """
    Extract the information about the electrodes from the EDF data : 
        nb_contacts = number of contacts per electrodes.
    Args:
        data_edf (numpy.ndarray): The data from the EDF file.
        channel_names (list): The list of channel names.
    Returns:
        csv file: A CSV file containing the electrode information
    """
    electrodes = defaultdict(list)
    output_path = os.path.join(output_dir, os.path.basename(path_edf) + "_electrode_info.csv")  
    for ch in channel_names:
        if ch.startswith("EEG"):
            name = ch.replace("EEG","").strip()
            match =re.match(r"([A-Z]+)(\d+)", name)
            if match:
                electrode_name = match.group(1)
                contact_number = int(match.group(2))
                electrodes[electrode_name].append(contact_number)
    with open(output_path, "w") as f:
        for electrode, contacts in electrodes.items():
            f.write(f"{electrode},{len(contacts)}\n")
    print(f"Electrode information saved in {output_path}")
    return output_path
