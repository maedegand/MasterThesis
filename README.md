# Location-aware iEEG signal analysis for epileptic patients

This repository contains the code used for my master's thesis at UCLouvain. The objective is to develop a semi-automated preprocessing pipeline for iEEG signal from epileptic patients.
For further details on the methodology, refer to Annex B of my master thesis, which is included in this repository.

# Requirements

1. Make sure to have Python 3.10 and pip available on your computer:

   ```
   # Both of these commands should return the versions of your Python and pip
   python --version
   python -m pip --version
   ```
2. Install the required libraries:

   ```
   python -m pip install -r requirements.txt
   ```
3. Install the ElectroLoc module and its requirements from this following adress:

   https://github.com/Quent-DL/MasterThesis_ElectroLoc.git

# Repository structure

```
.
├── contact_extraction/
│   ├── main_contact_extraction.py   # Pipeline entry point
│   ├── nii_preprocessing.py         # CT/MRI mask generation and co-registration
│   ├── edf_preprocessing.py         # EDF loading and electrode extraction
│   ├── csv_preprocessing.py         # Entry point detection and CSV handling
│   ├── atlas.py                     # Atlas co-registration and label dilation
│   ├── labeling.py                  # Contact numbering and atlas-based labeling
│   └── shift_handling.py            # Electrode shift correction
│
├── seizure_detection/
│   ├── main_seizure_detection.py    # Pipeline entry point
│   └── library_SEEG.py              # Segmentation, scalogram generation, model
│
└── README.md
```

# Running

## Contact extraction module

This module localizes iEEG electrode contacts from a post-implantation CT scan, co-registers them to the patient's MRI, and labels each contact according to the AAL atlas.

```powershell
$CT    = "CT_path.nii.gz"
$MRI   = "MRI_path.nii.gz"
$MASK  = "brain_mask_CT_path.nii.gz"
$EDF   = "iEEG_path.edf"
$OUT   = "folder_out/"

python main_contact_extraction.py $CT $MRI $MASK $EDF -o $OUT
```

By default the AAL atlas bundled with the repository is used. To supply a custom atlas, all four atlas flags must be provided together:

```powershell
python main_contact_extraction.py $CT $MRI $MASK $EDF -o $OUT `
    --mni MNI152.nii.gz `
    --atlas my_atlas.nii.gz `
    --atlas-txt my_atlas_labels.txt `
    --brain-mask-mni brain_mask_mni.nii.gz
```

The ElectroLoc parent directory can also be overridden:

```powershell
python main_contact_extraction.py $CT $MRI $MASK $EDF -o $OUT `
    --electroloc path/to/MasterThesis_ElectroLoc
```

### Output structure

```
folder_out/
├── coregistration/          # Co-registered volumes and mapping files
├── input_ElectroLoc/        # Electrode info and entry point coordinates
├── mask_generated/          # Electrode and surface masks
├── output_ElectroLoc.csv    # Raw contact coordinates from ElectroLoc
├── contacts_shifted_corrected.csv   # Final contact coordinates after shift correction
├── contact_shifted_corrected.nii.gz # Contact mask in CT space
└── labeled_contacts.csv     # Contacts with AAL region labels
```

## Seizure detection module

This module preprocesses the iEEG signal into scalograms, embeds them as patch tokens, and runs a pre-trained transformer-based model to predict seizure probability for each 5-second window.

```bash
python main_seizure_detection.py -i iEEG_path.edf -o folder_out/
```

By default, the pretrained weights from https://github.com/tbary/EEGPreTrainingDatasets.git is used. To supply custom pretrained weights, the following command can be used. (Be carefull, the pretrained weights have to be trained on the same model architecture as the one in this work).

```bash
python main_seizure_detection.py -i iEEG_path.edf -o folder_out/ \
    --weights path/to/checkpoint
```

### Output structure

```
folder_out/
├── segments/                         # Temporary 5s bipolar segments (.npy)
├── scalograms/                       # Temporary scalogram files (.mat)
├── X_all.npy                         # Tokenized model inputs
├── y_all.npy                         # Labels
├── predictions.csv                   # Per-window seizure probabilities
├── preprocessing_step_counts.csv     # Window count at each pipeline step
└── model/                            # Saved model
```

# Optional parameters

Both modules accept additional parameters to override defaults. Full details are available via:

```bash
python main_contact_extraction.py --help
python main_seizure_detection.py --help
```

| Module             | Parameter                                                        | Default                  | Description                   |
| ------------------ | ---------------------------------------------------------------- | ------------------------ | ----------------------------- |
| seizure_detection  | `--channels`                                                   | 122                      | Number of iEEG channels       |
| seizure_detection  | `--segment-duration`                                           | 5                        | Segment duration (seconds)    |
| seizure_detection  | `--downsample-fs`                                              | 1024                     | Downsampling frequency (Hz)   |
| seizure_detection  | `--segment-dir`                                                | `<output>/segments/`   | Temporary segment directory   |
| seizure_detection  | `--scalogram-dir`                                              | `<output>/scalograms/` | Temporary scalogram directory |
| contact_extraction | `--electroloc`                                                 | see config               | ElectroLoc parent directory   |
| contact_extraction | `--mni` / `--atlas` / `--atlas-txt` / `--brain-mask-mni` | built-in AAL             | Custom atlas files            |

# Manual electrode identification

During the running of the contact extraction module, an interactive prompt will ask for two CSV files that must be prepared externally using ElecTool, based on the entry point coordinates generated in the preprocessing step:

- **`electrode_info.csv`** — required columns: `nb_contacts`, `vox_x`, `vox_y`, `vox_z`
- **`electrode_info_world.csv`** — required columns: `electrode_id`, `nb_contacts`, `world_x`, `world_y`, `world_z`

These files encode the manually identified tip and entry point of each electrode shaft in the CT volume. The pipeline will pause and display a prompt in the terminal; simply enter the path to each file when asked.

# Notes

* Input CT and MRI volumes must be **skull-stripped** before running the contact extraction pipeline.

# Credits

This project builds on the following external repositories:

- **ElectroLoc** — electrode localization algorithm used in the contact extraction pipeline. 
  Source: [github.com/Quent-DL/MasterThesis_ElectroLoc](https://github.com/Quent-DL/MasterThesis_ElectroLoc.git)
- **Transformed-based model** for EEG signals — `library.py` and the transformer model weights used in the seizure detection pipeline are adapted from this repository.
  Source: [github.com/tbary/EEGPreTrainingDatasets](https://github.com/tbary/EEGPreTrainingDatasets.git)
