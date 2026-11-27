C Problem independent preprocessing handoff

All data are in /root/autodl-tmp/C_Problem/data/processed/independent_v1.
This archive includes scripts, metadata and QC reports, NOT the large signal
or feature arrays. Originals are unchanged in /root/autodl-tmp/dataset.

Use /root/autodl-tmp/C_Problem/.venv/bin/python on the cloud instance.

Reproduce on the instance (completed subjects are skipped):
python independent_preprocess.py --trials trial_information.csv
python independent_features.py --root /root/autodl-tmp/C_Problem/data/processed/independent_v1
python finalize_preprocessing.py

For another machine pass --source, --archive and --output to preprocessing.
The report/packaging scripts contain this instance's project root and need
that root changed when used elsewhere. Dependencies are recorded in the lock.

Task 1: use windows_2s.parquet + baseline_features_2s.npy (same row order).
Task 2: feature[:,sensor_indices,:] uses only the selected sensors. Recompute
cross-sensor gait/relative features only when all dependencies are available.
Task 3: SubjectData.read_trial() keeps a whole trial; per-trial quaternion
normalization/sign handling is implemented without changing source arrays.

No standardization is stored in the common cache. Fit scaler/feature choice/
hyperparameters using each fold's training/validation subjects only.
Feature NaNs are undefined low-energy frequency estimates, NOT missing raw
sensor values. Any fitted imputer must also be training-fold-only.

The source publication declares acceleration m/s^2. The teammate's claimed g
interpretation is unverified. No g conversion, gravity subtraction, extra
global rotation, smoothing or label alteration was applied.

Provenance: authors FudanBSRL/Comprehensive-IMU-Dataset, trial_information.csv
https://raw.githubusercontent.com/FudanBSRL/Comprehensive-IMU-Dataset/main/trial_information.csv
Dataset: https://doi.org/10.6084/m9.figshare.30234940.v1
Paper: https://doi.org/10.1038/s41597-026-06710-9
Retain authors' attribution and source license when sharing source data.
