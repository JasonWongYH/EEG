[load_electrode_coordinates] Mean radius: 0.1032583036 meters
compute_laplace_beltrami: 0.13 seconds
NOTE: pick_channels() is a legacy function. New code should use inst.pick(...).
Filtering raw data in 1 contiguous segment
Setting up band-pass filter from 1 - 40 Hz

FIR filter parameters
---------------------
Designing a one-pass, zero-phase, non-causal bandpass filter:
- Windowed time-domain design (firwin) method
- Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation
- Lower passband edge: 1.00
- Lower transition bandwidth: 1.00 Hz (-6 dB cutoff frequency: 0.50 Hz)
- Upper passband edge: 40.00 Hz
- Upper transition bandwidth: 10.00 Hz (-6 dB cutoff frequency: 45.00 Hz)
- Filter length: 13517 samples (3.300 s)

[Parallel(n_jobs=1)]: Done  17 tasks      | elapsed:    1.3s
[load_preprocess_gdf] raw.info.ch_names: ['Fp1', 'Fp2', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10', 'FC5', 'FC1', 'FC2', 'FC6', 'T9', 'T7', 'C3', 'C4', 'T8', 'T10', 'CP5', 'CP1', 'CP2', 'CP6', 'P9', 'P7', 'P3', 'Pz', 'P4', 'P8', 'P10', 'O1', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'POz']
[load_preprocess_gdf] raw.info.sfreq: 256.0
[load_preprocess_gdf] Data snippet (shape: (5, 5)):
[[ 1.15016340e-16  8.00038613e-12  1.27415875e-11  1.30619341e-11
   1.03543949e-11]
 [ 9.23109672e-17  9.69276016e-12  1.52902173e-11  1.56674923e-11
   1.29649778e-11]
 [-2.37787666e-17  9.35959567e-12  1.62477948e-11  1.88140739e-11
   1.66182762e-11]
 [ 8.77509462e-17  1.11264126e-11  1.72922776e-11  1.69674658e-11
   1.26469321e-11]
 [ 9.68416678e-17  8.84830006e-12  1.37712854e-11  1.35706094e-11
   1.03509875e-11]]
[load_preprocess_gdf] Times (sec): [0.         0.00390625 0.0078125  0.01171875 0.015625  ]   
Used Annotations descriptions: ['255', '33124', '33126', '33127', '33132', '33289', '33296']
Not setting metadata
10 matching events found
No baseline correction applied
0 projection items activated
Using data from preloaded Raw for 10 events and 385 original time points ...
0 bad epochs dropped
Used Annotations descriptions: ['255', '33124', '33126', '33127', '33132', '33289', '33296']
Not setting metadata
90 matching events found
No baseline correction applied
0 projection items activated
Using data from preloaded Raw for 90 events and 385 original time points ...
0 bad epochs dropped
NOTE: pick_channels() is a legacy function. New code should use inst.pick(...).
Filtering raw data in 1 contiguous segment
Setting up band-pass filter from 1 - 40 Hz

FIR filter parameters
---------------------
Designing a one-pass, zero-phase, non-causal bandpass filter:
- Windowed time-domain design (firwin) method
- Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation
- Lower passband edge: 1.00
- Lower transition bandwidth: 1.00 Hz (-6 dB cutoff frequency: 0.50 Hz)
- Upper passband edge: 40.00 Hz
- Upper transition bandwidth: 10.00 Hz (-6 dB cutoff frequency: 45.00 Hz)
- Filter length: 13517 samples (3.300 s)

[Parallel(n_jobs=1)]: Done  17 tasks      | elapsed:    0.8s
[load_preprocess_gdf] raw.info.ch_names: ['Fp1', 'Fp2', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10', 'FC5', 'FC1', 'FC2', 'FC6', 'T9', 'T7', 'C3', 'C4', 'T8', 'T10', 'CP5', 'CP1', 'CP2', 'CP6', 'P9', 'P7', 'P3', 'Pz', 'P4', 'P8', 'P10', 'O1', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'POz']
[load_preprocess_gdf] raw.info.sfreq: 256.0
[load_preprocess_gdf] Data snippet (shape: (5, 5)):
[[ 1.66171082e-17 -6.64868990e-12 -8.60037143e-12 -5.04774361e-12
   5.48615357e-13]
 [-2.96635366e-18 -3.12848556e-13  1.10602837e-12  4.50330922e-12
   8.35919694e-12]
 [ 1.71233860e-17 -1.11296808e-11 -1.73951354e-11 -1.70725218e-11
  -1.24380802e-11]
 [ 1.76234095e-17 -6.02169533e-12 -1.02310003e-11 -1.18231921e-11
  -1.12453112e-11]
 [ 1.15668654e-17 -4.42615318e-12 -7.38162522e-12 -8.49421560e-12
  -8.53912137e-12]]
[load_preprocess_gdf] Times (sec): [0.         0.00390625 0.0078125  0.01171875 0.015625  ]   
Used Annotations descriptions: ['255', '33126', '33127', '33132', '33133', '33289', '33296']
Not setting metadata
90 matching events found
No baseline correction applied
0 projection items activated
Using data from preloaded Raw for 90 events and 385 original time points ...
0 bad epochs dropped
/home/jason/EEG/Libet-features-classifiers.py:254: RuntimeWarning: Concatenation of Annotations within Epochs is not supported yet. All annotations will be dropped.
  target_epochs = mne.concatenate_epochs([target_epochs,extract_epochs(raw_other,{'33127':4})])
Not setting metadata
100 matching events found
No baseline correction applied
/home/jason/EEG/Libet-features-classifiers.py:512: RuntimeWarning: Concatenation of Annotations within Epochs is not supported yet. All annotations will be dropped.
  all_epochs = mne.concatenate_epochs([target_epochs, nontarget_epochs])
Not setting metadata
190 matching events found
No baseline correction applied
/home/jason/EEG/Libet-features-classifiers.py:614: RuntimeWarning: This filename (wavelet-misclassified.fif) does not conform to MNE naming conventions. All epochs files should end with -epo.fif, -epo.fif.gz, _epo.fif or _epo.fif.gz
  mis_epochs.save(mis_fif, overwrite=True)
Overwriting existing file.
/home/jason/EEG/Libet-features-classifiers.py:614: RuntimeWarning: epochs.drop_log contains 106172 entries which will incur up to a 1.3 MiB writing overhead (per split file), consider using epochs.reset_drop_log_selection() prior to writing
  mis_epochs.save(mis_fif, overwrite=True)
Overwriting existing file.
/home/jason/EEG/Libet-features-classifiers.py:615: RuntimeWarning: This filename (wavelet-correct.fif) does not conform to MNE naming conventions. All epochs files should end with -epo.fif, -epo.fif.gz, _epo.fif or _epo.fif.gz
  correct_epochs.save(correct_fif, overwrite=True)
Overwriting existing file.
/home/jason/EEG/Libet-features-classifiers.py:615: RuntimeWarning: epochs.drop_log contains 106172 entries which will incur up to a 1.3 MiB writing overhead (per split file), consider using epochs.reset_drop_log_selection() prior to writing
  correct_epochs.save(correct_fif, overwrite=True)
Overwriting existing file.
[save_epochs_and_labels_by_indices] Saved 29 misclassified and 161 correct epochs and labels.
[save_classifier] Saved classifier and scaler to wavelet-model.pkl
[classify_wavelet_power] Mean accuracy: 0.847
Wavelet Power Accuracy: 0.847
/home/jason/EEG/Libet-features-classifiers.py:531: RuntimeWarning: Concatenation of Annotations within Epochs is not supported yet. All annotations will be dropped.
  all_epochs = mne.concatenate_epochs([target_epochs, nontarget_epochs])
Not setting metadata
190 matching events found
No baseline correction applied
Computing rank from data with rank=None
    Using tolerance 0.00086 (2.2e-16 eps * 10 dim * 3.9e+11  max singular value)
    Estimated rank (data): 10
    data: rank 10 computed from 10 data channels with 0 projectors
Reducing data rank from 10 -> 10
Estimating class=0.0 covariance using EMPIRICAL
Done.
Estimating class=1.0 covariance using EMPIRICAL
Done.
/home/jason/EEG/Libet-features-classifiers.py:614: RuntimeWarning: This filename (laplacian_csp-misclassified.fif) does not conform to MNE naming conventions. All epochs files should end with -epo.fif, -epo.fif.gz, _epo.fif or _epo.fif.gz
  mis_epochs.save(mis_fif, overwrite=True)
Overwriting existing file.
/home/jason/EEG/Libet-features-classifiers.py:614: RuntimeWarning: epochs.drop_log contains 106172 entries which will incur up to a 1.3 MiB writing overhead (per split file), consider using epochs.reset_drop_log_selection() prior to writing
  mis_epochs.save(mis_fif, overwrite=True)
Overwriting existing file.
/home/jason/EEG/Libet-features-classifiers.py:615: RuntimeWarning: This filename (laplacian_csp-correct.fif) does not conform to MNE naming conventions. All epochs files should end with -epo.fif, -epo.fif.gz, _epo.fif or _epo.fif.gz
  correct_epochs.save(correct_fif, overwrite=True)
Overwriting existing file.
/home/jason/EEG/Libet-features-classifiers.py:615: RuntimeWarning: epochs.drop_log contains 106172 entries which will incur up to a 1.3 MiB writing overhead (per split file), consider using epochs.reset_drop_log_selection() prior to writing
  correct_epochs.save(correct_fif, overwrite=True)
Overwriting existing file.
[save_epochs_and_labels_by_indices] Saved 12 misclassified and 178 correct epochs and labels.
[save_classifier] Saved classifier and scaler to laplacian_csp-model.pkl
[classify_laplacian_csp_lda] Mean accuracy: 0.937
Laplacian CSP+LDA Accuracy: 0.937
/home/jason/EEG/Libet-features-classifiers.py:559: RuntimeWarning: Concatenation of Annotations within Epochs is not supported yet. All annotations will be dropped.
  all_epochs = mne.concatenate_epochs([target_epochs, nontarget_epochs])
Not setting metadata
190 matching events found
No baseline correction applied
/home/jason/EEG/Libet-features-classifiers.py:614: RuntimeWarning: This filename (laplacian_fgmdm-misclassified.fif) does not conform to MNE naming conventions. All epochs files should end with -epo.fif, -epo.fif.gz, _epo.fif or _epo.fif.gz
  mis_epochs.save(mis_fif, overwrite=True)
Overwriting existing file.
/home/jason/EEG/Libet-features-classifiers.py:614: RuntimeWarning: epochs.drop_log contains 106172 entries which will incur up to a 1.3 MiB writing overhead (per split file), consider using epochs.reset_drop_log_selection() prior to writing
  mis_epochs.save(mis_fif, overwrite=True)
Overwriting existing file.
/home/jason/EEG/Libet-features-classifiers.py:615: RuntimeWarning: This filename (laplacian_fgmdm-correct.fif) does not conform to MNE naming conventions. All epochs files should end with -epo.fif, -epo.fif.gz, _epo.fif or _epo.fif.gz
  correct_epochs.save(correct_fif, overwrite=True)
Overwriting existing file.
/home/jason/EEG/Libet-features-classifiers.py:615: RuntimeWarning: epochs.drop_log contains 106172 entries which will incur up to a 1.3 MiB writing overhead (per split file), consider using epochs.reset_drop_log_selection() prior to writing
  correct_epochs.save(correct_fif, overwrite=True)
Overwriting existing file.
[save_epochs_and_labels_by_indices] Saved 41 misclassified and 149 correct epochs and labels.
[save_classifier] Saved classifier and scaler to laplacian_fgmdm-model.pkl
[classify_laplacian_fgmdm] Mean accuracy: 0.784
Laplacian FgMDM Accuracy: 0.784