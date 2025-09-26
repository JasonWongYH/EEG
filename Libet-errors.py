# --- Misclassification Extraction ---
def get_misclassified_indices(X, y, clf, cv=5):
    kf = KFold(cv)
    mis_idx = []
    for train_idx, test_idx in kf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        mis_idx.extend(test_idx[y_pred != y[test_idx]])
    return np.unique(mis_idx)

def extract_phase_itc_features(epochs, freqs=np.arange(12, 31, 2), n_cycles=5, picks=None):
    """
    Compute phase and ITC (inter-trial coherence) using Morlet wavelets.
    Returns: (n_epochs, n_freqs) average phase and ITC per frequency
    """
    # Power: per-epoch
    power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        average=False,
        picks=picks,
        decim=2,
        n_jobs=1,
        verbose=False
    )
    # ITC: averaged across epochs
    itc = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        average=True,
        picks=picks,
        decim=2,
        n_jobs=1,
        verbose=False
    )[1]  # [1] is ITC

    # Phase: get angle of complex power
    complex_data = power.data  # shape: (n_epochs, n_channels, n_freqs, n_times)
    phase = np.angle(complex_data)
    itc_data = itc.data  # (n_channels, n_freqs, n_times)

    # Mean phase per epoch, averaged across time and channels
    phase_features = np.mean(phase, axis=(1, 3))  # shape: (n_epochs, n_freqs)
    # ITC is global across epochs, so replicate per trial
    itc_features = np.tile(np.mean(itc_data, axis=2).T, (phase_features.shape[0], 1))  # (n_epochs, n_freqs)

    return phase_features, itc_features

def classify_phase_itc(target_epochs, nontarget_epochs, picks=motor_channels, prefix="phase_itc"):
    """
    Classify epochs using phase and ITC features from Morlet decomposition.
    """
    phase_t, itc_t = extract_phase_itc_features(target_epochs, picks=picks)
    phase_nt, itc_nt = extract_phase_itc_features(nontarget_epochs, picks=picks)

    X = np.concatenate([np.hstack([phase_t, itc_t]), np.hstack([phase_nt, itc_nt])])
    y = np.concatenate([np.ones(len(phase_t)), np.zeros(len(phase_nt))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_epochs = mne.concatenate_epochs([target_epochs, nontarget_epochs])
    mis_idx, correct_idx = [], []
    accs = []
    for train_idx, test_idx in skf.split(X_scaled, y):
        clf.fit(X_scaled[train_idx], y[train_idx])
        y_pred = clf.predict(X_scaled[test_idx])
        accs.append(np.mean(y_pred == y[test_idx]))
        mis_idx.extend(test_idx[y_pred != y[test_idx]])
        correct_idx.extend(test_idx[y_pred == y[test_idx]])
    mis_idx = np.unique(mis_idx)
    correct_idx = np.unique(correct_idx)
    mis_fif, correct_fif = save_epochs_by_indices(all_epochs, mis_idx, correct_idx, prefix)
    save_classifier(clf, scaler, prefix)
    print(f"[classify_phase_itc] Mean accuracy: {np.mean(accs):.3f}")
    return np.mean(accs), mis_idx, correct_idx, mis_fif, correct_fif

def get_misclassified_epochs(X, y, clf=None, cv=5):
    """Identify misclassified epochs using cross-validation"""
    if clf is None:
        clf = LogisticRegression(max_iter=1000)
    
    misclassified_indices = []
    for train_idx, test_idx in KFold(cv).split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        misclassified_indices.extend(test_idx[y_pred != y[test_idx]])
    
    return np.unique(misclassified_indices)

# Phase+ITC
phase_itc_acc, phase_itc_mis_idx, phase_itc_correct_idx, phase_itc_mis_fif, phase_itc_correct_fif = classify_phase_itc(target_epochs, nontarget_epochs, picks=motor_channels, prefix="phase_itc")
print(f"Phase+ITC Accuracy: {phase_itc_acc:.3f}")

def wavelet_feature_extractor(epoch):
    return extract_wavelet_features(epoch, picks=motor_channels)

print(target_epochs.copy().pick(picks).get_data().shape)
(100, 7, 385)