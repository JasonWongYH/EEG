import joblib
import mne
from sklearn.metrics import accuracy_score

# 0. time.time() each preprocessing, feature extraction, classification step
# 1. **Load test epochs for each classifier**  
#    - Load both correctly and misclassified epochs for each classifier from your saved files (e.g., `.npy` or `.fif`).

# 2. **Identify overlapping misclassified epochs**  
#    - Compare the indices of misclassified epochs across classifiers to find overlaps.

# 3. **Construct an ensemble guided by uncertainty**  
#    - Use classifier probability/confidence outputs to define uncertainty.
#    - Combine predictions (e.g., majority vote, weighted by uncertainty) and test ensemble accuracy.

# 4. **Obtain and plot intermediate probabilities**  
#    - Extract probability/uncertainty estimates at each step in the classifier’s decision process.
#    - Plot these for selected epochs to visualize decision evolution.
    
def extract_wavelet_features(epochs, picks=None, freqs=np.arange(12, 31, 2), n_cycles=5):
    if picks is None:
        picks = epochs.ch_names
    power = epochs.copy().pick(picks).compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        average=False,
        decim=2,
        n_jobs=1,
        verbose=False
    )
    # shape: (n_epochs, n_channels, n_freqs, n_times)
    features = np.mean(power.data, axis=(1, 3))  # (n_epochs, n_freqs)
    return features

def predict_wavelet(X_test_epochs, model_path="wavelet-model.pkl"):
    model = joblib.load(model_path)
    scaler = model['scaler']
    clf = model['clf']
    X_test_features = extract_wavelet_features(X_test_epochs)
    X_test_scaled = scaler.transform(X_test_features)
    y_pred = clf.predict(X_test_scaled)
    return y_pred

def extract_laplacian_csp_features(epochs, phi, n_components=10):
    X = epochs.get_data() * 1e6
    phi_reduced = phi[:, :n_components]
    X_reduced = np.tensordot(X, phi_reduced, axes=([1], [0]))
    X_csp = mne.decoding.CSP(n_components=4).fit_transform(np.transpose(X_reduced, (0, 2, 1)), np.zeros(len(X)))
    return X_csp

def predict_laplacian_csp(X_test_epochs, phi, model_path="laplacian_csp-model.pkl", n_components=10):
    model = joblib.load(model_path)
    scaler = model['scaler']
    clf = model['clf']
    X_test_features = extract_laplacian_csp_features(X_test_epochs, phi, n_components)
    X_test_scaled = scaler.transform(X_test_features)
    y_pred = clf.predict(X_test_scaled)
    return y_pred

def extract_laplacian_fgmdm_features(epochs, phi, n_components=10):
    from pyriemann.estimation import Covariances
    X = epochs.get_data() * 1e6
    phi_reduced = phi[:, :n_components]
    X_reduced = np.tensordot(X, phi_reduced, axes=([1], [0]))
    X_cov = np.transpose(X_reduced, (0, 2, 1))
    covs = Covariances().fit_transform(X_cov)
    reg = 1e-6
    covs += reg * np.eye(X_cov.shape[1])[None, :, :]
    return covs

def predict_laplacian_fgmdm(X_test_epochs, phi, model_path="laplacian_fgmdm-model.pkl", n_components=10):
    model = joblib.load(model_path)
    clf = model['clf']
    X_test_features = extract_laplacian_fgmdm_features(X_test_epochs, phi, n_components)
    y_pred = clf.predict(X_test_features)
    return y_pred

def get_label_name(label):
    return "target" if label == 1 else "nontarget"

wavelet_mis_idx = np.load("wavelet-misclassified-labels.npy")
lap_csp_mis_idx = np.load("laplacian_csp-misclassified-labels.npy")
lap_fgmdm_mis_idx = np.load("laplacian_fgmdm-misclassified-labels.npy")
#
overlap_wavelet_csp = np.intersect1d(wavelet_mis_idx, lap_csp_mis_idx)
overlap_wavelet_fgmdm = np.intersect1d(wavelet_mis_idx, lap_fgmdm_mis_idx)
overlap_csp_fgmdm = np.intersect1d(lap_csp_mis_idx, lap_fgmdm_mis_idx)
overlap_all = np.intersect1d(overlap_wavelet_csp, lap_fgmdm_mis_idx)
print("Overlap wavelet & CSP:", overlap_wavelet_csp)
print("Overlap wavelet & FgMDM:", overlap_wavelet_fgmdm)
print("Overlap CSP & FgMDM:", overlap_csp_fgmdm)
print("Overlap all:", overlap_all)

for i in range(len(epochs)):
    epoch = epochs[i].get_data()  # shape: (1, n_channels, n_times)
    # Preprocess/feature extraction as used during training:
    # For wavelet classifier, extract features as in classify_wavelet_power:
    extract_wavelet_features(epoch)
    if scaler is not None:
        X_single_scaled = scaler.transform(X_single)
    else:
        X_single_scaled = X_single
    pred = clf.predict(X_single_scaled)
    print(f"Epoch {i}: predicted label based on the morlet wavelet: {pred[0]}")


laplacian_fgmdm = joblib.load("laplacian_fgmdm-model.pkl")  # or "laplacian_csp-model.pkl", etc.
clf = laplacian_fgmdm['clf']
scaler = laplacian_fgmdm['scaler']  

 # Assume you have X_test (epochs), y_test (labels), phi (eigenvectors)
y_pred_wavelet = predict_wavelet(X_test)
# Example for Laplacian CSP+LDA classifier:
y_pred_csp = predict_laplacian_csp(X_test, phi)
# Example for Laplacian FgMDM classifier:
y_pred_fgmdm = predict_laplacian_fgmdm(X_test, phi)
print("Wavelet accuracy:", accuracy_score(y_test, y_pred_wavelet))
print("Laplacian CSP+LDA accuracy:", accuracy_score(y_test, y_pred_csp))
print("Laplacian FgMDM accuracy:", accuracy_score(y_test, y_pred_fgmdm))