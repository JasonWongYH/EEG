"""
EEG Ensemble Analysis Script
This script demonstrates how to:
1. Load misclassified epochs for each classifier
2. Identify overlapping misclassified epochs
3. Construct an ensemble guided by uncertainty
4. Plot intermediate classifier probabilities
"""
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import matplotlib.pyplot as plt

# 1. Load misclassified indices for each classifier
wavelet_mis_idx = np.load("wavelet-misclassified-labels.npy")
lap_csp_mis_idx = np.load("laplacian_csp-misclassified-labels.npy")
lap_fgmdm_mis_idx = np.load("laplacian_fgmdm-misclassified-labels.npy")

# 2. Identify overlapping misclassified epochs
overlap_wavelet_csp = np.intersect1d(wavelet_mis_idx, lap_csp_mis_idx)
overlap_wavelet_fgmdm = np.intersect1d(wavelet_mis_idx, lap_fgmdm_mis_idx)
overlap_csp_fgmdm = np.intersect1d(lap_csp_mis_idx, lap_fgmdm_mis_idx)
overlap_all = np.intersect1d(overlap_wavelet_csp, lap_fgmdm_mis_idx)
print("Overlap wavelet & CSP:", overlap_wavelet_csp)
print("Overlap wavelet & FgMDM:", overlap_wavelet_fgmdm)
print("Overlap CSP & FgMDM:", overlap_csp_fgmdm)
print("Overlap all:", overlap_all)

# 3. Load classifier models
wavelet_model = joblib.load("wavelet-model.pkl")['clf']
lap_csp_model = joblib.load("laplacian_csp-model.pkl")['clf']
lap_fgmdm_model = joblib.load("laplacian_fgmdm-model.pkl")['clf']

# 4. Reconstruct features for each classifier
# These must be computed as in your main classifier functions:
#   - X_wavelet: see classify_wavelet_power (mean wavelet power per epoch)
#   - X_csp: see classify_laplacian_csp_lda (CSP features per epoch)
#   - X_fgmdm: see classify_laplacian_fgmdm (covariance features per epoch)
#   - y_true: true labels for all epochs
# Example (replace ... with actual code):
# X_wavelet = ... # np.mean(power.data, axis=(1, 3)) for all epochs
# X_csp = ...     # CSP features for all epochs
# X_fgmdm = ...   # Covariance features for all epochs
# y_true = ...    # True labels

# 5. Get classifier probabilities (predict_proba)
# .predict_proba is a method of scikit-learn classifiers (e.g., LogisticRegression, LDA)
# It returns the probability estimates for each class for each sample.
# For MDM (pyriemann), use .predict_proba if available, else use .predict and treat as hard decision.
# Example:
# probs_wavelet = wavelet_model.predict_proba(X_wavelet)
# probs_csp = lap_csp_model.predict_proba(X_csp)
# probs_fgmdm = lap_fgmdm_model.predict_proba(X_fgmdm)

# 6. Ensemble prediction
# ensemble_probs = (probs_wavelet + probs_csp + probs_fgmdm) / 3
# ensemble_pred = np.argmax(ensemble_probs, axis=1)
# ensemble_acc = accuracy_score(y_true, ensemble_pred)
# print("Ensemble accuracy:", ensemble_acc)

# 7. Uncertainty measure
# uncertainty = entropy(ensemble_probs.T)

# 8. Plot intermediate probabilities for a given epoch
# epoch_idx = 0
# plt.plot(ensemble_probs[epoch_idx])
# plt.title(f"Ensemble probabilities for epoch {epoch_idx}")
# plt.xlabel("Class")
# plt.ylabel("Probability")
# plt.show()

"""
How/where are features X_wavelet, X_csp, X_fgmdm obtained?
----------------------------------------------------------
- X_wavelet: In classify_wavelet_power, features are computed as np.mean(power.data, axis=(1, 3)), where power is the time-frequency representation for each epoch. This gives a feature vector per epoch.
- X_csp: In classify_laplacian_csp_lda, features are obtained by projecting the data onto Laplacian eigenvectors, then applying CSP (CSP.fit_transform), and optionally scaling. This gives a feature vector per epoch.
- X_fgmdm: In classify_laplacian_fgmdm, features are covariance matrices computed from the Laplacian-projected data (Covariances().fit_transform). This gives a matrix per epoch.
- y_true: The true labels are constructed as np.concatenate([np.ones(len(target_epochs)), np.zeros(len(nontarget_epochs))]) in the main script.

Where is .predict_proba defined?
-------------------------------
- .predict_proba is a method of scikit-learn classifiers such as LogisticRegression and LDA. It returns probability estimates for each class for each sample. For MDM (pyriemann), .predict_proba is available for binary classification; otherwise, use .predict for hard decisions.

To reconstruct features for ensemble analysis, rerun the feature extraction code from each classifier function on the same epochs used for testing.
"""
