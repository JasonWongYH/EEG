import matplotlib.pyplot as plt
import numpy as np
import torch
import signatory
import time
import mne
from mne.decoding import CSP
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import entropy
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM, TSclassifier
import joblib

motor_channels=['Fz','FCz','Pz','C3','C4','T7','T8']

def load_electrode_coordinates(csv_file='na-265.csv'):
    """
    Load electrode locations from CSV and create a montage.    
    Parameters:
    - csv_file: Path to CSV file with electrode coordinates
  
    """
    ''' Decoding Acute Pain with Combined EEG and Physiological Data 
    FP1, FP2, FC6, FC2, FC1, FC5, Fz, C4, Cz,C3, P4, Pz, P3, O2, Oz, O1 with reference electrodes at the earlobes
    '''
    df = pd.read_csv(csv_file,sep='\t', skipinitialspace=True)#StringIO(data_string), sep='\t', skipinitialspace=True)
    df = df.iloc[:, :-1] # Remove the last empty column
    df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']] / 1000.0 # Convert to m from mm
    
    ch_names = df['labels'].tolist()
    ch_pos = df[['X', 'Y', 'Z']].values
    
    # Compute scaling factor (target radius = 0.095 m)
    radii = np.linalg.norm(ch_pos, axis=1)
    scale_factor = 0.095 / np.mean(radii)  # Scale to MNE's default head radius
    
    # Apply scaling to all channels
    scaled_ch_pos = {
        ch: ch_pos[i] * scale_factor for i, ch in enumerate(ch_names)
    }
    
    montage = mne.channels.make_dig_montage(
        ch_pos=scaled_ch_pos,
        coord_frame="head"  # Maintain head coordinate frame
    )
    info = mne.create_info(
        ch_names=df['labels'].tolist(),
        sfreq=4096,  # Set your actual sampling frequency (Hz)
        ch_types='eeg'  # All channels are EEG (adjust if needed)
    )
    
    info.set_montage(montage) 
    # info.plot_sensors(show_names=True);
    
    # Compute average head radius (should be ~0.095 m if normalized)
    radii = np.linalg.norm(df[['X', 'Y', 'Z']].values, axis=1)
    print(f"[load_electrode_coordinates] Mean radius: {np.mean(radii):.10f} meters")
    
    correct_labels = [
        'Fp1', 'Fp2', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8',  # Channels 1-8
        'F10', 'FC5', 'FC1', 'FC2', 'FC6', 'T9', 'T7', 'C3',  # Channels 9-16
        'C4', 'T8', 'T10', 'CP5', 'CP1', 'CP2', 'CP6', 'P9',  # Channels 17-24
        'P7', 'P3', 'Pz', 'P4', 'P8', 'P10', 'O1', 'O2',     # Channels 25-32
        'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',   # Channels 33-40
        'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3',   # Channels 41-48
        'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4',    # Channels 49-56
        'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'POz' # Channels 57-64
    ]

    return montage, correct_labels, ch_names, ch_pos


def compute_area_matrix(pos):
    """
    Compute the area matrix S where S[i,i] = Voronoi area around vertex i.    
    Args:
        pos: (n_vertices, 3) array of electrode coordinates    
    Returns:
        S: (n_vertices, n_vertices) diagonal sparse matrix
    """
    # Step 1: Create mesh via Delaunay triangulation
    tri = Delaunay(pos[:, :2])  # Use 2D projection for EEG cap
    
    # Step 2: Initialize area array
    n = len(pos)
    S = np.zeros(n)
    
    # Step 3: Compute Voronoi areas for each vertex
    for simplex in tri.simplices:
        i, j, k = simplex
        vi, vj, vk = pos[i], pos[j], pos[k]
        
        # Edge vectors
        e_ij = vj - vi
        e_ik = vk - vi
        e_jk = vk - vj
        
        # Triangle area
        tri_area = 0.5 * np.linalg.norm(np.cross(e_ij, e_ik))
        
        # Compute cotangents of all angles
        def cot(a, b):
            return np.dot(a, b) / np.linalg.norm(np.cross(a, b))
        
        cot_i = cot(e_ij, e_ik)
        cot_j = cot(-e_ij, e_jk)
        cot_k = cot(-e_ik, -e_jk)
        
        # Voronoi area contributions (Meyer et al. 2003)
        if cot_j > 0 and cot_k > 0:  # Non-obtuse at j and k
            S[i] += 0.125 * (cot_j * np.sum(e_ik**2) + cot_k * np.sum(e_ij**2))
        else:  # Fallback to barycentric area
            S[i] += tri_area / 3
            
        if cot_i > 0 and cot_k > 0:  # Non-obtuse at i and k
            S[j] += 0.125 * (cot_i * np.sum(e_jk**2) + cot_k * np.sum(e_ij**2))
        else:
            S[j] += tri_area / 3
            
        if cot_i > 0 and cot_j > 0:  # Non-obtuse at i and j
            S[k] += 0.125 * (cot_i * np.sum(e_jk**2) + cot_j * np.sum(e_ik**2))
        else:
            S[k] += tri_area / 3
    
    # Step 4: Convert to diagonal sparse matrix
    return diags(S, 0)

def compute_laplace_beltrami(pos, n_components=10):
    """Algorithm 1: Compute Laplacian eigenvectors/values."""
    # Step 1: Delaunay triangulation (2D projection)
    tri = Delaunay(pos[:, :2])
    edges = set()
    for simplex in tri.simplices:
        edges.update([(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])])
    
    # Step 2: Cotangent weights matrix (M) and area matrix (S)
    n = len(pos)
    from scipy.sparse import lil_matrix
    M = lil_matrix((n, n))
    # Use compute_area_matrix for robust area calculation
    S = compute_area_matrix(pos)

    for i, j in edges:
        # Find adjacent triangles (i-j-k and i-j-l)
        triangles = [s for s in tri.simplices if i in s and j in s]
        if len(triangles) != 2:
            continue  # Skip boundary edges

        # Angles opposite edge i-j in the two triangles
        def cotangent(a, b, c):
            ba = a - b
            ca = a - c
            return np.dot(ba, ca) / np.linalg.norm(np.cross(ba, ca))

        # Triangle 1 (i-j-k)
        k = [v for v in triangles[0] if v != i and v != j][0]
        cot_alpha = cotangent(pos[k], pos[i], pos[j])

        # Triangle 2 (i-j-l)
        l = [v for v in triangles[1] if v != i and v != j][0]
        cot_beta = cotangent(pos[l], pos[i], pos[j])

        # Update M (symmetric)
        weight = 0.5 * (cot_alpha + cot_beta)
        M[i, j] = -weight
        M[j, i] = -weight
        M[i, i] += weight
        M[j, j] += weight

    # S is already a diagonal sparse matrix from compute_area_matrix
    M = M.tocsc()

    # Step 3: Generalized eigenvalue problem M v = λ S v
    eigenvalues, eigenvectors = eigsh(M, k=n_components, M=S, sigma=0, which='LM')
    return eigenvectors, eigenvalues

def compute_cotangent_matrix(pos, tri):
    """
    Compute the cotangent weight matrix M.
    
    Args:
        pos: (n_vertices, 3) array of electrode coordinates
        tri: Delaunay triangulation object
    
    Returns:
        M: (n_vertices, n_vertices) sparse matrix
    """
    from scipy.sparse import lil_matrix
    n = len(pos)
    M = lil_matrix((n, n))
    
    # Precompute edges and triangles
    edges = {}
    for simplex in tri.simplices:
        i, j, k = simplex
        # Store triangles sharing each edge
        edges.setdefault(frozenset({i, j}), []).append(k)
        edges.setdefault(frozenset({j, k}), []).append(i)
        edges.setdefault(frozenset({k, i}), []).append(j)
    
    # Process each edge and its two adjacent triangles
    for edge, opposite_vertices in edges.items():
        i, j = tuple(edge)
        if len(opposite_vertices) != 2:
            continue  # Skip boundary edges (only one adjacent triangle)
        
        k, l = opposite_vertices
        # Vectors
        e_ik = pos[k] - pos[i]
        e_jk = pos[k] - pos[j]
        e_il = pos[l] - pos[i]
        e_jl = pos[l] - pos[j]
        
        # Cotangents of opposite angles
        cot_alpha = np.dot(e_ik, e_jk) / np.linalg.norm(np.cross(e_ik, e_jk))
        cot_beta  = np.dot(e_il, e_jl) / np.linalg.norm(np.cross(e_il, e_jl))
        weight = 0.5 * (cot_alpha + cot_beta)
        
        # Update M
        M[i, j] = -weight
        M[j, i] = -weight
        M[i, i] += weight
        M[j, j] += weight
    
    return M.tocsc()

def preprocess(base_gdf, other_gdf, ced_file, target_event, nontarget_event, n_components=10):
    montage, correct_labels, ch_names, pos = load_electrode_coordinates(ced_file)
    
    #time start
    start_preproc = time.time()
    phi, eigenvalues = compute_laplace_beltrami(pos, n_components)
    end_preproc = time.time()
    print(f"compute_laplace_beltrami: {end_preproc - start_preproc:.2f} seconds")
    #time end
    
    raw = load_preprocess_gdf(base_gdf, correct_labels, montage)
    target_epochs = extract_epochs(raw,target_event)
    nontarget_epochs = extract_epochs(raw,nontarget_event)
    raw_other =  load_preprocess_gdf(other_gdf, correct_labels, montage) 
    target_epochs = mne.concatenate_epochs([target_epochs,extract_epochs(raw_other,{'33127':4})])     
    n_target = len(target_epochs)
    n_nontarget = len(nontarget_epochs)
    labels = np.concatenate([np.ones(n_target, dtype=int), np.zeros(n_nontarget, dtype=int)])
    return target_epochs, nontarget_epochs, montage, phi, eigenvalues, labels

def classify_eeg(target_epochs, nontarget_epochs, montage, phi, eigenvalues, n_components=10):
    """Algorithm 2: Classify EEG using Laplacian-based features. Laplacian+FgMDM gave highest accuracy with ~10 spectral components"""
    
    n_target = len(target_epochs)
    n_nontarget = len(nontarget_epochs)
    labels = np.concatenate([np.ones(n_target, dtype=int), np.zeros(n_nontarget, dtype=int)])

    all_epochs = mne.concatenate_epochs([target_epochs, nontarget_epochs])
    # Project channel dimension to Laplacian eigenvector space
    X = all_epochs.get_data() * 1e6  # (n_epochs, n_channels, n_times)
    phi_reduced = phi[:, :n_components]  # (n_channels, n_components)
    X_reduced = np.tensordot(X, phi_reduced, axes=([1], [0]))  # (n_epochs, n_times, n_components)
    # Classification (CSP+LDA or FgMDM)
    # Option 1: CSP + LDA
    csp = CSP(n_components=4)
    # CSP expects (n_epochs, n_channels, n_times), so transpose to (n_epochs, n_components, n_times)
    X_csp = csp.fit_transform(np.transpose(X_reduced, (0, 2, 1)), labels)
    scores_csp = cross_val_score(LDA(), X_csp, labels, cv=5)

    # Option 2: FgMDM (Riemannian geometry)
    # Swap axes so n_components is the feature/channel dimension
    X_cov = np.transpose(X_reduced, (0, 2, 1))  # (n_epochs, n_components, n_times)
    covs = Covariances().fit_transform(X_cov)
    reg = 1e-6
    covs += reg * np.eye(X_cov.shape[1])[None, :, :]
    scores_mdm = cross_val_score(MDM(), covs, labels, cv=5)

    return np.mean(scores_csp), np.mean(scores_mdm)

def extract_epochs(raw,event_marker):
    # The GDF file should contain annotations that mark the different trial periods
    events, event_id = mne.events_from_annotations(raw)
    # Define the time window (tmin, tmax) for the epochs, relative to the event markers
    tmin, tmax = -2.0, -0.5  # 2 seconds before to 0.5 seconds after the event
    epochs = mne.Epochs(raw, events, event_marker, tmin=tmin, tmax=tmax,
                                 proj=True, baseline=None, preload=True,
                                 event_repeated='drop')
    return epochs

def load_preprocess_gdf(file_path, correct_labels, montage):
    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose='error')
    raw = raw.pick_channels(raw.ch_names[:64])
    raw.filter(l_freq=1., h_freq=40., fir_design='firwin')
    #AJDC removal of eye blinks if it hasn't already been removed 
    raw.resample(256)
    raw.rename_channels(dict(zip(raw.ch_names, correct_labels)))
    raw.set_montage(montage, on_missing='ignore')
    print(f"[load_preprocess_gdf] raw.info.ch_names: {raw.info['ch_names']}")
    print(f"[load_preprocess_gdf] raw.info.sfreq: {raw.info['sfreq']}")
    # Load and inspect just the first 5 samples of the first 5 channels
    small_data, times = raw[:5, :5]  # (channels × time)
    print(f"[load_preprocess_gdf] Data snippet (shape: {small_data.shape}):\n{small_data}")
    print(f"[load_preprocess_gdf] Times (sec): {times}")
    return raw


def bandpass(data, sfreq, low, high, order=2):
    b, a = butter(order, [low/(sfreq/2), high/(sfreq/2)], btype='band')
    return filtfilt(b, a, data, axis=-1)


#untested 
def compute_signatures(power_time, depth=2):
    """
    Compute path signatures for each epoch and component.
    power_time: shape (n_epochs, n_times, n_components)
    Returns: shape (n_epochs, n_components * signature_dim)
    """
    import torch
    import signatory
    n_epochs, n_times, n_components = power_time.shape
    sigs = []
    for epoch in range(n_epochs):
        sig_epoch = []
        for comp in range(n_components): #Jason: 
            path = torch.from_numpy(power_time[epoch, :, comp][:, None].astype(np.float32))  # shape (n_times, 1)
            sig = signatory.signature(path.unsqueeze(0), depth).squeeze(0).numpy()
            sig_epoch.extend(sig)
        sigs.append(sig_epoch)
    return np.array(sigs)


#time-frequency of voltage oscillations ~ spatial (scalp manifold) voltage oscillation distribution
def analyze_eigenvector_tfr(epochs, phi, n_components=10, freqs=np.arange(12, 31, 2)):
    """Compute time-frequency power for each eigenvector projection."""
    # Project data onto eigenvectors (n_epochs, n_times, n_components)
    X_proj = np.tensordot(epochs.get_data(), phi[:, :n_components], axes=([1], [0]))
    
    # Create dummy epochs object for TFR (MNE expects epochs in (n_epochs, n_channels, n_times))
    epochs_proj = mne.EpochsArray(
        np.transpose(X_proj, (0, 2, 1)),  # Reshape to (n_epochs, n_components, n_times)
        info=mne.create_info([f"ϕ{i}" for i in range(n_components)], epochs.info['sfreq'], ch_types='eeg')
    )
    
    # Compute TFR for beta band using .compute_tfr
    power = epochs_proj.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=5,
        use_fft=True,
        average=False,
        decim=1,
        n_jobs=1,
        verbose=False
    )[0]  # [0] is power
    return power

def path_signature_classifier(data, depth):
    '''
    return the path signature truncated at level 'depth' as feature vector
    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_samples)
        EEG data.
    depth : int
        truncation depth of the path signature.
    Returns
    -------
    feature : ndarray, shape (n_epochs, \sum_{d \le depth} C_{n_channels}^d)
            truncated path signature as feature vector.
    '''
    data_sig = np.swapaxes(data, 1, 2)
    path = torch.from_numpy(data_sig)
    signature = signatory.signature(path, depth)
    feature = signature.numpy()
    return feature

def path_signature_SPD(data, depth, epsilon):
    '''
    return signature-based SPD matrices as features
    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_samples)
        EEG data.
    depth : int
        truncation depth of the path signature.
    epsilon : float
        regularization parameter.

    Returns
    -------
    feature : ndarray, shape (n_epochs, n_channels, n_channels).
        signature-based SPD matrices as features.

    '''
    n_chan = data.shape[1]
    data_sig = np.swapaxes(data, 1, 2)
    path = torch.from_numpy(data_sig)
    signature = signatory.signature(path, depth)
    feature = signature.numpy()    
    # lead matrix Jason: debug this..
    sig2 = feature[:, n_chan:].reshape(-1, n_chan, n_chan)
    L = sig2 - np.swapaxes(sig2, 1, 2)
    A = - np.matmul(L, L)
    feature = A + epsilon * np.identity(n_chan)
    return feature

#not called 
def classify_signatures(data,labels):
    ###1. data.get_data() * 1e6
    depth = 2
    feature = path_signature_classifier(data, depth)
    sca = StandardScaler()
    lr = LogisticRegression()
    ppl = Pipeline([('scaler', sca), ('clf', lr)]) #Jason: 
    n_fold = 2 #improves things when data is sparse
    cv = KFold(n_fold, shuffle=False)
    cv_results = cross_validate(ppl,feature,labels,cv=cv,n_jobs=-1,return_indices=True)
    # scores_1 = cross_val_score(ppl,feature,labels,cv=cv,n_jobs=-1)
    print(cv_results["indices"]["train"])
    print(cv_results["indices"]["test"])
    ###2.
    epsilon = 0.001  # regularization parameter
    feature = path_signature_SPD(data, depth, epsilon)
    # print("labels=",labels)
    tsc = TSclassifier()  # Riemannian tangent space classifier: how do I visualize this..
    #Jason: I want to splice in a visualization here..
    scores = cross_val_score(tsc, X=feature, y=labels, cv=cv)
    np.mean(scores)

#not used
def apply_epoch_window(data, window_type='hanning'):
    """
    Apply a window function (Hanning or Hamming) - to reduce spectral leakage - to each epoch before averaging.
    *. hanning: when we want better frequency resolution and lower side lobes e.g for power estimation, ERDS..
    *. hamming: when we want to preserve amplitude of main lobe. Side lobes are slightly higher than hanning but main lobe less attenuated. Use when amplitude more useful than frequency resolution.
    ----------
    data : np.ndarray
        Shape (n_epochs, n_times) n_times is the # of data points (scalar samples) so columns of length n_times 
    window_type : str
        'hanning' or 'hamming'
    Returns
    -------
    windowed_data : np.ndarray
        Data after windowing, same shape as input.
    """
    n_times = data.shape[1]
    if window_type == 'hanning':
        window = np.hanning(n_times)
    elif window_type == 'hamming':
        window = np.hamming(n_times)
    else:
        raise ValueError("window_type must be 'hanning' or 'hamming'")
    return data * window  



def classify_wavelet_power(target_epochs, nontarget_epochs, picks=None, freqs=np.arange(12, 31, 2), n_cycles=5, prefix="wavelet"):
    """
    Applies a Morlet wavelet transform to epochs to get a time-frequency representation that's used by the classifiers
    """
    if picks is None:
        picks = target_epochs.ch_names

    # Compute wavelet power for target epochs using .compute_tfr
    power_t = target_epochs.copy().pick(picks).compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        average=False,
        decim=2,
        n_jobs=1,
        verbose=False
    )
    # power_t is a single AverageTFR object with shape (n_epochs, n_channels, n_freqs, n_times)

    # Compute wavelet power for nontarget epochs using .compute_tfr
    power_nt = nontarget_epochs.copy().pick(picks).compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        average=False,
        decim=2,
        n_jobs=1,
        verbose=False
    )
    # power_nt is a single AverageTFR object with shape (n_epochs, n_channels, n_freqs, n_times)

    def extract_features(power):
        return np.mean(power.data, axis=(1, 3))  # shape: (n_epochs, n_freqs)

    X_t = extract_features(power_t)
    X_nt = extract_features(power_nt)
    X = np.vstack([X_t, X_nt])
    y = np.hstack([np.ones(len(X_t)), np.zeros(len(X_nt))])
    # import pdb; pdb.set_trace()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)    
    skf = StratifiedKFold(n_splits=5, shuffle=False)
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
    mis_fif, correct_fif, mis_labels_file, correct_labels_file = save_epochs_and_labels_by_indices(
        all_epochs, y, mis_idx, correct_idx, prefix
    )
    save_classifier(clf, scaler, prefix)
    print(f"[classify_wavelet_power] Mean accuracy: {np.mean(accs):.3f}")
    return np.mean(accs), mis_idx, correct_idx, mis_fif, correct_fif

def classify_laplacian_csp_lda(target_epochs, nontarget_epochs, phi, n_components=10, prefix="laplacian_csp"):
    all_epochs = mne.concatenate_epochs([target_epochs, nontarget_epochs])
    labels = np.concatenate([np.ones(len(target_epochs)), np.zeros(len(nontarget_epochs))])
    X = all_epochs.get_data() * 1e6
    phi_reduced = phi[:, :n_components]
    X_reduced = np.tensordot(X, phi_reduced, axes=([1], [0]))
    X_csp = CSP(n_components=4).fit_transform(np.transpose(X_reduced, (0, 2, 1)), labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_csp)
    clf = LDA()
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    mis_idx, correct_idx = [], []
    accs = []
    for train_idx, test_idx in skf.split(X_scaled, labels):
        clf.fit(X_scaled[train_idx], labels[train_idx])
        y_pred = clf.predict(X_scaled[test_idx])
        accs.append(np.mean(y_pred == labels[test_idx]))
        mis_idx.extend(test_idx[y_pred != labels[test_idx]])
        correct_idx.extend(test_idx[y_pred == labels[test_idx]])
    mis_idx = np.unique(mis_idx)
    correct_idx = np.unique(correct_idx)
    mis_fif, correct_fif, mis_labels_file, correct_labels_file = save_epochs_and_labels_by_indices(
        all_epochs, labels, mis_idx, correct_idx, prefix
    )
    save_classifier(clf, scaler, prefix)
    print(f"[classify_laplacian_csp_lda] Mean accuracy: {np.mean(accs):.3f}")
    return np.mean(accs), mis_idx, correct_idx, mis_fif, correct_fif

def classify_laplacian_fgmdm(target_epochs, nontarget_epochs, phi, n_components=10, prefix="laplacian_fgmdm"):
    all_epochs = mne.concatenate_epochs([target_epochs, nontarget_epochs])
    labels = np.concatenate([np.ones(len(target_epochs)), np.zeros(len(nontarget_epochs))])
    X = all_epochs.get_data() * 1e6
    phi_reduced = phi[:, :n_components]
    X_reduced = np.tensordot(X, phi_reduced, axes=([1], [0]))
    X_cov = np.transpose(X_reduced, (0, 2, 1))
    covs = Covariances().fit_transform(X_cov)
    reg = 1e-6
    covs += reg * np.eye(X_cov.shape[1])[None, :, :]
    clf = MDM()
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    mis_idx, correct_idx = [], []
    accs = []
    for train_idx, test_idx in skf.split(covs, labels):
        clf.fit(covs[train_idx], labels[train_idx])
        y_pred = clf.predict(covs[test_idx])
        accs.append(np.mean(y_pred == labels[test_idx]))
        mis_idx.extend(test_idx[y_pred != labels[test_idx]])
        correct_idx.extend(test_idx[y_pred == labels[test_idx]])
    mis_idx = np.unique(mis_idx)
    correct_idx = np.unique(correct_idx)
    mis_fif, correct_fif, mis_labels_file, correct_labels_file = save_epochs_and_labels_by_indices(
        all_epochs, labels, mis_idx, correct_idx, prefix
    )
    save_classifier(clf, None, prefix)
    print(f"[classify_laplacian_fgmdm] Mean accuracy: {np.mean(accs):.3f}")
    return np.mean(accs), mis_idx, correct_idx, mis_fif, correct_fif

def save_epochs_by_indices(epochs, mis_idx, correct_idx, prefix="epochs"):
    mis_epochs = epochs[mis_idx]
    correct_epochs = epochs[correct_idx]
    mis_fif = f"{prefix}-misclassified.fif"
    correct_fif = f"{prefix}-correct.fif"
    mis_epochs.save(mis_fif, overwrite=True)
    correct_epochs.save(correct_fif, overwrite=True)
    print(f"[save_epochs_by_indices] Saved {len(mis_idx)} misclassified and {len(correct_idx)} correct epochs.")
    return mis_fif, correct_fif

def save_classifier(clf, scaler, filename_prefix):
    joblib.dump({'clf': clf, 'scaler': scaler}, f"{filename_prefix}-model.pkl")
    print(f"[save_classifier] Saved classifier and scaler to {filename_prefix}-model.pkl")

def save_epochs_and_labels_by_indices(epochs, labels, mis_idx, correct_idx, prefix="epochs"):
    """
    Save misclassified and correctly classified epochs to FIF files,
    and their true labels to .npy files.
    """
    mis_epochs = epochs[mis_idx]
    correct_epochs = epochs[correct_idx]
    mis_labels = labels[mis_idx]
    correct_labels = labels[correct_idx]
    mis_fif = f"{prefix}-misclassified.fif"
    correct_fif = f"{prefix}-correct.fif"
    mis_labels_file = f"{prefix}-misclassified-labels.npy"
    correct_labels_file = f"{prefix}-correct-labels.npy"
    mis_epochs.save(mis_fif, overwrite=True)
    correct_epochs.save(correct_fif, overwrite=True)
    np.save(mis_labels_file, mis_labels)
    np.save(correct_labels_file, correct_labels)
    print(f"[save_epochs_and_labels_by_indices] Saved {len(mis_idx)} misclassified and {len(correct_idx)} correct epochs and labels.")
    return mis_fif, correct_fif, mis_labels_file, correct_labels_file


if __name__ == '__main__':
    ced_file = "na-265.csv"
    base_gdf = "data/Andrei.gdf"
    other_gdf = 'data/YunDa-90KeyPresses/rp-train-[2025.04.25.4.14pm]_90_keyPresses.gdf'
    target_epochs, nontarget_epochs, montage, phi, eigenvalues, labels = preprocess(base_gdf, other_gdf, ced_file, target_event={'33127':4}, nontarget_event={'33124':2}, n_components=10)

    wavelet_acc, wavelet_mis_idx, wavelet_correct_idx, wavelet_mis_fif, wavelet_correct_fif = classify_wavelet_power(target_epochs, nontarget_epochs, picks=motor_channels, prefix="wavelet")
    print(f"Wavelet Power Accuracy: {wavelet_acc:.3f}")
    
    # Laplacian CSP+LDA
    lap_csp_acc, lap_csp_mis_idx, lap_csp_correct_idx, lap_csp_mis_fif, lap_csp_correct_fif = classify_laplacian_csp_lda(target_epochs, nontarget_epochs, phi, n_components=10, prefix="laplacian_csp")
    print(f"Laplacian CSP+LDA Accuracy: {lap_csp_acc:.3f}")

    # Laplacian FgMDM
    lap_fgmdm_acc, lap_fgmdm_mis_idx, lap_fgmdm_correct_idx, lap_fgmdm_mis_fif, lap_fgmdm_correct_fif = classify_laplacian_fgmdm(target_epochs, nontarget_epochs, phi, n_components=10, prefix="laplacian_fgmdm")
    print(f"Laplacian FgMDM Accuracy: {lap_fgmdm_acc:.3f}")
    
    
