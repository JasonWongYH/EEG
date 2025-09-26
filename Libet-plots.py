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
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM, TSclassifier
import joblib

motor_channels=['Fz','FCz','Pz','C3','C4','T7','T8']


def plot_laplacian_temporal_dynamics(target_epochs, nontarget_epochs, phi, n_components=10):
    """
    Plot temporal evolution of Laplacian eigenvector projections for slow ramping and beta desynchronization.
    """
    sfreq = target_epochs.info['sfreq']
    times = target_epochs.times
    # Project data onto Laplacian eigenvectors
    X_target = target_epochs.get_data() * 1e6  # (n_epochs, n_channels, n_times)
    X_nontarget = nontarget_epochs.get_data() * 1e6
    phi_reduced = phi[:, :n_components]
    X_target_proj = np.tensordot(X_target, phi_reduced, axes=([1], [0]))  # (n_epochs, n_times, n_components)
    X_nontarget_proj = np.tensordot(X_nontarget, phi_reduced, axes=([1], [0]))

    # --- Slow ramping: mean power over time for first 5 eigenvectors ---
    power_target_time = X_target_proj ** 2  # (n_epochs, n_times, n_components)
    power_nontarget_time = X_nontarget_proj ** 2
    
    plot_and_classify_signatures(power_target_time, power_nontarget_time, label='Power (Slow Ramping)', depth=2)
    
    mean_power_target_time = power_target_time.mean(axis=0)  # (n_times, n_components)
    mean_power_nontarget_time = power_nontarget_time.mean(axis=0)

    plt.figure(figsize=(12, 5))
    for i in range(min(5, n_components)):
        plt.plot(times, mean_power_target_time[:, i], label=f'target Phi {i+1}')
        plt.plot(times, mean_power_nontarget_time[:, i], '--', label=f'nontarget Phi {i+1}')
    plt.title('Slow Ramping: Mean Power Over Time (First 5 Laplacian Eigenvectors)')
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Beta desynchronization: beta envelope over time for first 5 eigenvectors ---
    beta_power_target = []
    beta_power_nontarget = []
    for i in range(min(5, n_components)):
        # Filter in beta band
        filtered_target = bandpass(X_target_proj[:, :, i], sfreq, 13, 30)
        filtered_nontarget = bandpass(X_nontarget_proj[:, :, i], sfreq, 13, 30)
        # Hilbert envelope
        envelope_target = np.abs(hilbert(filtered_target, axis=1))
        envelope_nontarget = np.abs(hilbert(filtered_nontarget, axis=1))
        # Mean across epochs
        beta_power_target.append(envelope_target.mean(axis=0))
        beta_power_nontarget.append(envelope_nontarget.mean(axis=0))
    beta_power_target = np.array(beta_power_target)  
    beta_power_nontarget = np.array(beta_power_nontarget)

    plt.figure(figsize=(12, 5))
    for i in range(min(5, n_components)):
        plt.plot(times, beta_power_target[i], label=f'Target EV {i+1}')
        plt.plot(times, beta_power_nontarget[i], '--', label=f'Nontarget EV {i+1}')
    plt.title('Beta Desynchronization: Envelope Over Time (First 5 Laplacian Eigenvectors)')
    plt.xlabel('Time (s)')
    plt.ylabel('Beta Power Envelope')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Reshape to (n_epochs, n_times, n_components) for signature computation
    # For simplicity, stack all envelopes for signature computation   
    env_target = np.stack([np.abs(hilbert(bandpass(X_target_proj[:, :, i], sfreq, 13, 30), axis=1)) for i in range(min(5, n_components))], axis=-1)
    env_nontarget = np.stack([np.abs(hilbert(bandpass(X_nontarget_proj[:, :, i], sfreq, 13, 30), axis=1)) for i in range(min(5, n_components))], axis=-1)

    plot_and_classify_signatures(env_target, env_nontarget, label='Beta Envelope', depth=2)

def plot_beta_desync_timecourse(target_epochs, nontarget_epochs, motor_channels, threshold_ratio=0.8, window_type='hanning'):
    """
    Plot the mean beta power time course for target (movement) and nontarget (rest) epochs.
    """
    # Hilbert transform to get power envelope
    hilbert_target = target_epochs.apply_hilbert(picks=motor_channels, envelope=True, verbose=False)
    hilbert_nontarget = nontarget_epochs.apply_hilbert(picks=motor_channels, envelope=True, verbose=False)
    # Average across channels
    mov_data = (hilbert_target.get_data(picks=motor_channels) * 1e6).mean(axis=1)  # (n_epochs, n_times)
    rest_data = (hilbert_nontarget.get_data(picks=motor_channels) * 1e6).mean(axis=1)

    mov_data = apply_epoch_window(mov_data, window_type=window_type)
    rest_data = apply_epoch_window(rest_data, window_type=window_type)
   
    # Mean across epochs
    mov_mean = mov_data.mean(axis=0)
    rest_mean = rest_data.mean(axis=0)
    # Time axis
    times = target_epochs.times

    # Calculate desync_ratio (reuse logic from find_beta_desync_onset)
    avg_rest_power = rest_mean.mean()
    avg_mov_power = mov_mean.mean()
    desync_ratio = avg_mov_power / avg_rest_power if avg_rest_power != 0 else np.nan

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(times, mov_mean, label='Target (Movement)')
    plt.plot(times, rest_mean, label='Nontarget (Rest)')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Beta Power')
    plt.title(f'Beta Desynchronization: Target vs Nontarget\nDesync Ratio (Target/Rest): {desync_ratio:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    movement_onsets = []
    for epoch in mov_data:
        # Find first time where beta power drops below threshold_ratio * rest_mean
        below_thresh = np.where(epoch < threshold_ratio * rest_mean)[0]
        if len(below_thresh) > 0:
            onset_time = times[below_thresh[0]]
            movement_onsets.append(onset_time)
        else:
            movement_onsets.append(np.nan)
    
    if movement_onsets is not None and len(movement_onsets) > 0:
        plt.figure(figsize=(7,4))
        plt.hist(movement_onsets, bins=20, color='skyblue', edgecolor='k')
        plt.xlabel('Inferred Movement Onset (s)')
        plt.ylabel('Count')
        plt.title('Histogram of Inferred Movement Onsets')
        plt.tight_layout()
        plt.show()
        # Print summary statistics
        print(f"[plot_beta_desync_timecourse] Movement Onset: mean={np.mean(movement_onsets):.3f} s, std={np.std(movement_onsets):.3f} s, min={np.min(movement_onsets):.3f} s, max={np.max(movement_onsets):.3f} s")
    
def plot_laplacian_eigenvectors(montage, eigenvectors, eigenvalues, target_epochs, nontarget_epochs, n_components=10):
    """    """
    # Get electrode positions from montage
    ch_pos = montage.get_positions()['ch_pos']
    pos = np.array(list(ch_pos.values()))
    
    # Create dummy info object for plotting
    info = mne.create_info(
        ch_names=list(ch_pos.keys()),
        sfreq=target_epochs.info['sfreq'],
        ch_types='eeg'
    )
    info.set_montage(montage)
    
    # Prepare figure
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Laplace-Beltrami Analysis: Eigenvectors and Class Differences', y=1.02, fontsize=16)
    
    # Create gridspec for layout control
    gs = fig.add_gridspec(3, n_components, height_ratios=[1, 1, 0.5])
    
    # Plot eigenvectors in first row
    for i in range(n_components):
        ax = fig.add_subplot(gs[0, i])
        mne.viz.plot_topomap(
            eigenvectors[:, i],
            info,
            axes=ax,
            show=False,
            contours=False,
            sensors=True
        )
        ax.set_title(f'ϕ{i+1}\nλ={eigenvalues[i]:.2f}', pad=10)

    # Project data onto eigenspace and compute mean power for each class
    phi_reduced = eigenvectors[:, :n_components]

    # Get data (n_epochs, n_channels, n_times)
    X_target = target_epochs.get_data() * 1e6
    X_nontarget = nontarget_epochs.get_data() * 1e6

    # Project onto eigenspace (n_epochs, n_times, n_components)
    X_target_proj = np.tensordot(X_target, phi_reduced, axes=([1], [0]))
    X_nontarget_proj = np.tensordot(X_nontarget, phi_reduced, axes=([1], [0]))

    # Compute mean power (across time and epochs) for each component
    power_target = np.mean(X_target_proj**2, axis=(0, 1))
    power_nontarget = np.mean(X_nontarget_proj**2, axis=(0, 1))

    print(f"[plot_laplacian_eigenvectors] power_target:{power_target}")
    print(f"[plot_laplacian_eigenvectors] power_nontarget:{power_nontarget}")

    # Plot spectral power comparison in second row (move from third row)
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(1, n_components+1)
    width = 0.35
    ax.bar(x - width/2, power_target, width, label='Target')
    ax.bar(x + width/2, power_nontarget, width, label='Non-target')
    ax.set_xticks(x)
    ax.set_xticklabels([f'ϕ{i}' for i in x])
    ax.set_xlabel('Eigenvector Component')
    ax.set_ylabel('Mean Power')
    ax.set_title('Spectral Power Comparison')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_and_classify_signatures(target_proj, nontarget_proj, label, depth=2):
    print(f"[plot_and_classify_signatures] Computing path signatures (depth={depth}) for {label}...")
    X_target_sigs = compute_signatures(target_proj, depth=depth)
    X_nontarget_sigs = compute_signatures(nontarget_proj, depth=depth)
    X = np.vstack([X_target_sigs, X_nontarget_sigs])
    y = np.hstack([np.ones(len(X_target_sigs)), np.zeros(len(X_nontarget_sigs))])

    # Plot mean signature
    plt.figure(figsize=(10,5))
    plt.plot(X_target_sigs.mean(axis=0), label='Target mean signature')
    plt.plot(X_nontarget_sigs.mean(axis=0), label='Nontarget mean signature')
    plt.legend()
    plt.title(f'Mean Path Signature Coefficients ({label})')
    plt.xlabel('Signature term')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()

    # --- Feature scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Classification ---
    clf = LogisticRegression(max_iter=1000) #Jason: do I want to use a sparse logistic regression ?
    acc = np.mean(cross_val_score(clf, X_scaled, y, cv=5))
    print(f'[plot_and_classify_signatures] Path signature classification accuracy ({label}): {acc:.3f}')

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(6,5))
    plt.scatter(X_pca[y==1,0], X_pca[y==1,1], label='Target', alpha=0.7)
    plt.scatter(X_pca[y==0,0], X_pca[y==0,1], label='Nontarget', alpha=0.7)
    plt.title(f'PCA of Path Signature Features ({label})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    try:
        X_tsne = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(X_scaled)
        plt.figure(figsize=(6,5))
        plt.scatter(X_tsne[y==1,0], X_tsne[y==1,1], label='Target', alpha=0.7)
        plt.scatter(X_tsne[y==0,0], X_tsne[y==0,1], label='Nontarget', alpha=0.7)
        plt.title(f't-SNE of Path Signature Features ({label})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[plot_and_classify_signatures] t-SNE failed: {e}")

def plot_epoch_comparison(target_epochs, nontarget_epochs, montage, phi, 
                          n_components=10, plot_type='topomap'):
    """
    Plot comparison between misclassified and correctly classified epochs.
    plot_type: 'topomap' for Laplacian eigenvectors or 'beta' for ERD/ERS
    """
    # Prepare data and labels
    all_epochs = mne.concatenate_epochs([target_epochs, nontarget_epochs])
    n_target = len(target_epochs)
    n_nontarget = len(nontarget_epochs)
    labels = np.concatenate([np.ones(n_target, dtype=int), 
                            np.zeros(n_nontarget, dtype=int)])
    
    # Project data onto Laplacian eigenvectors for classification
    X = all_epochs.get_data() * 1e6  # (n_epochs, n_channels, n_times)
    phi_reduced = phi[:, :n_components]
    X_proj = np.tensordot(X, phi_reduced, axes=([1], [0]))  # (n_epochs, n_times, n_components)
    
    # Flatten time and component dimensions for classification
    X_flat = X_proj.reshape(len(X_proj), -1)
    
    # Get misclassified indices
    mis_idx = get_misclassified_epochs(X_flat, labels)
    print(f"[plot_epoch_comparison] Found {len(mis_idx)} misclassified epochs")
    
    # Sample correct epochs (same number as misclassified)
    correct_idx = np.setdiff1d(np.arange(len(labels)), mis_idx)
    sample_size = min(len(mis_idx), len(correct_idx))
    correct_sample = np.random.choice(correct_idx, size=sample_size, replace=False)
    
    if plot_type == 'topomap':
        _plot_topomap_comparison(all_epochs, mis_idx, correct_sample, montage, phi, n_components)
    elif plot_type == 'beta':
        _plot_beta_comparison(all_epochs, mis_idx, correct_sample, labels, motor_channels)
    else:
        raise ValueError("plot_type must be 'topomap' or 'beta'")

def _plot_topomap_comparison(epochs, mis_idx, correct_idx, montage, phi, n_components):
    """Plot topomaps of Laplacian eigenvectors for misclassified vs correct epochs"""
    # Get electrode positions
    ch_pos = montage.get_positions()['ch_pos']
    info = mne.create_info(
        ch_names=list(ch_pos.keys()),
        sfreq=epochs.info['sfreq'],
        ch_types='eeg'
    )
    info.set_montage(montage)
    
    # Select epochs and project to Laplacian space
    mis_data = epochs[mis_idx].get_data() * 1e6
    correct_data = epochs[correct_idx].get_data() * 1e6
    
    mis_proj = np.tensordot(mis_data, phi[:, :n_components], axes=([1], [0]))
    correct_proj = np.tensordot(correct_data, phi[:, :n_components], axes=([1], [0]))
    
    # Average over time and epochs
    mis_mean = mis_proj.mean(axis=(0, 1))  # (n_components,)
    correct_mean = correct_proj.mean(axis=(0, 1))
    
    # Plot topomaps
    n_cols = min(9, n_components)
    fig, axes = plt.subplots(2, n_cols, figsize=(20, 6))
    fig.suptitle('Laplacian Eigenvector Topomaps: Misclassified vs Correct Epochs', y=1.05)
    
    for i in range(n_cols):
        # Misclassified
        mne.viz.plot_topomap(
            phi[:, i] * mis_mean[i], info,
            axes=axes[0, i], show=False, contours=False
        )
        axes[0, i].set_title(f'ϕ{i+1} (Misclassified)')
        
        # Correct
        mne.viz.plot_topomap(
            phi[:, i] * correct_mean[i], info,
            axes=axes[1, i], show=False, contours=False
        )
        axes[1, i].set_title(f'ϕ{i+1} (Correct)')
    
    plt.tight_layout()
    plt.show()

def _plot_beta_comparison(epochs, mis_idx, correct_idx, labels, picks):
    """Plot beta power time courses for individual misclassified vs correct epochs"""
    # Get data for selected epochs
    mis_data = epochs[mis_idx].get_data(picks=picks) * 1e6  # (n_epochs, n_channels, n_times)
    correct_data = epochs[correct_idx].get_data(picks=picks) * 1e6
    
    # Compute beta power (13-30Hz) using Hilbert transform
    def compute_beta_power(data, sfreq):
        filtered = np.array([bandpass(ch, sfreq, 13, 30) for ch in data])
        return np.abs(hilbert(filtered, axis=-1)).mean(axis=0)  # Avg across channels
    
    sfreq = epochs.info['sfreq']
    times = epochs.times
    
    # Plot individual epochs
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Beta Power in Individual Epochs')
    
    # Misclassified
    for i, idx in enumerate(mis_idx):
        beta_power = compute_beta_power(mis_data[i], sfreq)
        axes[0].plot(times, beta_power, 
                    color='red' if labels[idx] == 1 else 'blue',
                    alpha=0.5)
    axes[0].set_title(f'Misclassified Epochs (N={len(mis_idx)})')
    axes[0].set_ylabel('Beta Power')
    axes[0].axvline(0, color='k', linestyle='--')  # Event marker
    
    # Correct
    for i, idx in enumerate(correct_idx):
        beta_power = compute_beta_power(correct_data[i], sfreq)
        axes[1].plot(times, beta_power,
                    color='red' if labels[idx] == 1 else 'blue',
                    alpha=0.5)
    axes[1].set_title(f'Correctly Classified Epochs (N={len(correct_idx)})')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Beta Power')
    axes[1].axvline(0, color='k', linestyle='--')
    
    plt.tight_layout()
    plt.show()

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
    return data * window  # broadcasting over epochs

def plot_epoch_comparison_by_indices(epochs, mis_idx, correct_idx, montage, phi, n_components=10, plot_type='topomap'):
    if plot_type == 'topomap':
        _plot_topomap_comparison(epochs, mis_idx, correct_idx, montage, phi, n_components)
    elif plot_type == 'beta':
        _plot_beta_comparison(epochs, mis_idx, correct_idx, None, motor_channels)
    else:
        raise ValueError("plot_type must be 'topomap' or 'beta'")


# plot_laplacian_eigenvectors(montage, phi, eigenvalues, target_epochs, nontarget_epochs, 10)
# plot_beta_desync_timecourse(target_epochs, nontarget_epochs, motor_channels)
# plot_laplacian_temporal_dynamics(target_epochs, nontarget_epochs, phi, 10)
# plot_epoch_comparison(target_epochs, nontarget_epochs, montage, phi, plot_type='topomap')
# plot_epoch_comparison(target_epochs, nontarget_epochs, montage, phi, plot_type='beta')

