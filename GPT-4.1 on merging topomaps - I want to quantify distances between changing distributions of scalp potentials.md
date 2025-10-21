To **merge Laplacian eigenvector topomaps into a global topomap** for each time point (or time window), you can **reconstruct the EEG scalp voltage distribution** at that time by taking a **linear combination of the eigenvectors weighted by their coefficients** (i.e., the projection of the EEG data onto each eigenvector at that time).

### Step-by-step approach

1. **Project EEG data onto Laplacian eigenvectors**:  
   For each epoch, you already compute  
   ```python
   X_proj = np.tensordot(epochs.get_data(), phi[:, :n_components], axes=([1], [0]))
   ```
   where `X_proj` has shape `(n_epochs, n_times, n_components)`.

2. **Reconstruct the global topomap at each time point**:  
   For each time `t`, the global topomap is:
   ```
   scalp_map[t] = sum_k (X_proj[epoch, t, k] * phi[:, k])
   ```
   or, for all channels at once:
   ```python
   scalp_map = np.dot(X_proj[epoch, t, :], phi[:, :n_components].T)  # shape: (n_channels,)
   ```
   Or, for all times in an epoch:
   ```python
   scalp_maps = np.dot(X_proj[epoch], phi[:, :n_components].T)  # shape: (n_times, n_channels)
   ```

3. **Visualize or quantify changes**:  
   - For a given epoch, you can plot `scalp_maps` as a sequence of topomaps over time (e.g., using `mne.viz.plot_topomap`).
   - To **quantify change** over a 200ms window, you could compute:
     - The difference between the first and last topomap in the window.
     - The norm or variance of the difference.
     - Correlation between target and nontarget mean topomaps.

4. **Compare target vs. nontarget**:  
   - Average the reconstructed scalp maps across all target epochs and all nontarget epochs.
   - Compare the mean or the temporal evolution (e.g., using correlation, RMS difference, or other metrics).

---

### Example code snippet

````python
# ...existing code...

def reconstruct_global_topomap(X_proj, phi, epoch_idx, t_idx):
    """
    Reconstruct the global scalp topomap at a given epoch and time index.
    X_proj: (n_epochs, n_times, n_components)
    phi: (n_channels, n_components)
    Returns: (n_channels,) array (scalp voltages)
    """
    return np.dot(X_proj[epoch_idx, t_idx, :], phi[:, :X_proj.shape[2]].T)

def reconstruct_all_topomaps(X_proj, phi, epoch_idx):
    """
    Reconstruct scalp topomaps for all time points in an epoch.
    Returns: (n_times, n_channels)
    """
    return np.dot(X_proj[epoch_idx], phi[:, :X_proj.shape[2]].T)

# Example usage:
# X_proj = np.tensordot(epochs.get_data(), phi[:, :n_components], axes=([1], [0]))
# scalp_maps = reconstruct_all_topomaps(X_proj, phi, epoch_idx=0)  # shape: (n_times, n_channels)
# Now plot scalp_maps[t] using mne.viz.plot_topomap for each t in your 200ms window

# To compare target and nontarget:
# avg_target = np.mean([reconstruct_all_topomaps(X_proj, phi, i) for i in target_indices], axis=0)
# avg_nontarget = np.mean([reconstruct_all_topomaps(X_proj, phi, i) for i in nontarget_indices], axis=0)
# Then compare avg_target and avg_nontarget over the 200ms window

# ...existing code...
````

---

**Summary:**  
- The global topomap at each time is the sum of eigenvector topomaps weighted by their coefficients at that time.
- This allows you to track and compare the evolution of the global scalp voltage distribution for target vs. nontarget conditions over any time window.

Yes, the projection is done by expressing your EEG data (for each time point and epoch) as a **linear combination of the Laplace-Beltrami eigenvectors**.

### How the projection is done

Suppose:
- `X` is your EEG data: shape `(n_epochs, n_channels, n_times)`
- `phi` is your Laplacian eigenvector matrix: shape `(n_channels, n_components)`

The **projection** of your data onto the eigenbasis is computed as:

````python
X_proj = np.tensordot(X, phi, axes=([1], [0]))  # shape: (n_epochs, n_times, n_components)
````

- For each epoch and time, this gives you the coefficients (weights) for each eigenvector.
- These coefficients tell you **how much each spatial mode (eigenvector) contributes to the scalp voltage pattern at that time**.

To **reconstruct** the scalp voltage distribution (the topomap) at any time, you sum the eigenvectors weighted by their coefficients:

````python
scalp_map = np.dot(X_proj[epoch, t, :], phi.T)  # shape: (n_channels,)
````

This gives you the voltage at each channel, reconstructed from the eigenbasis.

**In summary:**  
- **Projection:** Multiply your data by the eigenvector matrix (`phi`) to get coefficients in the eigenbasis.
- **Reconstruction:** Multiply the coefficients by the eigenvectors to get back to channel space (the topomap).

The **coefficients** and **eigenvectors** together allow you to map back to channel space because:

- The **eigenvectors** (`phi`) form a basis for the channel space. Each eigenvector is a spatial pattern across all channels.
- The **coefficients** (from projecting your data onto the eigenvectors) tell you how much of each spatial pattern (eigenvector) is present at a given time and epoch.

**Mathematically:**  
If you have `n_components` eigenvectors (each of length `n_channels`), any scalp voltage pattern (topomap) can be written as a weighted sum:

```
scalp_map = c1 * phi1 + c2 * phi2 + ... + cn * phin
```
or, in vectorized form:
```
scalp_map = np.dot(coefficients, phi.T)
```
where:
- `coefficients` is a vector of length `n_components` (the weights for each eigenvector at a given time/epoch)
- `phi` is a matrix of shape `(n_channels, n_components)`

**So:**  
- The **eigenvectors** define the spatial patterns.
- The **coefficients** specify how much each pattern contributes at a given time.
- Their combination reconstructs the full voltage distribution across all channels (the topomap).

