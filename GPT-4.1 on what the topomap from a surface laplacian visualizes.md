A **topomap** in this context is a spatial map showing the voltage (or other scalar quantity) at each EEG electrode, visualized over the scalp surface.

When you **project EEG data onto the eigenbasis of the surface Laplacian** (i.e., the Laplace-Beltrami eigenvectors), you are expressing the scalp voltage distribution as a sum of spatial "modes" (the eigenvectors), each with a time-varying coefficient. The **Laplace-Beltrami operator** computes these spatial eigenmodes based on the geometry of the electrode positions.

**Yes, you are correct:**  
- The Laplace-Beltrami operator constructs an eigenbasis (the eigenvectors) of the surface Laplacian.
- You project your EEG data onto this basis, so each time point (or epoch) is represented as a weighted sum of these spatial modes.
- A **topomap** reconstructed from these projections shows the global voltage distribution across the scalp at a given time, as represented in the Laplacian eigenbasis.

**In summary:**  
- The topomap visualizes the spatial pattern of EEG activity, reconstructed from the Laplacian eigenmodes.
- The Laplace-Beltrami eigenbasis provides a spatially smooth, physically meaningful decomposition of scalp potentials.