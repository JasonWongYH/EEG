import pandas as pd
import re
import mne

file_path = 'rp-train-[2025.07.16-14.19.00]-Stuart.csv'
output_file = 'output.gdf'

# --- 1. Read header and extract sampling frequency ---
with open(file_path, 'r') as f:
    header = f.readline().strip()

sfreq_match = re.search(r'(\d+)Hz', header)
if not sfreq_match:
    raise ValueError("Sampling frequency not found in header.")
sfreq = int(sfreq_match.group(1))
print(f"Sampling frequency found: {sfreq} Hz")

# --- 2. Determine columns to load (up to 'Event Duration') ---
cols = header.split(',')
if 'Event Duration' in cols:
    last_col = cols.index('Event Duration') + 1
    usecols = cols[:last_col]
else:
    raise ValueError("Event Duration column not found in header.")

# --- 3. Load data ---
df = pd.read_csv(file_path, header=0, usecols=usecols)
df.rename(columns={df.columns[0]: 'Time'}, inplace=True)

# --- 4. Identify data channels ---
ch_names = list(df.loc[:, 'Channel 1':'preProcessedEMG'].columns)
data = df[ch_names].values.T  # shape: (n_channels, n_samples)

# --- 5. Identify events ---
events_df = df[['Time', 'Event Id', 'Event Duration']].copy()
events_df.dropna(subset=['Event Id'], inplace=True)

print(f"Found {len(ch_names)} data channels.")
print(f"Found {len(events_df)} events.")

# --- 6. Create MNE Info and Raw objects ---
ch_types = ['eeg'] * (len(ch_names) - 1) + ['emg']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(data, info)

# --- 7. Add annotations if present ---
if not events_df.empty:
    onset = events_df['Time'].astype(float).values
    duration = events_df['Event Duration'].fillna(0).astype(float).values
    description = events_df['Event Id'].astype(str).values
    annots = mne.Annotations(onset=onset, duration=duration, description=description)
    raw.set_annotations(annots)

# --- 8. Export to GDF ---
mne.io.write_raw_gdf(raw, output_file, overwrite=True)
print(f"\n✅ Successfully converted data to {output_file}")

# convert that data into a GDF (General Data Format) file using the **MNE-Python** library: reading the data, correctly parsing the channels and event markers, and then exporting it.

# The process requires a few data manipulation steps because your CSV file has metadata (the sampling rate) in the header and combines continuous data with event information in the same columns.

# First, you'll need to install the necessary Python libraries if you haven't already. MNE uses the `pygdf` library for GDF support.

# ```bash
# pip install mne pandas pygdf
# ```

# 1.  **Load and Prepare Data**:

#       * We first read only the **header line** of your file to find the sampling frequency (`1024Hz`) using a regular expression.
#       * Then, we load the rest of the file into a **pandas DataFrame**.
#       * The first column's name is cleaned up to just `Time`.

# 2.  **Separate Data and Events**:

#       * We identify the columns that contain the continuous signal (from `Channel 1` to `preProcessedEMG`).
#       * The data from these columns is extracted into a NumPy array and **transposed** to fit the `(channels x samples)` shape required by MNE.
#       * We then separately handle the event columns (`Event Id`, etc.) by creating a new DataFrame and dropping any rows that don't contain an event.

# 3.  **Create MNE Info**:

#       * An `mne.Info` object is a container for all the metadata.
#       * We provide it with the **channel names**, the **sampling frequency** we found, and the **channel types**. **Note:** I've assumed the first 90 channels are 'eeg' and the last one is 'emg'. You should adjust this if you know the correct types.

# 4.  **Create Raw Object**:

#       * `mne.io.RawArray` is used to create the main MNE data object from the NumPy data array and the `Info` object.

# 5.  **Add Annotations**:

#       * MNE stores events as `Annotations`. We create this object using the onset times (`Time` column), durations, and descriptions (`Event Id`) from our events DataFrame.
#       * These annotations are then attached to the `raw` object using `raw.set_annotations()`.

# 6.  **Export to GDF**:

#       * Finally, `mne.io.write_raw_gdf()` is called to save the fully assembled `raw` object, including its data and annotations, into a GDF file.