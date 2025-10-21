import time
import numpy as np
import mne
import joblib
import csv
from datetime import datetime

# Add these imports for LSL and PyRiemann pipeline
from pylsl import StreamInlet, resolve_stream
import pickle
import pyriemann
from scipy.signal import butter, sosfiltfilt

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    meandat = np.mean(data, axis=1)
    data = data - meandat[:, np.newaxis]
    y = sosfiltfilt(sos, data)
    return y

def buffer_to_riemann_classifier(
    buffer_data,
    model_path="mdm-model.pkl",
    lowbp=8,
    highbp=30,
    filterorder=4,
    sfreq=256,
    ch_indices=None
):
    # buffer_data: (1, n_channels, n_times)
    # Optionally select channels
    if ch_indices is not None:
        buffer_data = buffer_data[:, ch_indices, :]
    # Bandpass filter
    filtered = butter_bandpass_filter(buffer_data[0], lowbp, highbp, sfreq, filterorder)
    filtered = np.expand_dims(filtered, axis=0)  # (1, n_channels, n_times)
    # Covariance
    cov = pyriemann.estimation.Covariances().fit_transform(filtered)
    # Load model
    with open(model_path, "rb") as f:
        trained = pickle.load(f)
    mdm = pyriemann.classification.MDM()
    mdm.metric = 'Riemann'
    mdm.fit(trained['COV'], trained['Labels'])
    pred = mdm.predict(cov)
    return pred

def simulate_eeg_stream_gdf(
    gdf_file,
    classifier_func,
    classifier_args=None,
    buffer_ms=200,
    picks=None,
    log_file="stream_log.csv"
):
    """
    Simulate real-time EEG streaming by reading a GDF file and buffering every buffer_ms milliseconds.
    Each buffer is sent to the classifier, and the result is logged with a timestamp.
    """
    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose='error')
    sfreq = raw.info['sfreq']
    buffer_samples = int(buffer_ms / 1000 * sfreq)
    n_samples = raw.n_times
    ch_names = raw.ch_names if picks is None else picks
    ch_indices = [raw.ch_names.index(ch) for ch in ch_names]
    classifier_args = classifier_args or {}

    print(f"Simulating real-time stream from {gdf_file} at {sfreq} Hz, buffer size: {buffer_samples} samples ({buffer_ms} ms)")
    with open(log_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "buffer_start", "buffer_stop", "prediction"])
        for start in range(0, n_samples - buffer_samples + 1, buffer_samples):
            stop = start + buffer_samples
            buffer_data, times = raw[ch_indices, start:stop]
            buffer_data = buffer_data[np.newaxis, :, :]
            pred = classifier_func(buffer_data, **classifier_args)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            writer.writerow([timestamp, start, stop, int(pred[0])])
            print(f"[{timestamp}] Buffer {start}:{stop} Prediction: {pred[0]}")
            # Optional: simulate real time
            # time.sleep(buffer_ms / 1000.0)

def stream_eeg_lsl(
    classifier_func,
    classifier_args=None,
    buffer_ms=200,
    sfreq=256,
    n_channels=8,
    log_file="lsl_stream_log.csv"
):
    """
    Stream from an LSL EEG source using mne-realtime (mne-lsl). This implementation uses
    mne_realtime.LSLClient to retrieve blocks of samples and passes fixed-size buffers
    to the classifier function.
    """
    classifier_args = classifier_args or {}

    try:
        from mne_realtime import LSLClient
    except Exception as e:
        raise RuntimeError("mne_realtime (mne-lsl) is required for LSL streaming. Install with: pip install mne-realtime") from e

    print("Connecting to LSL stream via mne_realtime.LSLClient (mne-lsl)...")
    # Create an LSL client that listens for an EEG stream. By default it will resolve by type='EEG'.
    client = LSLClient(name=None, type='EEG', timeout=1.0)

    buffer_samples = int(buffer_ms / 1000 * sfreq)
    print(f"Receiving LSL stream: {n_channels} channels, buffer size: {buffer_samples} samples ({buffer_ms} ms)")

    with open(log_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "buffer_start", "buffer_stop", "prediction"])
        sample_count = 0
        while True:
            # Request exactly buffer_samples; get_data may block until that many samples have arrived.
            data = client.get_data(buffer_samples)
            if data is None or data.size == 0:
                # nothing received, wait a bit
                time.sleep(0.01)
                continue

            # Ensure shape is (n_channels, n_samples)
            if data.shape[0] != n_channels and data.shape[1] == n_channels:
                data = data.T

            # If more than buffer_samples were returned, process in sliding chunks
            cur_idx = 0
            while data.shape[1] - cur_idx >= buffer_samples:
                chunk = data[:, cur_idx:cur_idx + buffer_samples]
                cur_idx += buffer_samples
                buffer_data = chunk[np.newaxis, :, :]  # (1, n_channels, buffer_samples)
                pred = classifier_func(buffer_data, **classifier_args)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                writer.writerow([timestamp, sample_count, sample_count + buffer_samples, int(pred[0])])
                print(f"[{timestamp}] Buffer {sample_count}:{sample_count + buffer_samples} Prediction: {pred[0]}")
                sample_count += buffer_samples

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['gdf', 'lsl'], default='gdf', help='Run in GDF simulation mode or LSL real-time mode')
    parser.add_argument('--gdf_file', default="data/Andrei.gdf")
    parser.add_argument('--model_path', default="mdm-model.pkl")
    parser.add_argument('--buffer_ms', type=int, default=200)
    parser.add_argument('--n_channels', type=int, default=8)
    parser.add_argument('--sfreq', type=int, default=256)
    parser.add_argument('--log_file', default="stream_log.csv")
    parser.add_argument('--lowbp', type=int, default=8)
    parser.add_argument('--highbp', type=int, default=30)
    parser.add_argument('--filterorder', type=int, default=4)
    args = parser.parse_args()

    classifier_args = {
        "model_path": args.model_path,
        "lowbp": args.lowbp,
        "highbp": args.highbp,
        "filterorder": args.filterorder,
        "sfreq": args.sfreq
    }

    if args.mode == 'gdf':
        simulate_eeg_stream_gdf(
            gdf_file=args.gdf_file,
            classifier_func=buffer_to_riemann_classifier,
            classifier_args=classifier_args,
            buffer_ms=args.buffer_ms,
            picks=None,  # or specify channel names
            log_file=args.log_file
        )
    elif args.mode == 'lsl':
        stream_eeg_lsl(
            classifier_func=buffer_to_riemann_classifier,
            classifier_args=classifier_args,
            buffer_ms=args.buffer_ms,
            sfreq=args.sfreq,
            n_channels=args.n_channels,
            log_file=args.log_file
        )