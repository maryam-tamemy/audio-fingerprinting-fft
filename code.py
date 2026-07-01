import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black',
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'legend.frameon': True,
    'legend.edgecolor': '#CCCCCC',
})

AUDIO_PATH = 'full_audio.wav'
CLIP_START_S = 5.0
CLIP_DURATION_S = 2.0

fs, raw = wavfile.read(AUDIO_PATH)

if raw.ndim == 2:
    raw = raw.mean(axis=1)

if raw.dtype == np.int16:
    full_signal = raw.astype(np.float64) / 32768.0
elif raw.dtype == np.int32:
    full_signal = raw.astype(np.float64) / 2147483648.0
else:
    full_signal = raw.astype(np.float64)

signal_length = len(full_signal)
duration_s = signal_length / fs

print('=' * 60)
print('AUDIO MATCHING USING FFT — COMM 401')
print('=' * 60)
print(f'Sampling frequency      : {fs} Hz')
print(f'Length of full signal   : {signal_length} samples / {duration_s:.1f}s')

clip_start = int(CLIP_START_S * fs)
clip_length = int(CLIP_DURATION_S * fs)
clip = full_signal[clip_start : clip_start + clip_length]

print(f'Clip length             : {clip_length} samples / {CLIP_DURATION_S:.1f}s')
print(f'Original clip position  : {CLIP_START_S:.3f}s / sample {clip_start}')

t_full = np.arange(signal_length) / fs
t_clip = np.arange(clip_length) / fs

def compute_fft(segment, sampling_rate):
    N = len(segment)
    fft_vals = np.fft.rfft(segment)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(N, 1/sampling_rate)
    return freqs, fft_mag

clip_freqs, clip_fft = compute_fft(clip, fs)
clip_norm = np.linalg.norm(clip_fft)

step_size = int(0.5 * fs)
scores = []
positions = []

for i in range(0, signal_length - clip_length, step_size):
    segment = full_signal[i : i + clip_length]

    _, seg_fft = compute_fft(segment, fs)
    seg_norm = np.linalg.norm(seg_fft)

    if seg_norm == 0 or clip_norm == 0:
        score = 0.0
    else:
        score = np.dot(clip_fft, seg_fft) / (clip_norm * seg_norm)

    scores.append(score)
    positions.append(i / fs)

best_idx = np.argmax(scores)
detected_time = positions[best_idx]
best_score = scores[best_idx]

print(f'Detected position       : {detected_time:.2f}s')
print(f'Best similarity score   : {best_score:.4f}')
print('=' * 60)

plt.figure(figsize=(10, 3))
plt.plot(t_full, full_signal, color='#1f77b4', linewidth=0.5, label='Full Signal')
plt.title("Full Signal (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(t_clip, clip, color='#2ca02c', linewidth=0.8, label='Query Clip')
plt.title("Clip (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(clip_freqs, clip_fft, color='#9467bd', linewidth=0.8, label='FFT Magnitude')
plt.title("Clip (Frequency Domain)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(positions, scores, color='#1f77b4', linewidth=0.9, label='Similarity Score')
plt.axvline(CLIP_START_S, color='green', linewidth=2, linestyle='--',
            label=f'Original Position ({CLIP_START_S}s)')
plt.axvline(detected_time, color='red', linewidth=2, linestyle='--',
            label=f'Detected Position ({detected_time:.2f}s)')
plt.scatter([detected_time], [best_score], color='red',
            zorder=5, s=100,
            label=f'Best Score = {best_score:.4f}')

plt.title("Similarity Score vs Time")
plt.xlabel("Time [s]")
plt.ylabel("Similarity")
plt.legend(loc='upper right', fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.show()

detected_seg = full_signal[
    int(detected_time * fs):
    int(detected_time * fs) + clip_length
]

plt.figure(figsize=(10, 4))
plt.plot(t_clip, clip,
         color='#2ca02c',
         linewidth=0.8,
         label='Original Clip')

plt.plot(t_clip, detected_seg,
         color='#d62728',
         linewidth=0.8,
         label='Detected Segment',
         alpha=0.85)

plt.title(f"Clip vs Detected Segment  (t = {detected_time:.2f}s  |  Score = {best_score:.4f})")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()