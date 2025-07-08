import numpy as np
import matplotlib.pyplot as plt

# Continuous time signal
t = np.linspace(0, 0.01, 1000)  # Time from 0 to 10ms
x = 3*np.cos(200*np.pi*t) + 5*np.sin(600*np.pi*t) + 10*np.cos(1200*np.pi*t)

# Sampling rates to try
fs_list = [5000, 1200, 800]  # Hz
titles = ['No Aliasing (fs=5000 Hz)', 'Nyquist (fs=1200 Hz)', 'Aliasing (fs=800 Hz)']
colors = ['r', 'g', 'm']

plt.figure(figsize=(12, 8))#tarts a new figure with 12x8 inch size to fit all subplots.

for i, fs in enumerate(fs_list): #It’s a for loop that goes through the list fs_list, which contains the sampling
    ts = np.arange(0, 0.01, 1/fs)  # Sample points
    xs = 3*np.cos(200*np.pi*ts) + 5*np.sin(600*np.pi*ts) + 10*np.cos(1200*np.pi*ts)

    plt.subplot(3, 1, i+1)
    plt.plot(t, x, label="Original Signal", color='black')
    plt.stem(ts, xs, linefmt=colors[i]+'-', markerfmt=colors[i]+'o', basefmt="gray", label=f"Sampled at {fs} Hz")
    plt.title(titles[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
