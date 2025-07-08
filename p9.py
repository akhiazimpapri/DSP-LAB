# 9. FT of xa(t)=sin(2π⋅1000t)+0.5sin(2π⋅2000t+4π). Also IDFT. DFT with 
#    window + window function realization

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 64
fs = 8000
n = np.arange(N)
t = n / fs

# Input signal
def xa(t):
    return np.sin(2000 * np.pi * t) + 0.5 * np.sin(4000 * np.pi * t + 4 * np.pi)

# Hanning window function
def hanning(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))  # Corrected formula

# Manual DFT
def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Manual IDFT
def IDFT(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):  # Fixed range
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    x /= N  # Moved outside loop
    return x

# Frequency vector
a_freq = np.arange(N) * fs / N
x = xa(t)

# Apply Hanning
Hanning = hanning(N)
Hanning_x = x * Hanning

# DFTs
dft = DFT(x)
Hanning_dft = DFT(Hanning_x)

# IDFTs
idft = IDFT(dft)
Hanning_idft = IDFT(Hanning_dft)

# ----------------------------- Plotting -----------------------------

plt.figure(figsize=(12, 10))

# Original vs Hanning windowed signal
plt.subplot(3, 2, 1)
plt.plot(t, x, label='Original Signal', color='g')
plt.plot(t, Hanning_x, label='Hanning Windowed Signal', color='r')
plt.title('Time Domain Signals')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()

# Hanning window shape
plt.subplot(3, 2, 2)
plt.stem(n, Hanning, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.title('Hanning Window')
plt.grid(True)

# Magnitude of DFT without window
plt.subplot(3, 2, 3)
plt.stem(a_freq, abs(dft), linefmt='y-', markerfmt='yo', basefmt='k-')
plt.title('DFT Magnitude (No Window)')
plt.grid(True)

# Magnitude of DFT with Hanning window
plt.subplot(3, 2, 4)
plt.stem(a_freq, abs(Hanning_dft), linefmt='m-', markerfmt='mo', basefmt='k-')
plt.title('DFT Magnitude (Hanning Window)')
plt.grid(True)

# Phase of DFT with Hanning
plt.subplot(3, 2, 5)
plt.stem(a_freq, np.angle(Hanning_dft, deg=True), linefmt='c-', markerfmt='co', basefmt='k-')
plt.title('DFT Phase (Hanning Window)')
plt.grid(True)

# Reconstructed signals from IDFT
plt.subplot(3, 2, 6)
plt.plot(t, idft.real, label='IDFT of Original', color='b')
plt.plot(t, Hanning_idft.real, label='IDFT of Hanning Windowed', color='orange')
plt.title('Reconstructed Signal via IDFT')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()