import numpy as np
import matplotlib.pyplot as plt

# Continuous signal parameters
f_signal = 5  # 5 Hz sine wave
t_cont = np.linspace(0, 1, 1000)  # Continuous time from 0 to 1 sec
x_cont = np.sin(2 * np.pi * f_signal * t_cont)

# Sampling frequencies
fs_high = 30    # >> 2*f (safe)
fs_nyquist = 10 # = 2*f (borderline)
fs_low = 6      # < 2*f (aliasing)

# Discrete sample points
normal_f = np.arange(0, 1, 1/f_signal)
x_normal = np.sin(2 * np.pi * f_signal * normal_f)

n_high = np.arange(0, 1, 1/fs_high)
x_high = np.sin(2 * np.pi * f_signal * n_high)

n_nyquist = np.arange(0, 1, 1/fs_nyquist)
x_nyquist = np.sin(2 * np.pi * f_signal * n_nyquist)

n_low = np.arange(0, 1, 1/fs_low)
x_low = np.sin(2 * np.pi * f_signal * n_low)

# Plotting
plt.figure(figsize=(15, 8))

# 1.Normal
plt.subplot(2, 2, 1)
plt.plot(t_cont, x_cont, label='Original Continuous Signal (5Hz)')
plt.stem(normal_f, x_normal, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled at 5 Hz')
plt.title("Normal (fs = 5 Hz > 2*f)")
plt.xlabel("Time [s]"), plt.ylabel("Amplitude")
plt.legend(loc="best")
plt.grid(True)

# 2. Continuous Signal
plt.subplot(2, 2, 2)
plt.plot(t_cont, x_cont, label='Original Continuous Signal (5Hz)')
plt.stem(n_high, x_high, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled at 30 Hz')
plt.title("No Aliasing (fs = 30 Hz > 2*f)")
plt.xlabel("Time [s]"), plt.ylabel("Amplitude")
plt.legend(loc="upper left")
plt.grid(True)

# 3. Nyquist Rate
plt.subplot(2, 2, 3)
plt.plot(t_cont, x_cont, label='Original Continuous Signal (5Hz)')
plt.stem(n_nyquist, x_nyquist, linefmt='g-', markerfmt='go', basefmt='g-', label='Sampled at 10 Hz')
plt.title("Nyquist Rate Sampling (fs = 10 Hz = 2*f)")
plt.xlabel("Time [s]"), plt.ylabel("Amplitude")
plt.legend(loc="best")
plt.grid(True)

# 4. Aliasing Case
plt.subplot(2, 2, 4)
plt.plot(t_cont, x_cont, label='Original Continuous Signal (5Hz)')
plt.stem(n_low, x_low, linefmt='m-', markerfmt='mo', basefmt='m-', label='Sampled at 6 Hz')
plt.title("Aliasing Occurs (fs = 6 Hz < 2*f)")
plt.xlabel("Time [s]"), plt.ylabel("Amplitude")
plt.legend(loc="best")
plt.grid(True)

plt.tight_layout()
plt.show()
