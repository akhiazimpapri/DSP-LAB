import numpy as np
import matplotlib.pyplot as plt

# Generate time samples from 0 to 2 seconds (200 points)
n = np.linspace(0, 2, 200)

# Generate a pure 300 Hz sine wave
pure_signal = np.sin(2 * np.pi * 300 * n)

# Add Gaussian noise to simulate real-world signal corruption
noise = 0.1 * np.random.randn(len(n))
x = pure_signal + noise  # Noisy input signal

def compute(x, alpha):
    y = []
    y_previous = 0  # Initial condition

    for n in range(len(x)):
        y_current = (1 - alpha) * y_previous + alpha * x[n]
        y.append(y_current)
        y_previous = y_current

    return y

# Apply EMA filtering with different alpha values
y1 = compute(x, 0.25)
y2 = compute(x, 0.5)
y3 = compute(x, 0.8)

# Plot the original and filtered signals
plt.figure(figsize=(12,6))
plt.plot(x, label='Input Signal x(n)', color='gray')
plt.plot(y1, label='Filtered y(n), alpha = 0.25', color='green')
plt.plot(y2, label='Filtered y(n), alpha = 0.5', color='red')
plt.plot(y3, label='Filtered y(n), alpha = 0.8', color='blue')

plt.title('Exponential Moving Average Filtering')
plt.xlabel('n (sample index)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()