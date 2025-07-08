# 8. Filter realization using 6-point averaging, 6-point differencing equations.

#6-point averaging filter = Low-pass filter (smooths the signal)
#6-point differencing filter = High-pass filter (detects sharp changes

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = np.linspace(0,1,200)
x = np.sin(2*np.pi*5*n) + 0.6*np.random.rand(len(n))

def averaging(x):
    y = np.zeros_like(x)#nitializes y as zeros.
    for i in range(5, len(x)):
        y[i] = x[i] + x[i-1] + x[i-2] + x[i-3] + x[i-4] + x[i-5]
    return y    


def differencing(x):
    y = np.zeros_like(x)
    for i in range(5, len(x)):
        y[i] = x[i] - x[i-1] + x[i-2] - x[i-3] + x[i-4] - x[i-5]
    return y 

avg_filter = averaging(x)
diff_filter = differencing(x)

#Plot Orginal Siganl
plt.subplot(3, 1, 1)
plt.plot(n, x, 'g', label='Input signal')
plt.title('Orginal Signal')
plt.grid(True)

#Plot Averaging Siganl
plt.subplot(3, 1, 2)
plt.plot(n, avg_filter, 'r', label='Averaging signal')
plt.title('Averaging Signal')
plt.grid(True)

#Plot Differencing Siganl
plt.subplot(3, 1, 3)
plt.plot(n, diff_filter, 'b', label='Differencing signal')
plt.title('Dufferencing Signal')
plt.grid(True)

plt.tight_layout()
plt.legend()
plt.show()

# Differencing Signal (Blue, Bottom Plot)
# This is the high-pass filtered version.

# Removes the smooth part (like the sine wave) and keeps sharp changes.

# Emphasizes the random noise and fast transitions.

# The plot appears spiky and centered around zero.

# Use-case: Great for edge detection, detecting transitions, or finding sudden changes in a signal.