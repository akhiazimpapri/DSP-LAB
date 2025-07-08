
#1. Unit Step Signal
import numpy as np
import matplotlib.pyplot as plt

# Time ranges
n = np.arange(-10, 11, 1)       # Discrete time
t = np.linspace(-10, 10, 1000)  # Continuous time

# Unit step
u_n = np.where(n >= 0, 1, 0)
u_t = np.where(t >= 0, 1, 0)

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.stem(n, u_n) # is used for discrete signals (with stems)
plt.title("Discrete Unit Step")
plt.xlabel("n"), plt.ylabel("u[n]")

plt.subplot(1, 2, 2)
plt.plot(t, u_t) # draws a smooth curve
plt.title("Continuous Unit Step")
plt.xlabel("t"), plt.ylabel("u(t)")

plt.tight_layout()
plt.show()


# 2.Ramp
r_n = np.where(n >= 0, n, 0)
r_t = np.where(t >= 0, t, 0)

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.stem(n, r_n)
plt.title("Discrete Ramp Signal")
plt.xlabel("n"), plt.ylabel("r[n]")

plt.subplot(1, 2, 2)
plt.plot(t, r_t)
plt.title("Continuous Ramp Signal")
plt.xlabel("t"), plt.ylabel("r(t)")

plt.tight_layout()
plt.show()

#  3. Exponential Signal
a = 0.8
x_n = np.where(n >= 0, a ** n, 0)
x_t = np.where(t >= 0, np.exp(a * t), 0)

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.stem(n, x_n)
plt.title("Discrete Exponential Signal")
plt.xlabel("n"), plt.ylabel("x[n]")

plt.subplot(1, 2, 2)
plt.plot(t, x_t)
plt.title("Continuous Exponential Signal")
plt.xlabel("t"), plt.ylabel("x(t)")

plt.tight_layout()
plt.show()



#4. Sine Signal

# Discrete-time index
n = np.arange(0, 20, 1)   # n = 0 to 19
f = 0.1                   # Frequency in Hz

# Discrete-time signal
x_discrete = np.sin(2 * np.pi * f * n)

# Continuous-time signal
t = np.linspace(0, 19, 1000)  # More points between 0 and 19 for smoothness
x_continuous = np.sin(2 * np.pi * f * t)

# Plot both signals
plt.figure(figsize=(10, 5))

# Continuous signal as line
plt.plot(t, x_continuous, label='Continuous-Time Sine', color='blue')

# Discrete signal as stem
plt.stem(n, x_discrete, basefmt=' ', linefmt='r-', markerfmt='ro', label='Discrete-Time Sine')

# Labels and grid
plt.title('Continuous-Time vs Discrete-Time Sine Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 5. Cosine Signal
x_n = np.cos(2 * np.pi * f * n)
x_t = np.cos(2 * np.pi * f  * t)

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.stem(n, x_n)
plt.title("Discrete Cosine Signal")
plt.xlabel("n"), plt.ylabel("x[n]")

plt.subplot(1, 2, 2)
plt.plot(t, x_t)
plt.title("Continuous Cosine Signal")
plt.xlabel("t"), plt.ylabel("x(t)")

plt.tight_layout()
plt.show()