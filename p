# 1. Generating elementary signals like Unit Step, Ramp, Exponential, Sine, and 
#    Cosine sequences.

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



# 2. Demonstrates the effect of sampling, aliasing. 

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


# 3. Show that the highest rate of oscillation in a discrete-time sinusoidal is obtained when ω=π

#The rate of oscillation means how fast the signal alternates or cycles between peaks.
#In discrete-time signals, the oscillation rate depends on the angular frequency 𝜔
#ω

import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10, 11, 1)
y = np.cos(np.pi*n/2)

plt.title('cos(wn)')
plt.stem(n,y, label='cos(wn)')
plt.legend()
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()



# 4. Consider the continuous-time analog signal x(t)=3cos(100πt). Sample the analog 
# signal at 200 Hz and 75 Hz. Show the discrete-time signal after sampling. ⟹ 
# realization. 

import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(0, 0.04, 1000)  #(continious time) dense time values between 0 and 0.04 seconds (1000 points for smoothness)
y = 3 * np.cos(100*np.pi*n)
plt.plot(n, y, label = 'input signal')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.legend()

n = np.arange(0, 0.04, 1/75) #(discteat time) from 0 to 0.04 seconds in steps of 1/75 sec
y = y = 3 * np.cos(100*np.pi*n)
plt.stem(n, y, 'g', label = 'sample 75',  basefmt=" ")
plt.xlabel('time')
plt.ylabel('amplitude')
plt.legend()

n = np.arange(0, 0.04, 1/200)
y = y = 3 * np.cos(100*np.pi*n)
plt.stem(n, y, 'b', label = 'sample 200')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 5.Consider the analog signal: xa(t)=3cos(200πt)+5sin(600πt)+10cos(1200πt). 
#  Show the effect of sampling rate.

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



# 6. The impulse response of a discrete-time LTI system is h(n)={u(n)−u(n−5)}.
#Determine the output of the system for the input x[n]=u(n), using the convolution
#sum

import numpy as np
import matplotlib.pyplot as plt

def unit(n):
    return np.where(n>=0, 1, 0) #Returns 1 when 𝑛≥0, else 0

def x(n):
    return unit(n)

def h(n):
    return unit(n) - unit(n-5) #1 for 0≤n<5,0 elsewhere


n = np.arange(-4, 10) #Creates a time axis n from -4 to 9 (both inclusive)

plt.subplot(3,1,1)
plt.stem(n, x(n), label="unit step input", linefmt='b-', basefmt= ' ')
plt.grid(True)
plt.legend()
#basefmt=' ' hides base line

plt.subplot(3,1,2)
plt.stem(n, h(n),label="unit impulse response", linefmt ='g-', basefmt=' ')
plt.legend()
plt.grid(True)

y = np.zeros(len(n)) # Initialize output array y with all zeros — same length as n.

for i in range(len(n)):
    sum = 0
    for k in range(len(n)):
        if i-k >= 0:
            sum += x(k) * h(i-k)
    y[i] = sum


plt.subplot(3,1,3)
plt.stem(n, y, label="Convolutin sum", linefmt='b-', basefmt=' ')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 7. Given 
# x(n)=[1,3,−2,4] 
# y(n)=[2,3,−1,3] 
# z(n)=[2,−1,4,−2] 
# Find the correlation between x(n) & y(n) and y(n) & z(n). ⟹ observe the 
# realization.

import numpy as np
import matplotlib.pyplot as plt

# Define the sequences
x = np.array([1, 3, -2, 4])
y = np.array([2, 3, -1, 3])
z = np.array([2, -1, 4, -2])


# Function to calculate normalized correlation 
def normalized_corr(x, y):
    numerator = np.sum(x * y)
    denominator = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2))
    return numerator / denominator


# Calculate the normalized correlation
def correlation(x, y):
        N = len(x) + len(y) - 1
        result = np.zeros(N)

        for i in range(N):
              sum = 0
              for k in range(len(x)): #Loop through the values of x[k].
                    if i-k>=0 and i-k<len(y):#Check if y[i-k] is within valid index range (to avoid indexing errors)
                        sum += x[k] * y[i-k]
              result[i] = sum
        return result


r_xy = normalized_corr(x, y) 
r_yz = normalized_corr(y, z)
s = str(r_xy)#Converts r_xy (an array) to string,
s = 'correlation value: ' + s[:5]#takes only the first 5 characters of the string.

r_xy_0 = correlation(x, y[::-1])#y[::-1] means reversed y in time (time-flipping)
r_yz_0 = correlation(y, z[::-1])
lag = np.arange(-len(x) + 1, len(y))#Lags=[−3,−2,−1,0,1,2,3]

# Display the results
plt.subplot(2, 1, 1)
plt.title('Correlation between x(n) and y(n)')
plt.stem(lag , r_xy_0, label= s, linefmt='b-', basefmt='k-')
plt.legend()
plt.xlabel('Lag')
plt.ylabel('Amplitude')
plt.grid(True)

s = str(r_yz)
s = 'correlation value: ' +s[:5]

plt.subplot(2, 1, 2)
plt.title('Correlation between y(n) and z(n)')
plt.stem(lag, r_yz_0, label= s, linefmt='g-', basefmt='k-')
plt.legend()
plt.xlabel('Lag')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()


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