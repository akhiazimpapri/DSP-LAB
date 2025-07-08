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


n = np.arange(-4, 10)
# h1 = unit(n) - unit(n-5)
# x1 = unit(n)

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