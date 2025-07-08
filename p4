# '''
# Consider the continuous-time analog signal x(t)=3cos(100πt). Sample the analog 
# signal at 200 Hz and 75 Hz. Show the discrete-time signal after sampling. ⟹ 
# realization. 

# '''
import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(0, 0.04, 1000) #n: dense time values between 0 and 0.04 seconds (1000 points for smoothness)
y = 3 * np.cos(100*np.pi*n)
plt.plot(n, y, label = 'input signal')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.legend()

n = np.arange(0, 0.04, 1/75) #You're creating sample times: from 0 to 0.04 seconds in steps of 1/75 sec
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