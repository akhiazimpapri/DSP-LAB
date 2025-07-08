'''
Show that the highest rate of oscillation in a discrete-time sinusoidal is obtained 
when ω=π.  
'''

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