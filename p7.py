# Given 
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