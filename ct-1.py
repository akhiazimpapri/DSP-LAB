##consider a differencing equation y(n)=(1-a)*y(n-1)+a*x(n)
# x(n)={1,2,2,10,2,2,1} here 0<a<1 obsevation of y(n) apply different value 
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 2, 10, 2, 2, 1]
a_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Different smoothing factors
n = np.arange(len(x))

plt.figure(figsize=(10, 6))
plt.plot(n, x, 'ko--', label="Input x(n)", linewidth=2)

# Apply the filter for different values of a
for a in a_values:
    y = [0]  # y(0) = 0
    for i in range(1, len(x)):
        y.append((1 - a) * y[i - 1] + a * x[i])
    plt.plot(n, y, label=f"a = {a}")

plt.title("Output y(n) for Different 'a' Values")
plt.xlabel("n")
plt.ylabel("y(n)")
plt.grid(True)
plt.legend()
plt.show()   