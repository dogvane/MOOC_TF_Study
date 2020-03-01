import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(5)

# 等差数列，100个点， -1~+1之间
x_data = np.linspace(-1, 1, 100)

y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape)*0.4

# print(x_data)
# print(y_data)

plt.scatter(x_data, y_data)

plt.plot(x_data, 2 * x_data + 1.0, color='red', linewidth=3)
#plt.savefig('6.2.2.png')
plt.show()