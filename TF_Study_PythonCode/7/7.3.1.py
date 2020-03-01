import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# 这个路径是Vs code编辑时，相对于TF_Study_pythoncode目录，需要根据实际情况做修改
df = pd.read_csv("../data/boston.csv", header=0)
print(df.describe())

df = df.values
print("df.values", df)

dfArr = np.array(df);
print("dv.array", dfArr)

x_data = df[:,:12]
print("x_data")
print(x_data)
print("x_data.shape", x_data.shape)

y_data = df[:,12];
print("y_data")
print(y_data);
print("y_data.shape", y_data.shape)