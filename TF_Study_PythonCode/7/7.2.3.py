import numpy as np

matrix_a = np.array([[1,2,3], [4,5,6]])
matrix_b = np.array([[1],[2],[3]])

matrix_d = np.matmul(matrix_a, matrix_b)
print(matrix_d, 'shape=', matrix_d.shape)

matrix_a = np.array([[1,2,3]])
matrix_b = np.array([[2],[4],[-1]])

matrix_d = np.matmul(matrix_a, matrix_b)
print(matrix_d, 'shape=', matrix_d.shape)

print('matrix_a')
matrix_a = np.array([[1,2,3], [4,5,6]])
print(matrix_a, 'shape=', matrix_a.shape, '\n', matrix_a.T, '.Tshapre=', matrix_a.T.shape)


# 行列转置
print('#行列转置')
vector_row = np.array([[1,2,3]])
print(vector_row, 'shape=', vector_row.shape, '\n', vector_row.T, '.Tshapre=', vector_row.T.shape)

print('#reshape')
vector_row = np.array([[1,2,3]])
vector_column = vector_row.reshape(3,1)

print(vector_row, 'shape=', vector_row.shape, '\n', vector_column, 'reshape=', vector_column.T.shape)


