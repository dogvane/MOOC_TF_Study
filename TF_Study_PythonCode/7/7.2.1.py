import numpy as np

scalar_value = 18
print(scalar_value)

scalar_np = np.array(scalar_value)
print(scalar_np, scalar_np.shape)

# 向量
vector_value = [1,2,3]
vector_np = np.array(vector_value)

print(vector_np, vector_np.shape)

# 矩阵
matrix_list = [[1,2,3],[4,5,6]]
matrix_np = np.array(matrix_list)

print('matrix_list=', matrix_list, '\n','matrix_np=\n', matrix_np, '\n', 'matrix_np.shape=', matrix_np.shape)

# 行向量
vector_row = np.array([[1,2,3]])
print(vector_row, 'shape=', vector_row.shape)

# 列向量
vector_column = np.array([[4], [5], [6]])
print(vector_column, 'shape=', vector_column.shape)

# 行列转置
vector_row = np.array([[1,2,3]])
print(vector_row, 'shape=', vector_row.shape, '\n', vector_row.T, '.Tshapre=', vector_row.T.shape)

# 矩阵运算

matrix_a = np.array([[1,2,3], [4,5,6]])

# 乘法
matrix_b = matrix_a * 2
print(matrix_b, 'shape=', matrix_b.shape)

# 加法
matrix_c = matrix_a + 2
print(matrix_c, 'shape=', matrix_c.shape)

# 矩阵+矩阵
matrix_a = np.array([[1,2,3], [4,5,6]])
matrix_b = np.array([[-1,-2,-3], [-4,-5,-6]])

matrix_c = matrix_a + matrix_b
print(matrix_c, 'shape=', matrix_c.shape)
