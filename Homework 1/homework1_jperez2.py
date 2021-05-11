import numpy as np
#Jerry Perez - CS4342 Machine Learning

def problem1 (A, B):
    return A + B
#right-multiply A by B and then subtract C
def problem2 (A, B, C):
    return np.dot(A,B) - C
#Compute element-wise product for A and B plus the transpose of C
def problem3 (A, B, C):
    return (A*B) + np.transpose(C)
#Compute dot of s transpose and dot that by y
def problem4 (x, S, y):
    return np.dot(np.dot(np.transpose(x),S),y)
#Given matrix A return a matrix with same dimension but with all zeros
#Keep data type the same throughout
def problem5 (A):
    return np.zeros(A.shape, dtype = A.dtype)
#Given matrix A return a matrix with same dimension but with all ones
#Keep data type the same throughout
def problem6 (A):
    return np.ones(A.shape, dtype = A.dtype)
#Compute A + scalar * Identity matrix using same dimensions as A
def problem7 (A, alpha):
    rows, columns = A.shape
    return A + alpha * np.eye(rows, columns, dtype = A.dtype)
#Return the jth column of the ith row for matrix A
def problem8 (A, i, j):
    return A[i][j]
#Sum all of the entries in the ith row of matrix A
def problem9 (A, i):
    return np.sum(A[i])
#Given matrix A and scalars c,d... find arithmetic mean over a between c and d
def problem10 (A, c, d):
    return np.mean(np.nonzero(A>=c, A, A<=d))
#Given A(nxn) and k, return nxk eigenvectors with k largest eigenvalues
def problem11 (A, k):
    eigen = np.linalg.eig(A)[1]
    rows, columns = A.shape
    columns = columns - k
    return eigen[:,columns:]
#Compute A^-1x
def problem12 (A, x):
    return np.linalg.solve(A, x)
#Compute xA^-1
def problem13 (A, x):
    return np.transpose(np.linalg.solve(np.transpose(A), np.transpose(x)))
