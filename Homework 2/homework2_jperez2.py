import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
    desMat = np.array([]) # create a numpy array
    range_definer = range(0, d+1)
    for i in range_definer:
        power = pow(x, i)
        desMat = np.append(desMat,power,axis = 0) #append to designMatrix
    desMat = np.reshape(desMat, (d+1,-1)) #reshape
    return np.linalg.solve(desMat.dot(np.transpose(desMat)), desMat.dot(y))

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    faces = faces.reshape(-1, 48, 48)
    faces = np.transpose(faces)
    faces = np.reshape(faces, (pow(faces.shape[0],2), faces.shape[2]))
    allOnes = np.ones((faces.shape[1]))
    faces = np.vstack((faces, allOnes))
    
    return faces

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    transpose = np.transpose((y - Xtilde.T.dot(w)))
    return  1/ 5000 * transpose.dot(y - Xtilde.T.dot(w))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    transpose = np.transpose(Xtilde[0:-1])
    guessIt = transpose.dot(w[0:-1]) + w[-1]
    tildeShape = Xtilde.shape[1]
    if alpha == 0:
        reg = 0
    else:
        reg = alpha / (2 * (tildeShape - 1))
        secondTran = np.transpose(w[0:-1])
        reg = reg * secondTran.dot(w[0:-1])
        reg = 0

    return (1 / float(tildeShape)) * (Xtilde.dot(guessIt - y)) + reg

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    transpose = np.transpose(Xtilde)
    dotProd = np.dot(Xtilde, transpose)
    dotProdShape = dotProd.shape[0]
    xEye = np.eye(dotProdShape)
    return np.dot(np.linalg.solve(dotProd, xEye), Xtilde.dot(y))

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    #print("The result from method 2 is", gradientDescent(Xtilde, y))
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 0.003  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations
    tildeShape = Xtilde.shape[0]
    
    w = 0.01 * np.random.randn(tildeShape)
    rangeDef = range(0, T)
    for i in rangeDef:
        w = w - EPSILON * gradfMSE(w, Xtilde, y, alpha)
    return w

def egregiousErrors(w, Xtilde, y):
    transpose = np.transpose(Xtilde)
    arr = np.array([])
    index = 0
    for i in transpose:
        dotProd = i[0:-1].dot(w[0:-1])
        guess = np.transpose(dotProd) + w[-1]
        powerCalc = pow((guess - y[index]), 2)
        error = pow(powerCalc,0.5)
        data = np.array([index, error, guess, y[index]])
        arr = np.append(arr, data, axis = 0)
        index = index + 1

    cleanForm =  np.reshape(arr, (-1, 4))
    argSort = cleanForm[:, 1].argsort()
    reverse = np.flip(cleanForm[argSort], 0)
    return reverse[0:5, :]

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)

    #print ("Training Set = ", fMSE(w1, Xtilde_tr, ytr), "Testing Set = ", fMSE(w1, Xtilde_te, yte))
    #print ("Training Set = ", fMSE(w2, Xtilde_tr, ytr), "Testing Set = ", fMSE(w2, Xtilde_te, yte))
    #print ("Training Set = ", fMSE(w3, Xtilde_tr, ytr), "Testing Set = ", fMSE(w3, Xtilde_te, yte))

    # Report fMSE cost using each of the three learned weight vectors
    # ...
    worst = egregiousErrors(w3, Xtilde_te, yte)
    #for _ in worst:
    #    print(_)
    #    transpose = np.transpose(Xtilde_te)
    #    image = transpose[int(_[0]), 0:-1]
    #    image = np.reshape(image, (48, 48))
    #    plt.imshow(image)
    #    plt.show()

    #firstIm = np.reshape(w1[:-1],(48,48))
    #plt.imshow(firstIm)
    #plt.show()

    #secondIm = np.reshape(w2[:-1], (48, 48))
    #plt.imshow(secondIm)
    #plt.show()

    #thirdIm = np.reshape(w3[:-1], (48, 48))
    #plt.imshow(thirdIm)
    #plt.show()
