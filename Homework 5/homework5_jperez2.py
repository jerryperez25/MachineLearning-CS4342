import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt

def phiPoly3 (x):
    radon = x[:, 0]
    asbestos = x[:, 1]
    phiPol = manipMatricies(radon, asbestos)
    return phiPol

def kerPoly3 (x, xprime):
    xarr = np.array(x)
    xprimearr = np.array(xprime)
    xprimeTransp = np.transpose(xprimearr)
    dotProd = np.dot(xarr, xprimeTransp)
    kerPol = pow((dotProd + 1),3) 
    return kerPol

def showPredictions (title, svm, X):  # feel free to add other parameters if desired
    #plt.scatter(..., ...)  # positive examples
    #plt.scatter(..., ...)  # negative examples
    yInc = 2
    xInc = .1
    pos = np.zeros([2,2])
    neg = np.zeros([2,2])
    xIncCalc = int(10/xInc)
    rangeDeter = range(0,xIncCalc)
    xIt = iterateSecStage(xInc, rangeDeter)
    for xCoor in xIt:
        yIncCalcOne = int(50/yInc)
        yIncCalcTwo = int(200/yInc)
        rangeDeter = range(yIncCalcOne,yIncCalcTwo)
        yIt = iterateSecStage(yInc, rangeDeter)
        for yCoor in yIt:
            if X.shape[1] == 2:
                predict = svm.predict([[xCoor,yCoor]])
            else:
                arr = np.array([[xCoor,yCoor],[xCoor,yCoor]])
                input = phiPoly3(arr)
                predict = svm.predict([input[:,1]])
            if predict <= 0:
                arr = np.array([[xCoor,yCoor]])
                neg = np.append(neg, arr, axis=0)
            else:
                arr = np.array([[xCoor,yCoor]])
                pos = np.append(pos, arr, axis=0)

    pos = pos[2:,:]
    neg = neg[2:, :]
    plt.scatter(neg[:, 0], neg[:, 1])
    plt.scatter(pos[:,0], pos[:, 1])
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,y1,160))
    plt.xlabel("Radon")
    plt.ylabel("Asbestos")
    plt.legend(["No lung disease", "Lung disease"])
    plt.title(title)
    plt.show()

def manipMatricies(radon, asbestos):
    asbestosTranspose = np.transpose(asbestos)
    radonTranspose = np.transpose(radon)
    asbestosShape = asbestos.shape
    oneConv = np.ones(asbestosShape)

    #Matrix Manipulation
    dNine = np.sqrt(3) 
    dNine = dNine * asbestos
    dSix = np.sqrt(3)
    dSix = dSix * radon
    dFive = np.sqrt(6)
    dFive = dFive * radon #by radon
    dFive = dFive * asbestosTranspose # then by transpose of asbestos
    radonPowTwo = pow(radon, 2)
    asbPowTwo = pow(asbestos, 2)
    dEight = np.sqrt(3) 
    dEight = dEight * asbPowTwo 
    dThree = np.sqrt(3) 
    dThree = dThree * radonPowTwo
    dTwo = np.sqrt(3) 
    dTwo = dTwo * radonPowTwo # by power of two
    dTwo = dTwo * asbestosTranspose # then by transpose of asb
    dFour = np.sqrt(3)
    dFour = dFour * radonTranspose #by tranpose of rad
    dFour = dFour * asbPowTwo #then by power of 2
    asbPowThr = pow(asbestos, 3)
    radonPowThr = pow(radon, 3)
    return np.array([oneConv,dNine,dSix,dFive,dEight,dThree,dTwo,dFour,asbPowThr,radonPowThr])

def iterateSecStage(increment, rangeDeter):
    arr = []
    for i in rangeDeter:
        floatConv = float(i)
        multi = floatConv * increment
        arr.append(multi)
    return arr
if __name__ == "__main__":
    # Load training data
    d = np.load("lung_toy.npy")
    X = d[:,0:2]  # features
    y = d[:,2]  # labels

    # Show scatter-plot of the data
    idxsNeg = np.nonzero(y == -1)[0]
    idxsPos = np.nonzero(y == 1)[0]
    plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1])
    plt.scatter(X[idxsPos, 0], X[idxsPos, 1])
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,y1,160))
    plt.legend(["No lung disease", "Lung disease"])
    plt.show()

    # (a) Train linear SVM using sklearn
    svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmLinear.fit(X, y)
    showPredictions("Linear", svmLinear, X)

    # (b) Poly-3 using explicit transformation phiPoly3
    print("Poly-3 Using Explicit Transformation phiPoly3")
    phiPol = phiPoly3(X)
    phiPolTrans = np.transpose(phiPol)
    svmLinearPhi = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmLinearPhi.fit(phiPolTrans, y)
    showPredictions("Explicit Transformation", svmLinearPhi, phiPol)
    
    # (c) Poly-3 using kernel matrix constructed by kernel function kerPoly3
    print("Poly-3 using kernel matrix using kerPoly3")
    svmLinearKer = sklearn.svm.SVC(kernel=kerPoly3, C=0.01)
    svmLinearKer.fit(X, y)
    showPredictions("Kernel Matrix", svmLinearKer, X)
    
    # (d) Poly-3 using sklearn's built-in polynomial kernel
    print("Poly-3 using sklearn's built-in polynomial kernel")
    svmPolynomial = sklearn.svm.SVC(kernel='poly', C=0.01, gamma=1, coef0 = 1, degree = 3)
    svmPolynomial.fit(X, y)
    showPredictions("sklearn with built-in Poly Kernel", svmPolynomial, X)

    # (e) RBF using sklearn's built-in polynomial kernel
    print("RBF using sklearn's built-in polynomial kernel, gamma = 0.1")
    firstsvmwithRBF = sklearn.svm.SVC(kernel='rbf', C=1, gamma = 0.1)
    firstsvmwithRBF.fit(X, y)
    showPredictions("sklearn with built-in RBF gamma = 0.1", firstsvmwithRBF, X)
    print("RBF using sklearn's built-in polynomial kernel, gamma = 0.03")
    secondsvmwithRBF = sklearn.svm.SVC(kernel='rbf', C=1, gamma = 0.03)
    secondsvmwithRBF.fit(X, y)
    showPredictions("sklearn with built-in RBF gamma = 0.3", secondsvmwithRBF, X)
