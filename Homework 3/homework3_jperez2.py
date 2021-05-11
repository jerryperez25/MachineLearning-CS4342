import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    epochs = 600
    W = np.random.rand(785,10)
    epochRange = range(epochs)
    for i in epochRange:  # Number of epochs
        secondRange = range(0, trainingLabels.shape[1], batchSize)
        for j in secondRange:
            minCalc = min(j + batchSize, trainingLabels.shape[1] - 1)
            end = minCalc
            imageSpecifier = trainingImages[:, j:end]
            Yhat = calulateNormal(W, imageSpecifier)
            labelSpecifier = trainingLabels[:, j:end]
            transp = np.transpose(Yhat - labelSpecifier)
            dotProd = np.dot(imageSpecifier, transp)
            grad = dotProd / (end - j)
            W = W - (epsilon * grad)
            trainChecker = trainingLabels.shape[1] - j
            if (trainChecker/batchSize) <= 20 and i == epochs - 1:  # If this is in the last 20 iterations
                calc = calculateCE(trainingImages, W, trainingLabels)
                stringify = str(calc)
                print('Cross entropy loss:',stringify)
                calc = calculateAccuracy(trainingImages, W, trainingLabels)
                stringify = str(calc)
                print('Percent accuracy: ',stringify)
    return W


def calulateNormal(W, imageSpecifier):
    transp = np.transpose(W)
    norm = np.dot(transp, imageSpecifier)
    norm = np.exp(norm)
    normSum = np.sum(norm, axis=0)
    norm = norm / normSum
    return norm


def calculateAccuracy(trainingImages, W, trainingLabels):
    norm = calulateNormal(W, trainingImages)
    guess = np.argmax(norm, axis=0)
    truth = np.argmax(trainingLabels, axis=0)
    acc = guess - truth
    acc[acc != 0] = 1
    return 1 - (acc.sum() / acc.shape[0])


def calculateCE(trainingImages, W, trainingLabels):
    norm = calulateNormal(W, trainingImages)
    loss = trainingLabels * np.log(norm) 
    lossCalc = loss.sum() / trainingLabels.shape[1]
    return 0-lossCalc


def reshapeAndAppend1s(image):
    ones = np.ones((image.shape[0], 1))
    image = np.hstack((image, ones))
    image = image.T
    return image


if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy")
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy")
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    # ...
    trainingImages = reshapeAndAppend1s(trainingImages)
    testingImages = reshapeAndAppend1s(testingImages)
    trainingLabels = np.transpose(trainingLabels)
    testingLabels = np.transpose(testingLabels)

    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.5, batchSize=100)
    print("Training Cross Entropy:", calculateCE(trainingImages, W, trainingLabels))
    print("Testing Cross Entropy:", calculateCE(testingImages, W, testingLabels))
    print("Training Percent Correct Accuracy:", calculateAccuracy(trainingImages, W, trainingLabels))
    print("Testing Percent Correct Accuracy:", calculateAccuracy(testingImages, W, testingLabels))

    # Visualize the vectors
    # ...
