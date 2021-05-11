import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time

def fPC (y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors (predictors, X, y):
    zero_matrix = np.zeros(y.shape, dtype = y.dtype)
    predictor_count = 0
    incrementer = 0
    #We have to run the ensemble for every image in the set
    for _ in predictors:
        predictor_count = predictor_count + 1
        row1,column1,row2,column2 = _
        new_vote = X[:,row1,column1] - X[:,row2,column2]
        new_vote[new_vote > 0] = 1
        new_vote[new_vote <= 0] = 0
        zero_matrix = zero_matrix + new_vote    
    divided_matrix = np.divide(zero_matrix,len(predictors))
    divided_matrix[divided_matrix > 0] = 1
    divided_matrix[divided_matrix <= 0] = 0
    yhat = divided_matrix

    return fPC(y, yhat) #returns accuracy
    
def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    features = 5
    range_setter = range(0,24)
    best_feature = None
    best_predictor = None
    target_score = []
    predictors = []
    for _ in range(features):
        best_feature = None
        best_predictor = None
        target_score = 0.0
        for row1 in range_setter:
            for column1 in range_setter:
                for row2 in range_setter:
                    for column2 in range_setter:
                        current_location = (row1,column1,row2,column2)
                        curr_list = list((current_location,))
                        current_score = measureAccuracyOfPredictors(predictors + curr_list, trainingFaces,trainingLabels)
                        if current_score >= target_score:
                            best_feature = current_location
                            target_score = current_score
    
        predictors.append(best_feature)
            
    return predictors


def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

def showFeatures(predictors, testingFaces):
    im = testingFaces[0,:,:]
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    #this will show r1,c1
    for _ in predictors:
        r1,c1,r2,c2 = _
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def createOutput(trainingFaces, trainingLabels, testingFaces, testingLabels):
    #Sample sizes
    start_time = time.time()
    sample_n = [400,800,1200,1600,2000]
    predictors = None
    print("n    trainingAccuracy    testingAccuracy")
    for _ in sample_n:
        #Which predictors where used
        predictors = stepwiseRegression(trainingFaces[:_], trainingLabels[:_], testingFaces, testingLabels)
        #Training Accuracy
        train_accuracy = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)
        #Testing Accuracy
        test_accuracy = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
        print("{}   {}  {}".format(_,train_accuracy, test_accuracy))
    #Show the face
    showFeatures(predictors, trainingFaces)
    #calculate the time it took to run 
    minute_time = ((time.time() - start_time)/60)
    print("It takes about %s minutes to finish execution" % minute_time)



if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    createOutput(trainingFaces, trainingLabels, testingFaces, testingLabels)
    
