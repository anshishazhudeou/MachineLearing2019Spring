from numpy import linalg as LA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import csv

# 10 Data Blocks
data = []
# 10 Label Blocks
labels = []

for i in range(10):
    # initialize these
    dataFileName = '/Documents/2019Spring/cs480/A1/q1/knn-dataset/trainData' + str(
        i + 1) + ".csv"
    labelsFileName = '/Documents/2019Spring/cs480/A1/q1/knn-dataset/trainLabels' + str(
        i + 1) + '.csv'
    # we dont have header in data, readcsv needs to read the first row of data
    data.append(pd.read_csv(dataFileName, header=None).values)
    labels.append(pd.read_csv(labelsFileName, header=None).values)

    (a, b) = data[i].shape
    (c, d) = labels[i].shape
    assert ((a == 100) and (c == 100) and (b == 64) and (d == 1))


def Knn(trainData, trainLabels, testData, testLabels, k):
    # (m,n) = trainData.shape

    (m, n) = trainData.shape
    # (p,q) = trainLabels.shape
    (p, q) = trainLabels.shape
    # (a,b) = testData.shape
    (a, b) = testData.shape
    # (c,d) = testLabels.shape
    (c, d) = testLabels.shape
    assert ((m == p) and (m == 900));
    assert ((a == c) and (a == 100));

    assert ((n == b) and (n == 64));
    assert ((q == d) and (q == 1));

    totalTestNumber = a
    totaldataNumber = m
    correctNumber = 0
    incorrectNumber = 0

    # for each test
    for i in range(totalTestNumber):
        # create an empty nparray
        e = np.zeros((1, 999))

        # for each training:, eae[1,j]ch row of training set
        for j in range(totaldataNumber):
            e[0, j] = LA.norm(trainData[j, :] - testData[i, :])

        index = np.argsort(e)
        sortedE = np.sort(e)

        numofTimesofFive = 0
        numofTimesofSix = 0

        # total k-nearst neigbours
        for i in range(1,k+1):
            indexPicked = index[0, i]  # first group number
            numMatched = labels[(int(indexPicked) + 1) // 100][(int(indexPicked) % 100)]

            assert ((numMatched == 5) or (numMatched == 6))
            if (numMatched == 5):
                numofTimesofFive += 1
            else:
                numofTimesofSix += 1

        # hypothesis prediction
        if (numofTimesofFive < numofTimesofSix):
            correctValue = 6;
        else:
            correctValue = 5;

        # underlying true value
        trueValue = testLabels[i, 0]
        assert ((trueValue == 5) or (trueValue == 6))

        if (correctValue == trueValue):
            correctNumber += 1
        else:
            incorrectNumber += 1
        accuracy = correctNumber / (correctNumber + incorrectNumber)

    return accuracy


# it returns the average accuracy of 10-crossfold validation with a particaluar
def crossValidation(data, labels, k):
    accumulativeAccuracy = 0;

    # split data and labelscrossValidation in ten different ways
    for i in range(10):
        counter = 0
        for j in range(10):
            if (i != j):
                if (counter == 0):
                    # first time
                    dataExpand = data[j]
                    labelsExpand = labels[j]
                    counter += 1
                else:
                    dataExpand = np.vstack((dataExpand, data[j]))
                    labelsExpand = np.vstack((labelsExpand, labels[j]))

        testData = data[i]
        testLabels = labels[i]
        accurancy = Knn(dataExpand, labelsExpand, testData, testLabels, k)
        accumulativeAccuracy += accurancy
    averageAccuracy = accumulativeAccuracy / 10
    return averageAccuracy


# check each possible k-nearst neighbour
x = []  # used to store k
y = []  # used to accuracy with corresponding k
for k in range(1, 31):
    x.append(k)
    y.append(crossValidation(data, labels, k))

plt.figure()
plt.plot(x, y)
print
plt.xlabel("k")
plt.ylabel("accuracy")
plt.title("Accuracy vs k")
plt.show()


