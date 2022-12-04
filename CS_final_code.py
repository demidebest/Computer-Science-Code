import numpy as np
import json
import random
from sklearn.utils import resample
from statistics import mean

# Hyperparameters and input
path = "/Users/demidebest/Desktop/Computational Science/TVs-all-merged.json"
bootstrapIt = 5
numberMinhashing = 100
beta = 0.95


# Get training and test sample
def bootstrap(data, totalNumber, random):
    bootstrapSample = resample(data, replace=True, n_samples=totalNumber, random_state=random * 100)
    train = []
    test = []
    for element in data:
        if element in bootstrapSample:
            train.append(element)
        else:
            test.append(element)
    return train, test


# Get IDs
def getID(data):
    ID = []
    for element in data:
        ID.append(element["modelID"])
    return ID


# Get titles
def getTitle(data):
    title = []
    for element in data:
        title.append(element["title"])
    return title


# Get clean titles
def getCleanTitle(data):
    titles = []
    for element in data:
        title = element
        for symbol in ["(", ")", "[", "]", "-", "–", "'", '"', "/", ";", ":", "", ".", ","]:
            title = title.replace(symbol, "")
        title = title.replace("Neweggcom", "")
        title = title.replace("  ", " ")
        titles.append(title.lower())
    return titles


# Get clean feature
def getCleanFeature(string):
    title = string
    for symbol in ["(", ")", "[", "]", "-", "–", "'", '"', "/", ";", ":", "", ".", ","]:
        title = title.replace(symbol, "")
    title = title.replace("  ", " ")
    return title


# Get Shingles
def getShingles(data):
    allShingles = []
    for title in data:
        words = title.split(" ")
        for word in words:
            if len(word) > 2:
                if not (word.lower() in allShingles):
                    allShingles.append(word.lower())
    return allShingles


# Get signature matrix
def getMinhasing(allShingles, data, number):
    signatureMatrix = np.zeros((number, len(data)))
    for i in range(0, number):
        listIndex = list(range(0, len(allShingles)))
        random.seed(100000 + i)
        random.shuffle(listIndex)
        indexCounter = 0
        for column in data:
            counter = 0
            for index in listIndex:
                if allShingles[index] in column:
                    signatureMatrix[i][indexCounter] = counter
                    break
                else:
                    counter = counter + 1
            indexCounter = indexCounter + 1
    return signatureMatrix


# Get number of matches for all pairs
def getBuckets(signatureMatrix, number):
    pairs = np.zeros((len(signatureMatrix[0]), (len(signatureMatrix[0]))))
    for index1 in range(0, len(signatureMatrix[0])):
        for index2 in range(index1 + 1, len(signatureMatrix[0])):
            counterCheck = 0
            for i in range(0, number):
                if signatureMatrix[i][index1] == signatureMatrix[i][index2]:
                    counterCheck = counterCheck + 1
            pairs[index1][index2] = counterCheck
            pairs[index2][index1] = counterCheck
    return pairs


# Get candidate pairs
def getBucketsThreshold(pairsCheck, t):
    counter = 0
    pairs = np.zeros((len(pairsCheck[0]), (len(pairsCheck[0]))))
    for index1 in range(0, len(pairsCheck[0])):
        for index2 in range(index1 + 1, len(pairsCheck[0])):
            if pairsCheck[index1][index2] >= t:
                pairs[index1][index2] = 1
                pairs[index2][index1] = 1
                counter = counter + 1
    return counter, pairs


# Find brand name of each product
def findBrand(data, allData):
    brands = [0 for x in range(len(data))]
    brandsOptions = ['Insignia', 'Toshiba', 'Vizio', 'Dynex', 'Proscan', 'SunBriteTV', 'Philips', 'Samsung', 'TCL',
                     'AQUOS', 'Sony', 'LG', 'Panasonic', 'Coby', 'ViewSonic', 'Sharp', 'JVC', 'Optoma', 'NEC',
                     'Venturer', 'RCA', 'Westinghouse', 'Sansui', 'Hisense', 'Supersonic', 'Magnavox', 'Mitsubishi',
                     'Haier', 'CurtisYoung', 'Hannspree', 'Elite', 'Seiki', 'SIGMAC', 'Affinity', 'Pyle', 'GPX',
                     'Sceptre', 'Sanyo', 'Naxa', 'Epson', 'UpStar', 'Azend', 'Craig', 'HP', 'Hiteker', 'Contex',
                     'Kitty', 'Avue', 'Viore']
    for index in range(0, len(data)):
        words = data[index].split(" ")
        for word in words:
            if word in brandsOptions:
                brands[index] = brandsOptions[brandsOptions.index(word)]
                break
            if word in ['Insignia\x99', 'TOSHIBA', 'VIZIO', 'Dynex\x99', 'ProScan', 'SunBriteTV,']:
                brands[index] = brandsOptions[['Insignia\x99', 'TOSHIBA', 'VIZIO', 'Dynex\x99', 'ProScan',
                                               'SunBriteTV,'].index(word)]
                break
            if word in ['Vizio,']:
                brands[index] = brandsOptions[2]
        if brands[index] == 0:
            brands[index] = allData[index]['featuresMap']['Brand'].split()[0]
    return brands


# Get final pairs by comparing the website, brand and model ID for all candidate pairs
def checkWebsiteAndBrandAndID(candidates, allData, titleTrain, brand):
    optionsPerElementFinal = findModelID(allData, titleTrain)
    counter = 0
    pairs = np.zeros((len(candidates[0]), (len(candidates[0]))))
    for index1 in range(0, len(candidates[0])):
        for index2 in range(index1 + 1, len(candidates[0])):
            if candidates[index1][index2] == 1:
                if allData[index1]['shop'] != allData[index2]['shop']:
                    if brand[index1] == brand[index2]:
                        for word in optionsPerElementFinal[index1]:
                            if word in optionsPerElementFinal[index2]:
                                pairs[index1][index2] = 1
                                pairs[index2][index1] = 1
                                counter = counter + 1
                                break
    return counter, pairs


# Get (possible) model ID for all products
def findModelID(allData, titleTrain):
    allOptions = []
    allOptionsFinal = []
    optionsPerElement = []
    optionsPerElementFinal = []
    for i in range(0, len(titleTrain)):
        words = titleTrain[i].split(" ")
        options = []
        for word in words:
            if checkFeature(word):
                options.append(word)
                allOptions.append(word)
        for value in list(allData[i]['featuresMap'].values()):
            value1 = getCleanFeature(value)
            for word in value1:
                if checkFeature(word) and (word not in options):
                    options.append(word)
                    allOptions.append(word)
        optionsPerElement.append(options)

    for word in allOptions:
        if allOptions.count(word) <= 4:
            if len(word) >= 4:
                if word not in allOptionsFinal:
                    if word.upper() == word:
                        if 'INCH' not in word.upper():
                            if word.upper() not in ['50HZ', '60HZ', '120HZ', '240HZ', '600HZ']:
                                if word.upper() not in ['720P', '1080P']:
                                    allOptionsFinal.append(word)

    for i in range(0, len(optionsPerElement)):
        find = []
        for word in optionsPerElement[i]:
            if word in allOptionsFinal:
                find.append(word)
        optionsPerElementFinal.append(find)
    return optionsPerElementFinal


# Get threshold for training data
def getDuplicates(allData, data, signatureMatrix, number, titleTest, beta):
    t = -1
    check = False
    brand = findBrand(titleTest, allData)
    candidatesValues = getBuckets(signatureMatrix, number)
    fractionComp = []
    PairC = []
    PairQ = []
    f1 = []
    f1star = []
    for b in range(number, -1, -1):
        counter, candidatesFound = getBucketsThreshold(candidatesValues, b)
        P, TP, FP, FN = checkResults(candidatesFound, data)
        fractionComp.append((TP + FP) / ((len(candidatesFound) * (len(candidatesFound) - 1)) / 2))
        if (TP + FP) == 0 or ((TP / P) + (TP / (TP + FP))) == 0:
            f1.append(0)
        else:
            f1.append((2 * (TP / P) * (TP / (TP + FP))) / ((TP / P) + (TP / (TP + FP))))
        PairC.append(TP / P)
        PairQ.append(TP / (TP + FP))
        if not check:
            if TP/P >= beta:
                t = b
                check = True

        counter2, candidates2 = checkWebsiteAndBrandAndID(candidatesFound, allData, titleTest, brand)
        P2, TP2, FP2, FN2 = checkResults(candidates2, data)
        if counter2 == 0:
            f1star.append(0)
        elif (TP2 / P2) + (TP2 / counter2) > 0:
            f1star.append((2 * (TP2 / P2) * (TP2 / (TP2 + FP2))) / ((TP2 / P2) + (TP2 / (TP2 + FP2))))
        else:
            f1star.append(0)

    return fractionComp, PairC, PairQ, f1, f1star, t


# Get duplicates for test data
def getDuplicatesTest(allData, data, signatureMatrix, number, titleTest):
    brand = findBrand(titleTest, allData)
    candidatesValues = getBuckets(signatureMatrix, number)
    fractionComp = []
    PairC = []
    PairQ = []
    f1 = []
    f1star = []

    for b in range(number, -1, -1):
        counter, candidatesFound = getBucketsThreshold(candidatesValues, b)
        P, TP, FP, FN = checkResults(candidatesFound, data)
        fractionComp.append((TP + FP) / ((len(candidatesFound) * (len(candidatesFound) - 1)) / 2))
        if (TP + FP) == 0 or ((TP / P) + (TP / (TP + FP))) == 0:
            f1.append(0)
        else:
            f1.append((2 * (TP / P) * (TP / (TP + FP))) / ((TP / P) + (TP / (TP + FP))))
        PairC.append(TP / P)
        PairQ.append(TP / (TP + FP))

        counter2, candidates2 = checkWebsiteAndBrandAndID(candidatesFound, allData, titleTest, brand)
        P2, TP2, FP2, FN2 = checkResults(candidates2, data)
        if counter2 == 0:
            f1star.append(0)
        elif (TP2 / P2) + (TP2 / counter2) > 0:
            f1star.append((2 * (TP2 / P2) * (TP2 / (TP2 + FP2))) / ((TP2 / P2) + (TP2 / (TP2 + FP2))))
        else:
            f1star.append(0)
    return fractionComp, PairC, PairQ, f1, f1star


# Check whether feature consists of at least one number and one letter
def checkFeature(string):
    check1 = False
    check2 = False
    for sub in string:
        if sub.isalpha():
            check1 = True
        elif sub.isdigit():
            check2 = True
        if check1 and check2:
            break
    return check1 and check2


# Get TP, FP, TN and FN
def checkResults(data, modelID):
    P = 0
    TP = 0
    FP = 0
    for i in range(0, len(modelID)):
        for j in range(i + 1, len(modelID)):
            if modelID[i] == modelID[j]:
                P = P + 1
    for index1 in range(0, len(modelID)):
        for index2 in range(index1 + 1, len(modelID)):
            if data[index1][index2] == 1:
                if modelID[index1] == modelID[index2]:
                    TP = TP + 1
                else:
                    FP = FP + 1
    FN = P - TP
    return P, TP, FP, FN


# Get total number of pairs
def getNumberPairs(modelID):
    P = 0
    for i in range(0, len(modelID)):
        for j in range(i + 1, len(modelID)):
            if modelID[i] == modelID[j]:
                P = P + 1
    print('Total number of pairs: ', P)


#######################################################################


# Main method which evaluates 5 bootstrap samples
def main(path, bootstrapIt, numberMinhashing, beta):
    # Read file
    dataJ = json.load(open(path))
    data = []
    totalNumber = 0
    for key, value in dataJ.items():
        for element in value:
            data.append(element)
            totalNumber = totalNumber + 1
    modelIDComplete = getID(data)
    getNumberPairs(modelIDComplete)

    threshold = []
    f1starT = []
    fractionCompT = []
    fractionComp = []
    PairC = []
    PairQ = []
    f1 = []
    f1star = []

    fractionComp_train = []
    PairC_train = []
    PairQ_train = []
    f1_train = []
    f1star_train = []

    for i in range(0, bootstrapIt):
        sampleTrain, sampleTest = bootstrap(data, totalNumber, i + 10)

        # Training
        modelIDTrain = getID(sampleTrain)
        titleTrain = getTitle(sampleTrain)
        titleTrainClean = getCleanTitle(titleTrain)
        allShingles = getShingles(titleTrainClean)
        signatureMatrix = getMinhasing(allShingles, titleTrainClean, numberMinhashing)
        fractionCompTrain, PairCTrain, PairQTrain, f1Train, f1starTrain, b = getDuplicates(sampleTrain, modelIDTrain,
                                                                                       signatureMatrix,
                                                                                       numberMinhashing, titleTrain,
                                                                                      beta)

        threshold.append(b)
        if len(fractionComp_train) == 0:
            for l in range(0, len(fractionCompTrain)):
                fractionComp_train.append(fractionCompTrain[l] * (1 / bootstrapIt))
                PairC_train.append(PairCTrain[l] * (1 / bootstrapIt))
                PairQ_train.append(PairQTrain[l] * (1 / bootstrapIt))
                f1_train.append(f1Train[l] * (1 / bootstrapIt))
                f1star_train.append(f1starTrain[l] * (1 / bootstrapIt))
        else:
            for l in range(0, len(fractionCompTrain)):
                fractionComp_train[l] = fractionComp_train[l] + fractionCompTrain[l]*(1/bootstrapIt)
                PairC_train[l] = PairC_train[l] + PairCTrain[l]*(1/bootstrapIt)
                PairQ_train[l] = PairQ_train[l] + PairQTrain[l]*(1/bootstrapIt)
                f1_train[l] = f1_train[l] + f1Train[l]*(1/bootstrapIt)
                f1star_train[l] = f1star_train[l] + f1starTrain[l]*(1/bootstrapIt)

        # Testing
        modelIDTest = getID(sampleTest)
        titleTest = getTitle(sampleTest)
        titleTestClean = getCleanTitle(titleTest)
        allShinglesTest = getShingles(titleTestClean)
        signatureMatrixTest = getMinhasing(allShinglesTest, titleTestClean, numberMinhashing)
        fractionCompTest, PairCTest, PairQTest, f1Test, f1starTest = getDuplicatesTest(sampleTest, modelIDTest,
                                                                                       signatureMatrixTest,
                                                                                       numberMinhashing, titleTest)

        if len(fractionComp) == 0:
            for l in range(0, len(fractionCompTest)):
                fractionComp.append(fractionCompTest[l] * (1 / bootstrapIt))
                PairC.append(PairCTest[l] * (1 / bootstrapIt))
                PairQ.append(PairQTest[l] * (1 / bootstrapIt))
                f1.append(f1Test[l] * (1 / bootstrapIt))
                f1star.append(f1starTest[l] * (1 / bootstrapIt))
        else:
            for l in range(0, len(fractionComp)):
                fractionComp[l] = fractionComp[l] + fractionCompTest[l]*(1/bootstrapIt)
                PairC[l] = PairC[l] + PairCTest[l]*(1/bootstrapIt)
                PairQ[l] = PairQ[l] + PairQTest[l]*(1/bootstrapIt)
                f1[l] = f1[l] + f1Test[l]*(1/bootstrapIt)
                f1star[l] = f1star[l] + f1starTest[l]*(1/bootstrapIt)
        f1starT.append(f1starTest[numberMinhashing - threshold[i]])
        fractionCompT.append(fractionCompTest[numberMinhashing - threshold[i]])

    with open('fractionComp_train.txt', 'w') as f:
        for line in fractionComp_train:
            f.write(f"{line}\n")

    with open('PairC_train.txt', 'w') as f:
        for line in PairC_train:
            f.write(f"{line}\n")

    with open('PairQ_train.txt', 'w') as f:
        for line in PairQ_train:
            f.write(f"{line}\n")

    with open('f1_train.txt', 'w') as f:
        for line in f1_train:
            f.write(f"{line}\n")

    with open('f1star_train.txt', 'w') as f:
        for line in f1star_train:
            f.write(f"{line}\n")

    print('Results of training (i.e. selected threshold, f1* test, fraction comparisons test): ', threshold, mean(f1starT), mean(fractionCompT))

    with open('fractionComp.txt', 'w') as f:
        for line in fractionComp:
            f.write(f"{line}\n")

    with open('PairC.txt', 'w') as f:
        for line in PairC:
            f.write(f"{line}\n")

    with open('PairQ.txt', 'w') as f:
        for line in PairQ:
            f.write(f"{line}\n")

    with open('f1.txt', 'w') as f:
        for line in f1:
            f.write(f"{line}\n")

    with open('f1star.txt', 'w') as f:
        for line in f1star:
            f.write(f"{line}\n")


main(path, bootstrapIt, numberMinhashing, beta)
