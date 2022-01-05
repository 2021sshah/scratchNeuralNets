# Siddharth Shah
import time, sys, re, math

def manageSingleWeightFile(filename): # Single Weights List
    weightsLst = []
    weightLines = open(filename, "r").read().splitlines()
    for line in weightLines:
        if not containsDigits(line): continue
        line = line.replace(",", "")
        weights = [float(num) for num in line.split(" ") if num and isDouble(num)] # Remove Empty Strings
        weightsLst.append(weights)
    return weightsLst

def containsDigits(line):
    for char in line:
        if char.isdigit(): return True
    return False

def isDouble(num):
    acceptable = "-.0123456789"
    for char in num:
        if not char in acceptable: return False
    return True

def determineCombinedLayerCount(weightsLst):
    layerCount, nodeIdx = [2], 0
    for weightLayer in weightsLst:
        layerCount.append(len(weightLayer)//layerCount[nodeIdx])
        nodeIdx += 1
    combinedLayerCount = [3] + [2*num for num in layerCount[1:-1]] + [1,1]
    return layerCount, combinedLayerCount

def manageGivenInequality(inequality):
    numIdx = re.search(r"\d", inequality).start()
    inSign = inequality[inequality.rfind("y")+1:numIdx]
    rSquared = float(inequality[numIdx:])
    return inSign, rSquared**(1/2), rSquared

def createCombinedNetwork(singleWeightsLst, singleLayerCount): # Placing the Additional Zeros
    combinedWeightsLst = []
    # First Weight Layer
    xPart, yPart = insertZerosPerPart(singleWeightsLst[0], 1), insertZerosPerPart(singleWeightsLst[0], 0)
    combinedWeightsLst.append(xPart+yPart)
    # Middle Weight Layers
    respIdx = 1
    for weights in singleWeightsLst[1:-1]:
        groupSize = singleLayerCount[respIdx]
        weightGroups = splitWeightLayerIntoGroups(weights, groupSize)
        xPart = [sepGroup + [0]*groupSize for sepGroup in weightGroups]
        yPart = [[0]*groupSize + sepGroup for sepGroup in weightGroups]
        combinedWeightsLst.append(combineXYParts(xPart, yPart))
        respIdx += 1
    # Last Weight Layer
    finalWeight = singleWeightsLst[-1][0]
    combinedWeightsLst.append([finalWeight, finalWeight])
    # Final Combined Weight
    return combinedWeightsLst + [[(1+math.e)/(2*math.e)]]

def insertZerosPerPart(weightLayer, zeroIdx): # Specifically for Input Layer
    newLayer = []
    for idx in range(len(weightLayer)):
        if idx%2 == zeroIdx: newLayer.append(0)
        newLayer.append(weightLayer[idx])
    return newLayer

def splitWeightLayerIntoGroups(weightsLayer, groupSize):
    weightGroups, group = [], []
    for weight in weightsLayer:
        group.append(weight)
        if len(group) == groupSize: 
            weightGroups.append(group)
            group = []
    return weightGroups

def combineXYParts(xPart, yPart):
    return [weight for group in xPart for weight in group] + [weight for group in yPart for weight in group]

def adjustWeightsForCircle(weightsLst, rVal, inSign):
    weightsLst[0] = [weight/rVal for weight in weightsLst[0]]
    if inSign == "<": 
        weightsLst[-2] = [-weight for weight in weightsLst[-2]]
        weightsLst[-1] = [(1+math.e)/2]
    return weightsLst

def printNetwork(layerCount, weightsLst):
    print("Layer counts: {}".format(" ".join([str(num) for num in layerCount])))
    for weightLayer in weightsLst:
        print(" ".join([str(num) for num in weightLayer]))

startTime = time.time()
# Inputs and Initializations
singleWeightsLst = manageSingleWeightFile(sys.argv[1])
singleLayerCount, combinedLayerCount = determineCombinedLayerCount(singleWeightsLst)
inSign, rVal, rSquared = manageGivenInequality(sys.argv[2])
# Generate Combined Circle Network
combinedWeightsLst = adjustWeightsForCircle(createCombinedNetwork(singleWeightsLst, singleLayerCount), rVal, inSign)
# Print Results
printNetwork(combinedLayerCount, combinedWeightsLst)