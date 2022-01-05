# Siddharth Shah
import time, sys, random, math

def manageInputs(filename): # Reads through File
    inputsLst, outputLst = [], []
    testCases = open(filename, "r").read().splitlines()
    for case in testCases: # Place in data structures
        mid = case.index("=")
        inputsLst.append([int(num) for num in case[:mid-1].split(" ")] + [1]) # Offset/Bias of 1
        outputLst.append(int(case[mid+3]))
    return inputsLst, outputLst

def setWeightsToArrows(weightsLst, inputsLst): # Global Dictionary
    weightPosToArrow = {}
    prevLen = len(inputsLst)
    for i in range(len(weightsLst)):
        for j in range(len(weightsLst[i])):
            weightPosToArrow[(i,j)] = [(i,j%prevLen),(i+1,j//prevLen)] # Prev Layer, Next Layer
        prevLen = len(weightsLst[i])//prevLen # Update Prev Len
    return weightPosToArrow

def dot(lstA, lstB): # Dot Product of two Lists of Length A
    return sum([lstA[idx]*lstB[idx] for idx in range(len(lstA))])

def transfer(x): # Calculate Logistic Function
    return 1/(1+math.exp(-x))

def feedForward(weightsLayers, inputLayer): # From nnLab1
    networkCopy = []
    outputWeight = weightsLayers[-1]
    currentLayer = inputLayer
    for idx in range(len(weightsLayers[:-1])):
        networkCopy.append(currentLayer)
        weightsLst = weightsLayers[idx]
        nextLayer = []
        while weightsLst:
            nextLayer.append(transfer(dot(currentLayer, weightsLst)))
            weightsLst = weightsLst[len(currentLayer):]
        currentLayer = nextLayer # Move to Next Layer
    networkCopy.append(currentLayer)
    output = currentLayer[0]*outputWeight[0]
    networkCopy.append([output])
    return output, networkCopy

def totalError(expOutput, actOutput):
    return (1/2)*((expOutput-actOutput)**2)

def createPartialNetwork(weightsLst, expOutput, networkCopy):
    partialNetwork = [[], [], [], [expOutput-networkCopy[-1][0]]] # First Output Layer
    for i in range(len(networkCopy)-2,0,-1): # No Error for Inputs
        for j in range(len(networkCopy[i])): 
            derivative = networkCopy[i][j]*(1-networkCopy[i][j]) # For New Circle
            partialNetwork[i].append(backpropDot((i,j), weightsLst[i], partialNetwork[i+1])*derivative)
    return partialNetwork

def backpropDot(newCell, weightLayer, computedLayer):
    dotWeights, dotCells = [], []
    for y in range(len(weightLayer)):
        leftCell, rightCell = weightPosToArrow[newCell[0], y] # Choose the right weight
        if leftCell != newCell: continue
        dotWeights.append(weightLayer[y])
        dotCells.append(computedLayer[rightCell[1]]) # Use y position of right cell
    return dot(dotCells, dotWeights)

def adjustWeights(weightsLst, networkCopy, partialNetwork): # Calculated Partial from Packet
    for i in range(len(weightsLst)):
        for j in range(len(weightsLst[i])): # Adding alpha*(-partial)
            leftCell, rightCell = weightPosToArrow[(i,j)]
            if not i or i == len(weightsLst)-1: toShift = alpha*networkCopy[leftCell[0]][leftCell[1]]*partialNetwork[rightCell[0]][rightCell[1]]
            else: toShift = alpha*(partialNetwork[leftCell[0]][leftCell[1]]+partialNetwork[rightCell[0]][rightCell[1]])
            weightsLst[i][j] += toShift
    return weightsLst

def printNetwork(weightsLst):
    print("Layer counts: {} 2 1 1".format(len(inputsLst[0])))
    weightArray = []
    for weightLayer in weightsLst:
        weightArray.append(" ".join([str(num) for num in weightLayer]))
    print("{}\n{}\n{}".format(weightArray[0], weightArray[1], weightArray[2]))
    

startTime = time.time()
# Inputs and Initializations
inputsLst, expOutputLst = manageInputs(sys.argv[1]) # List of Lists, List of Ints
weightsLst = [[random.random() for i in range(2*len(inputsLst[0]))], [random.random(), random.random()], [random.random()]]
weightPosToArrow = setWeightsToArrows(weightsLst, inputsLst[0])
alpha = 3
# Tune while Time Remaining
itr = 0
while(time.time()-startTime < 28):
    # Feed Forward through Network
    inputs, expOutput = inputsLst[itr%len(inputsLst)], expOutputLst[itr%len(inputsLst)]
    tempOutput, tempNetworkCopy = feedForward(weightsLst, inputs) # Fix Choosing Input
    tempError = totalError(expOutput, tempOutput)
    # Calculate Partials and Adjust Weights
    modPartialNetwork = createPartialNetwork(weightsLst, expOutput, tempNetworkCopy)
    weightsLst = adjustWeights(weightsLst, tempNetworkCopy, modPartialNetwork)
    # Print Occasionally to show Progress
    itr += 1
    if not itr%2317:
        print("Exp:{} vs Act:{}".format(expOutput, tempOutput))
# Print Results
printNetwork(weightsLst)