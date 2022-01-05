# Siddharth Shah
import time, sys, re, random, math

def manageInputs(inequality):
    numIdx = re.search(r"\d", inequality).start()
    inSign = inequality[inequality.rfind("y")+1:numIdx]
    rSquared = float(inequality[numIdx:])
    return inSign, rSquared

def setWeightsToArrows(weightsLst, inputsLst): # Global Dictionary
    weightPosToArrow = {}
    prevLen = len(inputsLst)
    for i in range(len(weightsLst)):
        for j in range(len(weightsLst[i])):
            weightPosToArrow[(i,j)] = [(i,j%prevLen),(i+1,j//prevLen)] # Prev Layer, Next Layer
        prevLen = len(weightsLst[i])//prevLen # Update Prev Len
    return weightPosToArrow

def createTestCase(inSign, rSquared):
    x, y = random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)
    if inSign == "<": return [x, y, 1], int(x**2 + y**2 < rSquared)
    if inSign == "<=": return [x, y, 1], int(x**2 + y**2 <= rSquared)
    if inSign == ">": return [x, y, 1], int(x**2 + y**2 > rSquared)
    return [x, y, 1], int(x**2 + y**2 >= rSquared)

# Neural Network Feed Forward
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
    partialNetwork = [[], [], [], [], [expOutput-networkCopy[-1][0]]] # First Comes Output Layer
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

def adjustAlpha(timeElapsed): # Linear Decrease of Alpha
    return -0.01*timeElapsed + 1

def printNetwork(weightsLst):
    print("Layer counts: 3 8 3 1 1")
    weightArray = []
    for weightLayer in weightsLst:
        weightArray.append(" ".join([str(num) for num in weightLayer]))
    print("{}\n{}\n{}\n{}".format(weightArray[0], weightArray[1], weightArray[2], weightArray[3]))

startTime = time.time()
# Inputs and Initializations
inSign, rSquared = manageInputs(sys.argv[1]) # Layer Counts: 3 8 3 1 1
weightsLst = [[random.uniform(-1.0, 1.0) for i in range(24)], [random.uniform(-1.0, 1.0) for i in range(24)], [random.uniform(-1.0, 1.0) for i in range(3)], [random.uniform(1.0, 2.0)]]
weightPosToArrow = setWeightsToArrows(weightsLst, [1,1,1]) # 2nd Arg is Sample Input List
alpha = 1
# Tune while Time Remaining
while(time.time()-startTime < 98):
    inputs, expOutput = createTestCase(inSign, rSquared)
    tempOutput, tempNetworkCopy = feedForward(weightsLst, inputs) # Fix Choosing Input
    tempError = totalError(expOutput, tempOutput)
    # Calculate Partials and Adjust Weights
    modPartialNetwork = createPartialNetwork(weightsLst, expOutput, tempNetworkCopy)
    weightsLst = adjustWeights(weightsLst, tempNetworkCopy, modPartialNetwork)
    # Adjust Alpha Based on Time Elapsed
    aplha = adjustAlpha(time.time()-startTime)
# Print Results
printNetwork(weightsLst)