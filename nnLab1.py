# Siddharth Shah
import sys, math

def manageWeights(filename): # Weights are Doubles
    weightsLayers = []
    layers = open(filename, "r").read().splitlines()
    for layer in layers:
        weightsLayers.append([float(weight) for weight in layer.split(" ")])
    return weightsLayers

def dot(lstA, lstB): # Dot Product of two Lists of Length A
    return sum([lstA[idx]*lstB[idx] for idx in range(len(lstA))])

# Transfer Functions
def linear(x):
    return x

def ramp(x):
    return x if x > 0 else 0

def logistic(x):
    return 1/(1+math.exp(-x))

def sigmoid(x):
    return 2/(1+math.exp(-x)) - 1

def transfer(x):
    if transferNum == 1: return linear(x)
    if transferNum == 2: return ramp(x)
    if transferNum == 3: return logistic(x)
    return sigmoid(x)

# Solve Function
def process(weightsLayers, inputLayer):
    outputWeights = weightsLayers[-1]
    currentLayer = inputLayer
    for weights in weightsLayers[:-1]:
        weightsLst = weights
        nextLayer = []
        while weightsLst:
            nextLayer.append(transfer(dot(currentLayer, weightsLst)))
            weightsLst = weightsLst[len(currentLayer):]
        currentLayer = nextLayer # Move to Next Input Layer
    for idx in range(len(currentLayer)):
        print(currentLayer[idx]*outputWeights[idx], end = " ") # Displey Output in Terminal

# Inputs and Initializations
weightsLayers = manageWeights(sys.argv[1]) # Weights Layer by Layer
transferNum = int(sys.argv[2][1]) # Identify by Number
inputLayer = [float(num) for num in sys.argv[3:]] # Converted to Doubles
# Solve Routine
process(weightsLayers, inputLayer)