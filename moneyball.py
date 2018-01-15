#moneyball.py
#Ryan, Isabel and Luis

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor as mlpr
from sklearn import linear_model
from knnMoneyball import knn
import numpy as np




########### FILES TO USE ###########

pitcherTraining = "pitcherFiles.txt"
batterTraining = "batterFiles.txt"
pitcherWARS = "pitcherWARS.txt"
batterWARS = "batterWARS.txt"
pitcherTesting = "2001_freeagentPitchingStats.csv"
batterTesting = "2001_freeagentBattingStats.csv"

outputFileB = open("moneyball_results_B.csv", "w+")
outputFileP = open("moneyball_results_P.csv", "w+")

 


########### GLOBAL VARS ###########

kList = [5, 10, 15] #use for knn
nList = [25, 50, 100] #use for nn
aList = [0.01, 0.000001, 0.00000000001] #use for bays

trainingStorageB = {}
trainingStorageP = {}
testInst = []




########### DEBUGGING VARIABLES ###########

printPred = False #prints data once instances are read in




########### CLASSES FOR INSTANCES ###########

class batter:
    def __init__(self, name, idd, year, pos):
        self.name = name
        self.pos = pos
        self.id = idd
        self.year = year
        self.data = []
        self.batter = True
        self.pitcher = False
        self.nextYear = self
        self.WAR = 0.0

    def setWAR(self, WAR):
        self.WAR = WAR

class pitcher:
    def __init__(self, name, idd, year):
        self.name = name
        self.pos = "Pitcher"
        self.id = idd
        self.year = year
        self.data = []
        self.batter = False
        self.pitcher = True
        self.nextYear = self
        self.WAR = 0.0

    def setWAR(self, WAR):
        self.WAR = WAR




########### FUNCTIONS THAT READ IN DATA ###########
        
def readInTraining(pitchers, batters):
#takes lists of files with data and creates yearly dictionaries
#which are stored in trainingStorage so we can pull out test set
    
    global trainingStorageB
    global trainingStorageP

    trainInst = []
    
    batterFiles = open(batters, "r")
    pitcherFiles = open(pitchers, "r")

    for f in batterFiles: #loop through files with batter data
        f = f.strip()
        year = eval(f.split("_")[0])
        file = open(f, "r")
        trainingStorageB[year] = {}

        linecount = 0
        for line in file: #each line is a player+their stats
            if linecount == 0: 
                linecount += 1
                continue #ignore first line
            line = line.strip()
            line = line.split(",") #now line is an array of data
            nameList = line[0].split("\\")
            name = nameList[0]
            ID = nameList[1]
            if ID in trainingStorageB: #only want one instance of each player per year
                continue
            else:
                line[0] = ID
                line.insert(0,name.strip("#"))
                inst = batter(name, ID, year, line[2])
                for i in range(3, len(line)): #dont add position
                    try:
                        inst.data.append(float(line[i]))
                    except:
                        pass
                if len(inst.data) == 12:
                    trainingStorageB[year][ID] = inst
                    trainInst.append(inst)

        file.close()

    for f in pitcherFiles: #loop through files with pitcher data
        f = f.strip()
        year = eval(f.split("_")[0])
        file = open(f, "r")

        trainingStorageP[year] = {}

        linecount = 0
        for line in file: #each line is a player+their stats
            if linecount == 0: 
                linecount += 1
                continue #ignore first line
            line = line.strip()
            line = line.split(",") #now line is an array of data
            nameList = line[0].split("\\")
            name = nameList[0]
            ID = nameList[1]
            if ID in trainingStorageP: #only want one instance of each player per year
                continue
            else:
                line[0] = ID
                line.insert(0,name.strip("#"))
                inst = pitcher(name, ID, year)

                for i in range(2, len(line)): #dont add position
                    try:
                        inst.data.append(float(line[i]))
                    except:
                        pass
                if len(inst.data) == 11:
                    trainingStorageP[year][ID] = inst
                    trainInst.append(inst)
        file.close()

    for i in range(0, len(trainInst)): #chains instances
        for j in range(i+1, len(trainInst)):
            if trainInst[i].id == trainInst[j].id and trainInst[i].year+1 == trainInst[j].year:
                trainInst[i].nextYear = trainInst[j]
                break


def makeInstances():
#turns whats left in storage after taking out testing into training numpy arrays
#also strings these instances together
    
    trainingDataB = []
    trainingLabelsB = []
    trainingDataP = []
    trainingLabelsP = []

    for year in trainingStorageB:
        for ID in trainingStorageB[year]:
            inst = trainingStorageB[year][ID] 
            try: #gets rid of weird strings that aren't floats
                trainingLabelsB.append([float(inst.WAR)])
                trainingDataB.append(inst.data)
            except:
                continue

    for year in trainingStorageP:
        for ID in trainingStorageP[year]:
            inst = trainingStorageP[year][ID]
            try:
                trainingLabelsP.append([float(inst.WAR)])
                trainingDataP.append(inst.data)
            except:
                continue
            
    np1 = np.asarray(trainingDataB, dtype=np.float32)
    np2 = np.asarray(trainingLabelsB, dtype=np.float32)
    np3 = np.asarray(trainingDataP, dtype=np.float32)
    np4 = np.asarray(trainingLabelsP, dtype=np.float32)
    return np1, np2, np3, np4

    
def readInTest(pitchers, batters):
#reads in data for free agents
#if agent has a 2001 instance already pull it out of the training set and put into test
#otherwise, use stats from this file and make new instance for the test set

    global trainingStorageB
    global trainingStorageP
    global testInst

    testDataB = []
    testLabelsB = []
    testDataP = []
    testLabelsP = []

    year = 2001
    batterFile = open(batters, "r")
    pitcherFile = open(pitchers, "r")

    outputFileB.write('NAME,')
    outputFileP.write('NAME,')

    linecount = 0
    for line in batterFile:
        if linecount == 0: 
            linecount += 1
            continue #ignore first line
        line = line.strip()
        line = line.split(",") #now line is an array of data
        nameList = line[0].split("\\")
        name = nameList[0]
        outputFileB.write(name)
        outputFileB.write(',')
        ID = nameList[1]
        if ID in trainingStorageB[year]:
            inst = trainingStorageB[year][ID]
            testInst.append(inst)
            testDataB.append(inst.data)
            testLabelsB.append([inst.WAR])
            del trainingStorageB[year][ID]
        else:
            inst = batter(name, ID, year, line[len(line)-1])
            for i in range(1, len(line)-2): #dont add position
                try:
                    inst.data.append(float(line[i]))
                except:
                    pass
            inst.setWAR(line[len(line)-2])
            testInst.append(inst)
            testDataB.append(inst.data)
            testLabelsB.append([inst.WAR])

    linecount = 0
    for line in pitcherFile:
        if linecount == 0: 
            linecount += 1
            continue #ignore first line
        line = line.strip()
        line = line.split(",") #now line is an array of data
        nameList = line[0].split("\\")
        name = nameList[0]
        outputFileP.write(name)
        outputFileP.write(',')
        ID = nameList[1]
        if ID in trainingStorageP[year]:
            inst = trainingStorageP[year][ID]
            testInst.append(inst)
            testDataP.append(inst.data)
            testLabelsP.append([inst.WAR])
            del trainingStorageP[year][ID]
        else:
            inst = pitcher(name, ID, year)            
            for i in range(1, len(line)-1): #dont add position
                try:
                    inst.data.append(float(line[i]))
                except:
                    pass
            inst.setWAR(line[len(line)-1])
            testInst.append(inst)
            testDataP.append(inst.data)
            testLabelsP.append([inst.WAR])

    outputFileB.write('\n'+"ACTUAL,")
    outputFileP.write('\n'+"ACTUAL,")

    for b in testLabelsB:
        outputFileB.write(b[0] + ",")

    for p in testLabelsP:
        outputFileP.write(p[0] + ",")


    np1 = np.asarray(testDataB, dtype=np.float32)
    np2 = np.asarray(testLabelsB, dtype=np.float32)
    np3 = np.asarray(testDataP, dtype=np.float32)
    np4 = np.asarray(testLabelsP, dtype=np.float32)

    return np1, np2, np3, np4


def setLabels(pitcherWARS, batterWARS):
#takes files of names, years and wars and sets label of correct instance
#this is because we had to get the wars from a different place and Luis is bad at copy paste

    global trainingStorageB
    global trainingStorageP
    
    batterFiles = open(batterWARS, "r")
    pitcherFiles = open(pitcherWARS, "r")

    for f in batterFiles: #loop through files with batter data
        f = f.strip()
        year = eval(f.split("_")[0])
        file = open(f, "r")

        linecount = 0
        for line in file:
            if linecount == 0: 
                linecount += 1
                continue #ignore first line
            
            line = line.strip()
            line = line.split("\\") #now line is an array of data
            if line[0] == '' or line[0] == ',':
                continue
            nameList = line[1].split(",")
            name = line[0]
            ID = nameList[0]
            WAR = nameList[1]

            if ID in trainingStorageB[year]:
                trainingStorageB[year][ID].setWAR(WAR)

        file.close()

    for f in pitcherFiles: #loop through files with pitcher data
        f = f.strip()
        year = eval(f.split("_")[0])
        file = open(f, "r")


        linecount = 0
        for line in file:
            if linecount == 0: 
                linecount += 1
                continue #ignore first line

            line = line.strip()
            line = line.split("\\") #now line is an array of data
            if line[0] == '' or line[0] == ",":
                continue
            nameList = line[1].split(",")
            name = line[0]
            ID = nameList[0]
            WAR = nameList[1]

            if ID in trainingStorageP[year]:
                trainingStorageP[year][ID].setWAR(WAR)          
            
        file.close()




########### FUNCTIONS THAT TRAIN MODELS ON DATA ###########

def knn(k, training, labels, test, testlabels):
#labels test set based on training set using knn
    neigh = KNeighborsRegressor(n_neighbors=k)
 #   print("training", training)
  #  print()
   # print("labels", labels)
    neigh.fit(training, labels)
    predicted = neigh.predict(test)
    if printPred:
        print()
        print("knn\n", predicted)
        
    return predicted

def nn(n, training, labels, test, testlabels):
#labels test set
   # print(training)
    #print()
   # print(labels)
    network = mlpr(n)
    nn = network.fit(training, labels)
    predicted = nn.predict(test)
    if printPred:
        print()
        print("nn\n", predicted)

    return predicted

def bays(a, training, labels, test, testlabels):
#labels test set
   # print(training)
    #print()
   # print(labels)
    bays = linear_model.BayesianRidge(alpha_1=a)
    bays.fit(training, labels)
    predicted = bays.predict(test)
    if printPred:
        print()
        print("bays\n", predicted)

    return predicted



###################### MAIN ######################
    
def main():
#creates instances for all players in training set, sets their labels
#then reads in test set (labels in file)
    
    readInTraining(pitcherTraining, batterTraining) #fill storage

    setLabels(pitcherWARS, batterWARS) #call first so order is there

    testDataB, testLabelsB, testDataP, testLabelsP = readInTest(pitcherTesting, batterTesting) #finds new players, pulls old ones out of storage

    trainingDataB, trainingLabelsB, trainingDataP, trainingLabelsP = makeInstances() #turn whats left in storage into training sets

   # print(trainingDataP)
   # print(trainingLabelsP)
   # print()

    for k in kList:
        knnOutputB = knn(k, trainingDataB, trainingLabelsB, testDataB, testLabelsB)
        knnOutputP = knn(k, trainingDataP, trainingLabelsP, testDataP, testLabelsP)

        outputFileB.write('\n'+"KNN"+str(k)+',')
        outputFileP.write('\n'+"KNN"+str(k)+',')

        for b in knnOutputB:
            outputFileB.write(str(b[0]) + ",")

        for p in knnOutputP:
            outputFileP.write(str(p[0]) + ",")

    for n in nList:
        nnOutputB = nn(n, trainingDataB, trainingLabelsB, testDataB, testLabelsB)
        nnOutputP = nn(n, trainingDataP, trainingLabelsP, testDataP, testLabelsP)

        outputFileB.write('\n'+"NN"+str(n)+',')
        outputFileP.write('\n'+"NN"+str(n)+',')

        for b in nnOutputB:
            outputFileB.write(str(b) + ",")

        for p in nnOutputP:
            outputFileP.write(str(p) + ",")

    for a in aList:
        baysOutputB = bays(a, trainingDataB, trainingLabelsB, testDataB, testLabelsB)
        baysOutputP = bays(a, trainingDataP, trainingLabelsP, testDataP, testLabelsP)

        outputFileB.write('\n'+"BAY"+str(a)+',')
        outputFileP.write('\n'+"BAY"+str(a)+',')

        for b in baysOutputB:
            outputFileB.write(str(b) + ",")

        for p in baysOutputP:
            outputFileP.write(str(p) + ",")

  #  Pdata = [testDataP, knnOutputP, nnOutputP, baysOutputP]   
   # Bdata = [testDataB, knnOutputB, nnOutputB, baysOutputB]



if __name__== '__main__':
    main()
