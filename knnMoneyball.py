#KNN on moneyball data
# Luis Solorzano
from time import clock
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np



#runs knn with ball tree algorithm and fits the training data
def knn(train,trainLabel,test,k):
    time1 = clock()
    neigh = KNeighborsClassifier(n_neighbors=1, algorithm=algo,metric = mesureP,p=2, n_jobs=-1)
    neigh.fit(train, trainLabel)
    predictLabel = neigh.predict(test)
    time3 = clock()
    print("time:",(time3-time1))
    predictionError = np.count_nonzero(predictLabel != trainTestLabel)/float(len(trainTestLabel))
    print("Prediction Error:",predictionError)


def main():
    print()
    





if __name__== '__main__':
    main()
