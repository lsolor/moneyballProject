Luis Solorzano, Isabel

##########################################
BACKGROUND:
This is a machine learning project that is based off of the popular sports movement, Sabermetrics, which was brought to fame by the film
Moneyball (2011). The goal of this project was to see if we could predict a player's WAR (wins above replacement) using their statline. We
separated by pitchers and batters. With pitchers we were only concerned with their pitching ability. With batters we were only concerned with
their offensive production. Using the sci-kit python module we pulled data from baseball reference and feed them into our preprocessing
algorithm which created a dictionary where the key was a player's id and their statline as the value. We then had to transform our data
structures into numpy arrays which is how sci-kit took its inputs. The data was then ran on multiple machine learning algorithms: bayesien
learning, k-nearest neighbors, and neural networks.

##########################################
DATASET:
All datasets were pulled from baseball-reference.com
We took batting and pitching stats from the regular season from 1997-2001
We also took the freeagents' stats for those years as well. 


##########################################
METHODS:
Bayesien learning:

k-Nearest Neighbors:

Neural Networks:
Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general 
internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors
of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.


##########################################
OUTPUT:
For a set of instances we predict WAR(wins above replacement) for each machine learning algorithm with different parameters. We then compare those 
WARs to the actual WAR.

##########################################
FUTURE PLANS:

I have recently been learning R, so I hope to graph the output that we have created in order to tell a more meaningful story of the data.
