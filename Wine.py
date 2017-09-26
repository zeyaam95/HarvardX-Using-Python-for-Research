import pandas as pd
import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import scipy.stats
from sklearn.neighbors import KNeighborsClassifier
# from matplotlib.backends.backend_pdf import PdfPages


def distance(p1, p2):
    """Find the distance between two points."""
    return np.sqrt(np.sum(pow(p1 - p2, 2)))


def majority_vote(votes):
    """Returns the vote count in a list."""
    vote_count = {}
    for vote in votes:
        if vote in vote_count:
            vote_count[vote] += 1
        else:
            vote_count[vote] = 1
    winners = []
    max_count = max(vote_count.values())
    for vote, count in vote_count.items():
        if count == max_count:
            winners.append(vote)
    return random.choice(winners)


def majority_vote_short(votes):
    """Returns the mode of the vote list."""
    mode, count = scipy.stats.mstats.mode(votes)
    return mode


def find_nearest_neighbours(p, points, k=5):
    """Find the k nearest neighbours of point p."""
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]


def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbours(p, points, k)
    return majority_vote(outcomes[ind])


data = pd.DataFrame(pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv"))
numeric_data = data.drop("color", axis=1)
numeric_data = (numeric_data - numeric_data.mean()) / (numeric_data.std(ddof=0))
pca = sklearn.decomposition.PCA(n_components=2)
principal_components = pca.fit(numeric_data).transform(numeric_data)
observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:, 0]
y = principal_components[:, 1]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = knn.predict(numeric_data)
n_rows = data.shape[0]
random.seed(123)
selection = random.sample(range(n_rows), 10)


def accuracy(predictions, outcomes):
    acc = 0
    for i in range(len(outcomes)):
        if predictions[i] == outcomes[i]:
            acc += 1
    return (acc / len(outcomes)) * 100


predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["high_quality"])
my_predictions = np.array([knn_predict(p, predictors[training_indices, :], outcomes, k=5) for p in predictors[selection]])
percentage = accuracy(my_predictions, data.high_quality[selection])
print(percentage)
plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha=0.2,
            c=data['high_quality'], cmap=observation_colormap, edgecolors='none')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
# plt.show()
print(accuracy(library_predictions, data['high_quality']))
