import numpy as np
import scipy.stats as ss
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
p = np.array([2.5, 2])
iris = datasets.load_iris()
predictors = iris.data[:, 0: 2]
outcomes = iris.target


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
    

def generate_synth_data(n=50):
    """Create two sets of points from bivariate normal distribution."""
    points = np.concatenate((ss.norm(0, 1).rvs((n, 2)), ss.norm(1, 1).rvs((n, 2))), axis=0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
    return (points, outcomes)


def make_prediction_grid(predictors, outcomes, limits, h, k):
    """Classify each point on the prediction grid."""
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)

    return (xx, yy, prediction_grid)


def plot_prediction_grid(xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap(["hotpink", "lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap(["red", "blue", "green"])
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap=background_colormap, alpha=0.5)
    plt.scatter(predictors[:, 0], predictors[:, 1], c=outcomes, cmap=observation_colormap, s=50)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.xticks(())
    plt.yticks(())
    plt.xlim(np.min(xx), np.max(xx))
    plt.ylim(np.min(yy), np.max(yy))
    plt.savefig(filename)


# predictors, outcomes = generate_synth_data()
limits = (4, 8, 1.5, 4.5)
filename = "iris.pdf"
xx, yy, prediction_grid = make_prediction_grid(predictors, outcomes, limits, 0.1, 5)
plt.plot(predictors[outcomes == 0][:, 0], predictors[outcomes == 0][:, 1], "ro")
plt.plot(predictors[outcomes == 1][:, 0], predictors[outcomes == 1][:, 1], "go")
plt.plot(predictors[outcomes == 2][:, 0], predictors[outcomes == 2][:, 1], "bo")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)
my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors])
print(np.mean(my_predictions == outcomes) * 100)
print(np.mean(sk_predictions == outcomes) * 100)
# plot_prediction_grid(xx, yy, prediction_grid, filename)
# plt.figure()
# plt.plot(points[:50, 0], points[:50, 1], "ro")
# plt.plot(points[50:, 0], points[50:, 1], "bo")
# outcomes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
# ind = find_nearest_neighbours(p, points, 2)
# print(knn_predict(np.array([2, 2.5]), points, outcomes, k=2))
# plt.plot(points[:, 0], points[:, 1], "ro")
# plt.plot(p[0], p[1], "bo")
# plt.axis([0.5, 3.5, 0.5, 3.5])
# plt.show()
