import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster
from sklearn.cluster.bicluster import SpectralCoclustering


whisky = pd.read_csv("whiskies.txt")
whisky["Regions"] = pd.read_csv("regions.txt")
flavors = whisky.iloc[:, 2:14]
corr_flavors = pd.DataFrame.corr(flavors)
corr_whisky = pd.DataFrame.corr(flavors.transpose())
model = SpectralCoclustering(n_clusters=6, random_state=0)
whisky["Group"] = pd.Series(model.row_labels_, index=whisky.index)
whisky = whisky.reset_index(drop=True)
correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())
correlations = np.array(correlations)
plt.figure(figsize=(10, 10))
plt.pcolor(correlations)
plt.colorbar()
plt.show()
