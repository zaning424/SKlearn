import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn import cluster
from sklearn.manifold import TSNE
import pandas as pd


def find_max():
    # load data, X is data, y is target
    X, y = datasets.load_boston(return_X_y=True)
    # save factor's name in list
    col_name = ['CRIM','ZN','INDUS','CHAS', 'NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    # turn X to data frame
    df = pd.DataFrame(X)
    # apply linear regression model
    regr = linear_model.LinearRegression()
    regr.fit(df, y)
    # output coef of each factor to list
    coef = np.fabs(regr.coef_).tolist()
    # find max coef and its location in list
    n = coef.index(max(coef))
    # find corresponding column name of max coef
    max_indicator = col_name[n]
    return print('Largest factor is %s'%max_indicator)


def kmeans(n):
    # load data, X is data, y is target
    X, y = datasets.load_iris(return_X_y=True)
    # apply KMeans
    km = cluster.KMeans(n_clusters=n)
    # change 4 feature data in X to 2-dimension data sets for plot
    tsne = TSNE(n_components=2, init='random', random_state=12).fit(X)
    clusters = pd.DataFrame(tsne.embedding_)
    clusters['cluster_pred'] = km.fit_predict(X)
    # plot the output of KMeans n-cluster
    plt.scatter(clusters[0],clusters[1], c=clusters['cluster_pred'],cmap='rainbow')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    find_max()
    kmeans(3)
    kmeans(4)
    kmeans(5)

