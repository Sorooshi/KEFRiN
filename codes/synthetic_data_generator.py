import os
import pickle
import random
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from copy import deepcopy
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.pyplot as plt
from pandas import DataFrame as df
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

np.set_printoptions(suppress=True, linewidth=120, precision=2)


def clusters_cardinality(N, K):
    """
    :param N: Number nodes
    :param K: Number of Communities
    :return: communi(I think it's completely natural)ties/clusters cardinality
    """
    minimum = 30  # Minimum Number of Entities in each cluster

    remaining = N - K * minimum

    ur = [np.random.uniform() for k in range(K - 1)]
    ur.sort()

    cardinality = []
    tmp = []
    for k in range(K - 1):
        if k == 0:
            tmp.append(ur[k] - 0)
        elif k == K - 2:
            tmp.append(1 - ur[k])
        else:
            tmp.append(ur[k + 1] - ur[k])

    cardinality = [int(remaining * i + minimum) for i in tmp]  # intergers
    cardinality += [N - sum(cardinality)]

    return cardinality


def generate_Y(N, V, K, pr_v, cardinality, features_type, V_noise1):

    """
    :param N: Int, Number of Entries
    :param V: Int, Number of Features
    :param K: Int, Number of Clusters
    :param pr_v: float, [0, 1], coefficient to control the cluster intermix and/or homogeneity of features in a cluster
    :param cardinality: list of lists, Cardinality of cluster plus each entry index in the attributed networks
    :param features_type: str, type of features, quantitative, categorical or the mix (combination of them)
    :param V_noise1: Int, number of columns to insert noisy features regardging the
                    noise model1 regarding Renato, Makarenkov, Mirkin 2016 -page 348.
    :return: A pickle of data, containing Entity-to-feature matrix with or without noise, and the networks
    """

    if features_type == 'Q':  # quantitative

        d1, d2 = -1, 1  # a range for mean
        d1_, d2_ = 0.025 * (d2 - d1), 0.05 * (d2 - d1)  # a range for covariance matrix

        Y = np.zeros([N, V])  # empty entity-to-feature matrix

        interval = 0
        for k in range(K):
            mean = np.multiply(pr_v, np.random.normal(loc=d1, scale=d2, size=V))
            cov_ = np.random.uniform(low=d1_, high=d2_, size=(V, V))
            cov_ = np.add(cov_, cov_.T) / 2
            cov_ = np.dot(cov_, cov_.T)  # Gram Matrix
            cov = np.zeros([V, V])
            for i in range(V):
                for j in range(V):
                    if i == j:
                        cov[i, j] = cov_[i, j]
            tmp = np.random.multivariate_normal(mean=mean, cov=cov, size=cardinality[k])
            row = 0
            for i in range(interval, cardinality[k] + interval):
                Y[i, :] = tmp[row, :]  # np.random.multivariate_normal(mean=mean, cov=cov_)
                row += 1

            interval += cardinality[k]

        # Y with noise
        # Noise Model1 noisy features to inset
        max_features = np.max(Y, axis=0)
        min_features = np.min(Y, axis=0)
        noise1_ = np.random.uniform(low=min_features, high=max_features, size=(N, V))
        column_to_insert = np.random.choice(V, V_noise1, replace=False)
        noise1 = noise1_[:, column_to_insert]

        Yn = np.concatenate((Y, noise1), axis=1)

        # Noise Model2:
        # indices = list(np.random.randint(low=0, high=N, size=int(np.ceil(N * V_noise2))))
        # noises2 = np.random.uniform(low=0, high=1, size=(len(indices), V_noise2))
        # e = 0
        # for i in indices:
        #     Yn[i, :] = noises2[e, :]
        #     e += 1

    elif features_type == 'C':  # categorical

        # As it is mentioned in the paper (just to reduce the space complexity)
        d1, d2 = 2, V + 5  # a range for subcategories

        ck = np.zeros([K, V])
        cntr = 0

        while cntr < K:
            coincidences = 0
            center = np.random.randint(low=d1, high=d2, size=V)
            if cntr != 0:
                for v in range(len(center)):
                    # if subcategory "v" exists and it forms more than 50% of that category
                    if center[v] in list(ck[:, v]) and list(ck[:, v]).count(center[v]) / len(list(ck[:, v])) > 0.5:
                        coincidences += 1

            # if the total number of coincidences of a cluster center is less than half of
            # the number of features that cluster
            if coincidences <= np.ceil(V / 2):
                ck[cntr, :] = center
                cntr += 1

        Y = np.zeros([N, V])  # empty entity-to-feature matrix
        interval = 0

        for k in range(K):
            for i in range(interval, cardinality[k] + interval):
                for v in range(V):
                    if random.random() > pr_v:
                        noisy_v = list(set(ck[:, v]).difference([ck[k, v]]))
                        noisy_v = np.random.choice(noisy_v)
                        Y[i, v] = noisy_v
                    else:
                        Y[i, v] = ck[k, v]

            interval += cardinality[k]

        Yn = []

    elif features_type == 'M':  # Mixed of Q and C

        Vq = int(np.ceil(V / 2))  # number of quant features
        Vc = int(np.floor(V / 2))  # number of categ features

        # Quantitative Section:
        d1, d2 = -1, 1  # a range for mean
        d1_, d2_ = 0.025 * (d2 - d1), 0.05 * (d2 - d1)  # a range for covariance matrix

        Y = np.zeros([N, V])  # empty entity-to-feature matrix

        interval = 0
        for k in range(K):
            mean = np.multiply(pr_v, np.random.normal(loc=d1, scale=d2, size=Vq))
            cov_ = np.random.uniform(low=d1_, high=d2_, size=(Vq, Vq))
            cov_ = np.add(cov_, cov_.T) / 2
            cov_ = np.dot(cov_, cov_.T)  # Gram Matrix
            cov = np.zeros([Vq, Vq])
            for i in range(Vq):
                for j in range(Vq):
                    if i == j:
                        cov[i, j] = cov_[i, j]
            tmp = np.random.multivariate_normal(mean=mean, cov=cov, size=cardinality[k])
            row = 0
            for i in range(interval, cardinality[k] + interval):
                Y[i, :Vq] = tmp[row, :]  # np.random.multivariate_normal(mean=mean, cov=cov_)
                row += 1

            interval += cardinality[k]

        # Categorical Section:
        # As it is mentioned in the paper (just to reduce the space complexity)
        d1, d2 = 2, Vc + 5  # a range for subcategories
        ck = np.zeros([K, Vc])
        cntr = 0

        while cntr < K:
            coincidences = 0
            center = np.random.randint(low=d1, high=d2, size=Vc)
            if cntr != 0:
                for v in range(len(center)):
                    # if subcategory "v" exists and it forms more than 50% of that category
                    if center[v] in list(ck[:, v]) and list(ck[:, v]).count(center[v]) / len(list(ck[:, v])) > 0.5:
                        coincidences += 1

            # if the total number of coincidences of a cluster center is less than half of
            # the number of features that cluster
            if coincidences <= np.ceil(Vc / 2):
                ck[cntr, :] = center
                cntr += 1

        interval = 0

        for k in range(K):
            for i in range(interval, cardinality[k] + interval):
                col = Vq
                for v in range(Vc):
                    if random.random() > pr_v:
                        noisy_v = list(set(ck[:, v]).difference([ck[k, v]]))
                        noisy_v = np.random.choice(noisy_v)
                        Y[i, col] = noisy_v
                    else:
                        Y[i, col] = ck[k, v]
                    col += 1

            interval += cardinality[k]

        # Y with noise
        # Noise Model1 noisy features to inset
        max_features = np.max(Y[:, :Vq], axis=0)
        min_features = np.min(Y[:, :Vq], axis=0)
        noise1_ = np.random.uniform(low=min_features, high=max_features, size=(N, Vq))
        column_to_insert = np.random.choice(Vq, V_noise1, replace=False)
        noise1 = noise1_[:, column_to_insert]

        Yn = np.concatenate((Y, noise1), axis=1)

        # Noise Model2:
        # indices = list(np.random.randint(low=0, high=N, size=int(np.ceil(N * V_noise2))))
        # noises2 = np.random.uniform(low=0, high=1, size=(len(indices), V_noise2))
        # e = 0
        # for i in indices:
        #     Yn[i, :] = noises2[e, :]
        #     e += 1

    return Y, Yn


def generate_P(N, cardinality, p_wth, p_btw, ):

    """
    :param cardinality: nodes list: list, node indices.
    :param p_wth:  float, [0,1], Probability for edge creation within the community.
    :param p_btw: float, [0,1], Probability for edge creation between the communities.
    :return: Dict of lists, each list in indeed the between-communities edge list of a community.
    """

    P = np.zeros([N, N])
    communities_structure = []

    intrv = 0
    for i in range(len(cardinality)):
        if i == 0:
            communities_structure.append(list(range(cardinality[i])))
        if i == len(cardinality) - 1:
            communities_structure.append(list(range(intrv, N)))
        elif 0 < i < len(cardinality) - 1:
            communities_structure.append(list(range(intrv, cardinality[i] + intrv)))
        intrv += cardinality[i]

    for k in communities_structure:
        for i in range(N):
            for j in range(i):
                if i in k and j in k:
                    if np.random.random() < p_wth:
                        P[i, j] = 1
                        P[j, i] = 1
                elif j in k and i not in k:
                    if np.random.random() < p_btw:
                        P[i, j] = 1
                        P[j, i] = 1

    return P



def adjacency(num_nodes, edges_list):
    G = np.zeros([num_nodes, num_nodes])
    for community, edges in edges_list.items():
        for edge in edges:
            G[edge[0], edge[1]] = 1
            G[edge[1], edge[0]] = 1
    return G



def flat_ground_truth(ground_truth):
    """
    :param ground_truth: the clusters/communities cardinality
                        (output of cluster cardinality from synthetic data generator)
    :return: two flat lists, the first one is the list of labels in an appropriate format
             for applying sklearn metrics. And the second list is the list of lists of
              containing indices of nodes in the corresponding cluster.
    """
    k = 1
    interval = 1
    labels_true, labels_true_indices = [], []
    for v in ground_truth:
        tmp_indices = []
        for vv in range(v):
            labels_true.append(k)
            tmp_indices.append(interval+vv)

        k += 1
        interval += v
        labels_true_indices += tmp_indices

    return labels_true, labels_true_indices

