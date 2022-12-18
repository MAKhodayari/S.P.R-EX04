import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def load_dataset():
    olivetti = datasets.fetch_olivetti_faces()
    x = olivetti.images
    y = olivetti.target
    x = x.reshape((400, 4096))
    return x, y


def calc_mu(x, y):
    classes, counts = np.unique(y, return_counts=True)
    c_class = len(classes)
    _, n_feature = x.shape
    mu = np.zeros((c_class, n_feature))
    for i in classes:
        mu[i] = np.sum(np.array(x[y == i]), axis=0) / counts[i]
    return mu


def cal_sw_sb(x,y):
    _, n_feature = x.shape
    classes, counts = np.unique(y, return_counts=True)
    S_W = np.zeros((n_feature,n_feature))
    S_B = np.zeros((n_feature,n_feature))
    mu = calc_mu(x, y)
    mean_overall = np.mean(x, axis=0)
    for cl in classes:                 
        m_s = (x[y == cl]-mu[cl])
        S_W += np.dot(m_s.T, m_s)
    for cl in classes:
        mean_diff = (mu[cl] - mean_overall).reshape(n_feature, 1)
        S_B += counts[cl] * np.dot(mean_diff, mean_diff.T)
    return S_W, S_B


def LDA(X, y):
    scatter_w, scatter_b = cal_sw_sb(X, y)
    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(np.linalg.inv(scatter_w), scatter_b))
    eigenvectors = eigenvectors.T
    index = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[index]
    return eigenvectors
