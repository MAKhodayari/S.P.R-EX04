import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import metrics

def load_dataset():
    olivetti = datasets.fetch_olivetti_faces()
    data = olivetti.data
    x = olivetti.images
    y = olivetti.target
    # x = x.reshape((400, 4096))
    return data, x, y


def calc_mu(x, y):
    classes, counts = np.unique(y, return_counts=True)
    c_class = len(classes)
    _, n_feature = x.shape
    mu = np.zeros((c_class, n_feature))
    for i in classes:
        mu[i] = np.sum(np.array(x[y == i]), axis=0) / counts[i]
    return mu


def cal_sw_sb(x, y):
    _, n_feature = x.shape
    classes, counts = np.unique(y, return_counts=True)
    S_W = np.zeros((n_feature, n_feature))
    S_B = np.zeros((n_feature, n_feature))
    mu = calc_mu(x, y)
    mean_overall = np.mean(x, axis=0)
    for cl in classes:                 
        m_s = (x[y == cl] - mu[cl])
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


def zero_mean(data):
    feature_means = np.mean(data, axis=0)
    zero_mean_data = data - feature_means
    return feature_means, zero_mean_data


def calc_eigen_params(data):
    cov = np.dot(data.T, data)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    return eigen_values, eigen_vectors


def pca(data):
    eigen_values, eigen_vectors = calc_eigen_params(data)
    index = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[index]
    return eigen_vectors


def project(data, vector):
    projection = np.dot(data, vector.T)
    return projection

def load_data(path):
    data = pd.read_csv(path,sep=" ",header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X,y

def logisticRegression(x,y):
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
    logreg=LogisticRegression(solver='liblinear') 
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
    return acc

def chi(x,y,K):
    chi2_features = SelectKBest(chi2,k=K)
    X_kbest_features = chi2_features.fit_transform(x, y)
    return X_kbest_features

def Rfe(x,y,k):
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=k)
    X_kbest_features=rfe.fit_transform(x, y)
    return X_kbest_features

def Univariate(x,y,K):
    sel_f = SelectKBest(f_classif, k=K)
    X_kbest_features= sel_f.fit_transform(x,y)
    return X_kbest_features

def Result(x,y,name):
    k=[5,10]
    df=pd.DataFrame(columns=['k','chi2','RFE','Univariate'])
    for i in k:
        X_kbest_features_chi=chi(x,y,i)
        chi_acc=logisticRegression(X_kbest_features_chi,y)
        X_kbest_features_rfe=Rfe(x,y,i)
        rfe_acc=logisticRegression(X_kbest_features_rfe,y)
        X_kbest_features_Univariate=Univariate(x,y,i)
        Univariate_acc=logisticRegression(X_kbest_features_Univariate,y)
        #dict= {'K': i, 'chi2':chi_acc , 'RFE':rfe_acc,'Univariate':Univariate_acc }
        df2 = pd.DataFrame([[i,chi_acc,rfe_acc,Univariate_acc ]],columns=['k','chi2','RFE','Univariate']) 
        df= pd.concat([df, df2])
    print("dataset:",name)    
    print("original datasets acc:")
    original_acc=logisticRegression(x,y)
    print(original_acc)
    return df          
