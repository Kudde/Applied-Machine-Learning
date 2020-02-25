import numpy as np
import scipy as sp
from sklearn.metrics.cluster import completeness_score
import matplotlib.pyplot as plt

class GMM:
    def __init__(self):
        #notmalised dataset
        self.nX = None
        #dataset shape
        self.m = None
        self.n = None
        #num clusters
        self.c = None
        #variabels
        self.mu = None
        self.sigma = None
        self.pi = None
        self.r_ic = None
        #add to sigma in order to avoid singular matrix
        self.reg_cov = None
    
    def initialise(self, X, Y):
        self.nX = X/16
        self.m, self.n = self.nX.shape
        self.c = len(set(Y))
        self.reg_cov = 1e-6*np.identity(self.n)

        self.mu = np.asmatrix(np.random.random((self.c, self.n)))
        self.sigma = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.c)])
        self.sigma *=8
        #self.sigma = np.array(np.asmatrix(np.random.random()))
        self.pi = np.ones(self.c)/self.c
        #?
        self.r_ic = np.asmatrix(np.empty((self.m, self.c), dtype=float))

    def fit(self, X, Y):
        self.initialise(X, Y)
        # print(self.nX.shape)
        # print("Mu: ", self.mu.shape)
        # print("sigma: ", self.sigma.shape)
        # print("Pi: ", self.pi.shape)
        iter_count = 0
        log_like = 0
        previous_log_like = 0

        while abs(log_like - previous_log_like) > 1e-8 or log_like == 0:
        # while iter_count < 5:
            print("i: ", iter_count)
            previous_log_like = log_like
            
            self.r_ic = self.e_step(self.nX, self.r_ic)
            
            self.m_step(X)
            
            
            
            log_like = self.log_likelihood()
            if iter_count % 10 == 0 or iter_count == 0:
                print(iter_count)
                print(abs(log_like - previous_log_like))
                self.plot(self.mu)
                
            iter_count += 1
        clusters = []
        for i in range(self.m):
            clusters.append(np.argmax(self.r_ic[i]))
        print("comp_score: ", completeness_score(Y, clusters))

        print("fit done", iter_count)

    def log_likelihood(self):
        #da poo poo, this needs work
        log_like = 0
        for i in range(self.m):
            temp = 0
            for k in range(self.c):
                # print("---------------------------------------------------------------")
                # print("X: ", self.nX[i, :])
                # print("mu: ", self.mu[k, :].A1)
                # print("sigma: ", self.sigma[k, :]) 
                self.sigma[k, :] += self.reg_cov
                
                temp += sp.stats.multivariate_normal.pdf(self.nX[i,:],
                                                        self.mu[k, :].A1,
                                                        self.sigma[k, :])*self.pi[k]
            log_like += np.log(temp)
        return log_like

    def e_step(self, features, r_ic):
        features_m = features.shape[0]
        for i in range(features_m):
            sum_of_probabilities = 0
            for k in range(self.c):
                #nominator
                self.sigma[k] += self.reg_cov
                probability_of_class = self.pi[k]*sp.stats.multivariate_normal.pdf(
                    self.nX[i,:],
                     mean = self.mu[k].A1,
                     cov = self.sigma[k])
                #add to denominator
                sum_of_probabilities += probability_of_class
                r_ic[i, k] = probability_of_class
            #divide nominator by denominator                
            r_ic[i, :] /= sum_of_probabilities
        return r_ic

    def m_step(self, train_features):
        # print(self.r_ic)
        for k in range(self.c):
            r_k = self.r_ic[:, k].sum()

            self.pi[k] = r_k/self.m

            mu_k = np.zeros(self.n)
            sigma_k = np.zeros((self.n, self.n))

            for i in range(self.m):
                mu_k += (self.nX[i, :] * self.r_ic[i, k])
            self.mu[k] = mu_k / r_k

            for i in range(self.m):
                sigma_k += np.dot(self.r_ic[i, k] * (self.nX[i,:]- self.mu[k]).T , (self.nX[i, :] - self.mu[k]))

            self.sigma[k] = (sigma_k / r_k)


            # #
            #
            #
            # rk = r.sum(axis=0)
            # weights = rk / len(x)
            #
            # delta_mean = -np.inf
            # delta_cov = -np.inf
            #
            # mean = np.sum([r[i, k] * xi for i, xi in enumerate(x)], axis=0) / np.sum(rk[k])
            # delta_mean = np.maximum(delta_mean, np.abs(means[k] - mean).max())
            # means[k] = mean
            #
            # cov = np.sum([r[i, k] * xi ** 2 for i, xi in enumerate(x)], axis=0) / np.sum(rk[k]) - mean ** 2 + 1e-2
            # delta_cov = np.maximum(delta_cov, np.abs(covs[k] - cov).max())
            # covs[k] = cov
            #
            # # ##



    def plot(self, means):
        
        fig, ax = plt.subplots(2, 5, subplot_kw=dict(xticks=[], yticks=[]), gridspec_kw={'hspace': 0, 'wspace': 0})
        fig.tight_layout()
        for i, axi in enumerate(ax.flat):
            im = axi.imshow(np.reshape(means[i], (8,8)), cmap=plt.cm.gray)
        plt.show()
    
    def predict(self, test_features):
        pred_r_ic = np.asmatrix(np.empty((test_features.shape[0], self.c), dtype=float))
        test_features /= 16
        pred_r_ic = self.e_step(test_features, pred_r_ic)
        preds = []
        for i in range(test_features.shape[0]):
            preds.append(np.argmax(pred_r_ic[i]))
        return preds
