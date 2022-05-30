import numpy as np
import pandas as pd
from numpy.linalg import solve


class ExplicitMF():
    def __init__(self, ratings):

        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.factors = 10
        self.item_reg = 0.1
        self.user_reg = 0.1

    def als_step(self, latent, fixed, ratings, _lambda, type='user'):
        """
        Alternating step — keep one matrix constant and find min of other matrix
        """
        if type == 'user':
            MTM = fixed.T.dot(fixed)
            lambdaI = np.eye(MTM.shape[0]) * _lambda

            for u in range(latent.shape[0]):
                latent[u, :] = solve((MTM + lambdaI),
                                             ratings[u, :].dot(fixed))
        elif type == 'item':
            # Precompute
            UTU = fixed.T.dot(fixed)
            lambdaI = np.eye(UTU.shape[0]) * _lambda

            for i in range(latent.shape[0]):
                latent[i, :] = solve((UTU + lambdaI),
                                             ratings[:, i].T.dot(fixed))
        return latent

    def train(self, n):
        " Train model for n iterations"
        # initialize latent vectors
        self.user_vecs = np.random.random((self.n_users, self.factors))
        self.item_vecs = np.random.random((self.n_items, self.factors))

        ctr = 1
        while ctr <= n:
            self.user_vecs = self.als_step(self.user_vecs, self.item_vecs, self.ratings, self.user_reg, type='user')
            self.item_vecs = self.als_step(self.item_vecs, self.user_vecs, self.ratings, self.item_reg, type='item')
            ctr += 1
