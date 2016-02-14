"""

Exposure Matrix Factorization for collaborative filtering

CREATED: 2015-05-28 01:16:56 by Dawen Liang <dliang@ee.columbia.edu>

"""


import os
import sys
import time
import numpy as np
from numpy import linalg as LA

from joblib import Parallel, delayed
from math import sqrt
from sklearn.base import BaseEstimator, TransformerMixin

import rec_eval

floatX = np.float32
EPS = 1e-8


class ExpoMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, max_iter=10, batch_size=1000,
                 init_std=0.01, n_jobs=8, random_state=None, save_params=False,
                 save_dir='.', early_stopping=False, verbose=False, **kwargs):
        '''
        Exposure matrix factorization

        Parameters
        ---------
        n_components : int
            Number of latent factors
        max_iter : int
            Maximal number of iterations to perform
        batch_size: int
            Batch size to perform parallel update
        init_std: float
            The latent factors will be initialized as Normal(0, init_std**2)
        n_jobs: int
            Number of parallel jobs to update latent factors
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        save_params: bool
            Whether to save parameters after each iteration
        save_dir: str
            The directory to save the parameters
        early_stopping: bool
            Whether to early stop the training by monitoring performance on
            validation set
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters
        '''
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.init_std = init_std
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_params = save_params
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        ''' Model hyperparameters

        Parameters
        ---------
        lambda_theta, lambda_beta: float
            Regularization parameter for user (lambda_theta) and item factors (
            lambda_beta). Default value 1e-5. Since in implicit feedback all
            the n_users-by-n_items data points are used for training,
            overfitting is almost never an issue
        lambda_y: float
            inverse variance on the observational model. Default value 1.0
        init_mu: float
            All the \mu_{i} will be initialized as init_mu. Default value is
            0.01. This number should change according to the sparsity of the
            data (sparser data with smaller init_mu). In the experiment, we
            select the value from validation set
        a, b: float
            Prior on \mu_{i} ~ Beta(a, b)
        '''
        self.lam_theta = float(kwargs.get('lambda_theta', 1e-5))
        self.lam_beta = float(kwargs.get('lambda_beta', 1e-5))
        self.lam_y = float(kwargs.get('lam_y', 1.0))
        self.init_mu = float(kwargs.get('init_mu', 0.01))
        self.a = float(kwargs.get('a', 1.0))
        self.b = float(kwargs.get('b', 1.0))

    def _init_params(self, n_users, n_items):
        ''' Initialize all the latent factors '''
        self.theta = self.init_std * \
            np.random.randn(n_users, self.n_components).astype(floatX)
        self.beta = self.init_std * \
            np.random.randn(n_items, self.n_components).astype(floatX)
        self.mu = self.init_mu * np.ones(n_items, dtype=floatX)

    def fit(self, X, vad_data=None, **kwargs):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape (n_users, n_items)
            Training data.

        vad_data: scipy.sparse.csr_matrix, shape (n_users, n_items)
            Validation data.

        **kwargs: dict
            Additional keywords to evaluation function call on validation data

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_users, n_items = X.shape
        self._init_params(n_users, n_items)
        self._update(X, vad_data, **kwargs)
        return self

    def transform(self, X):
        pass

    def _update(self, X, vad_data, **kwargs):
        '''Model training and evaluation on validation set'''
        n_users = X.shape[0]
        XT = X.T.tocsr()  # pre-compute this
        old_ndcg = -np.inf
        for i in xrange(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
                start_t = _writeline_and_time('\tUpdating user factors...')
            self.theta = recompute_factors(self.beta, self.theta, X,
                                           self.lam_theta / self.lam_y,
                                           self.lam_y,
                                           self.mu,
                                           self.n_jobs,
                                           batch_size=self.batch_size)
            if self.verbose:
                print('\r\tUpdating user factors: time=%.2f'
                      % (time.time() - start_t))
                start_t = _writeline_and_time('\tUpdating item factors...')
            self.beta = recompute_factors(self.theta, self.beta, XT,
                                          self.lam_beta / self.lam_y,
                                          self.lam_y,
                                          self.mu,
                                          self.n_jobs,
                                          batch_size=self.batch_size)
            if self.verbose:
                print('\r\tUpdating item factors: time=%.2f'
                      % (time.time() - start_t))
                sys.stdout.flush()

            if self.verbose:
                start_t = _writeline_and_time('\tUpdating consideration prior...')

            start_idx = range(0, n_users, self.batch_size)
            end_idx = start_idx[1:] + [n_users]

            A_sum = np.zeros_like(self.mu)
            for lo, hi in zip(start_idx, end_idx):
                A_sum += a_row_batch(X[lo:hi], self.theta[lo:hi], self.beta,
                                     self.lam_y, self.mu).sum(axis=0)
            self.mu = (self.a + A_sum - 1) / (self.a + self.b + n_users - 2)
            if self.verbose:
                print('\r\tUpdating consideration prior: time=%.2f'
                      % (time.time() - start_t))
                sys.stdout.flush()

            if vad_data is not None:
                vad_ndcg = rec_eval.normalized_dcg_at_k(X, vad_data,
                                                        self.theta,
                                                        self.beta,
                                                        **kwargs)

                if self.verbose:
                    print('\tValidation NDCG@k: %.4f' % vad_ndcg)
                    sys.stdout.flush()
                if self.early_stopping and old_ndcg > vad_ndcg:
                    break  # we will not save the parameter for this iteration
                old_ndcg = vad_ndcg
            if self.save_params:
                self._save_params(i)
        pass

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'ExpoMF_K%d_mu%.1e_iter%d.npz' % (self.n_components,
                                                     self.init_mu, iter)
        np.savez(os.path.join(self.save_dir, filename), U=self.theta,
                 V=self.beta, mu=self.mu)


# Utility functions #

def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]


def a_row_batch(Y_batch, theta_batch, beta, lam_y, mu):
    '''Compute the posterior of exposure latent variables A by batch'''
    pEX = sqrt(lam_y / 2 * np.pi) * \
        np.exp(-lam_y * theta_batch.dot(beta.T)**2 / 2)
    A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
    A[Y_batch.nonzero()] = 1.
    return A


def _solve(k, A_k, X, Y, f, lam, lam_y, mu):
    '''Update one single factor'''
    s_u, i_u = get_row(Y, k)
    a = np.dot(s_u * A_k[i_u], X[i_u])
    B = X.T.dot(A_k[:, np.newaxis] * X) + lam * np.eye(f)
    return LA.solve(B, a)


def _solve_batch(lo, hi, X, X_old_batch, Y, m, f, lam, lam_y, mu):
    '''Update factors by batch, will eventually call _solve() on each factor to
    keep the parallel process busy'''
    assert X_old_batch.shape[0] == hi - lo

    if mu.size == X.shape[0]:  # update users
        A_batch = a_row_batch(Y[lo:hi], X_old_batch, X, lam_y, mu)
    else:  # update items
        A_batch = a_row_batch(Y[lo:hi], X_old_batch, X, lam_y, mu[lo:hi,
                                                                  np.newaxis])

    X_batch = np.empty_like(X_old_batch, dtype=X_old_batch.dtype)
    for ib, k in enumerate(xrange(lo, hi)):
        X_batch[ib] = _solve(k, A_batch[ib], X, Y, f, lam, lam_y, mu)
    return X_batch


def recompute_factors(X, X_old, Y, lam, lam_y, mu, n_jobs, batch_size=1000):
    '''Regress X to Y with exposure matrix (computed on-the-fly with X_old) and
    ridge term lam by embarrassingly parallelization. All the comments below
    are in the view of computing user factors'''
    m, n = Y.shape  # m = number of users, n = number of items
    assert X.shape[0] == n
    assert X_old.shape[0] == m
    f = X.shape[1]  # f = number of factors

    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(delayed(_solve_batch)(
        lo, hi, X, X_old[lo:hi], Y, m, f, lam, lam_y, mu)
        for lo, hi in zip(start_idx, end_idx))
    X_new = np.vstack(res)
    return X_new
