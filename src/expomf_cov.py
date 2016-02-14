"""

Exposure Matrix Factorization with exposure covariates (e.g. topics, or
locations) for collaborative filtering

This file is largely adapted from expomf.py

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
                 batch_sgd=10, max_epoch=10, init_std=0.01, n_jobs=8,
                 random_state=None, save_params=False, save_dir='.',
                 early_stopping=False, verbose=False, **kwargs):
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
        batch_sgd: int
            Batch size for SGD when updating exposure factors
        max_epoch: int
            Number of epochs for SGD
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
        self.batch_sgd = batch_sgd
        self.max_epoch = max_epoch
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
        lambda_theta, lambda_beta, lambda_nu: float
            Regularization parameter for user (lambda_theta), item CF factors (
            lambda_beta) and user exposure factors (lambda_nu). Default value
            1e-5. Since in implicit feedback all the n_users-by-n_items data
            points are used for training, overfitting is almost never an issue
        lambda_y: float
            inverse variance on the observational model. Default value 1.0
        learning_rate: float
            Learning rate for SGD. Default value 0.1. Since for each user we
            are effectively doing logistic regression, constant learning rate
            should suffice
        init_mu: float
            init_mu is used to initalize the user exposure bias (alpha) such
            that all the \mu_{ui} = inv_logit(nu_u * pi_i + alpha_u) is roughly
            init_mu. Default value is 0.01. This number should change according
            to the sparsity of the data (sparser data with smaller init_mu).
            In the experiment, we select the value from validation set
        '''
        self.lam_theta = float(kwargs.get('lambda_theta', 1e-5))
        self.lam_beta = float(kwargs.get('lambda_beta', 1e-5))
        self.lam_nu = float(kwargs.get('lambda_nu', 1e-5))
        self.lam_y = float(kwargs.get('lam_y', 1.0))
        self.learning_rate = float(kwargs.get('learning_rate', 0.1))
        self.init_mu = float(kwargs.get('init_mu', 0.01))

    def _init_params(self, n_users, n_items):
        ''' Initialize all the latent factors '''
        # user CF factors
        self.theta = self.init_std * \
            np.random.randn(n_users, self.n_components).astype(floatX)
        # item CF factors
        self.beta = self.init_std * \
            np.random.randn(n_items, self.n_components).astype(floatX)
        # user exposure factors
        self.nu = self.init_std * \
            np.random.randn(n_users, self.n_components).astype(floatX)
        # user exposure bias
        self.alpha = np.log(self.init_mu / (1 - self.init_mu)) * \
            np.ones((n_users, 1), dtype=floatX)

    def fit(self, X, pi, vad_data=None, **kwargs):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape (n_users, n_items)
            Training data.

        pi : array-like, shape (n_items, n_components)
            item content representations (e.g. topics, locations)

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
        assert pi.shape[0] == n_items
        self._init_params(n_users, n_items)
        self._update(X, pi, vad_data, **kwargs)
        return self

    def transform(self, X):
        pass

    def _update(self, X, pi, vad_data, **kwargs):
        '''Model training and evaluation on validation set'''
        XT = X.T.tocsr()  # pre-compute this
        old_ndcg = -np.inf

        for i in xrange(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
                start_t = _writeline_and_time('\tUpdating user factors...')
            self.theta = recompute_factors(self.beta, self.theta, pi,
                                           self.nu, self.alpha, X,
                                           self.lam_theta / self.lam_y,
                                           self.lam_y,
                                           self.n_jobs,
                                           batch_size=self.batch_size)
            if self.verbose:
                print('\r\tUpdating user factors: time=%.2f'
                      % (time.time() - start_t))
                start_t = _writeline_and_time('\tUpdating item factors...')
            self.beta = recompute_factors(self.theta, self.beta, self.nu, pi,
                                          self.alpha, XT,
                                          self.lam_beta / self.lam_y,
                                          self.lam_y,
                                          self.n_jobs,
                                          batch_size=self.batch_size)
            if self.verbose:
                print('\r\tUpdating item factors: time=%.2f'
                      % (time.time() - start_t))
                start_t = _writeline_and_time('\tUpdating user consideration factors...\n')
            self.update_nu(XT, pi)
            if self.verbose:
                print('\tUpdating user consideration factors: time=%.2f'
                      % (time.time() - start_t))
                sys.stdout.flush()

            if vad_data is not None:
                mu = dict(params=[self.nu, pi, self.alpha],
                          func=get_mu)
                vad_ndcg = rec_eval.normalized_dcg_at_k(X, vad_data,
                                                        self.theta,
                                                        self.beta,
                                                        mu=mu,
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

    def update_nu(self, XT, pi):
        '''Update user exposure factors and bias with mini-batch SGD'''
        nu_old = self.nu.copy()
        alpha_old = self.alpha.copy()

        n_items = XT.shape[0]
        start_idx = range(0, n_items, self.batch_sgd)
        end_idx = start_idx[1:] + [n_items]

        # run SGD to learn nu and alpha
        for epoch in xrange(self.max_epoch):
            idx = np.random.permutation(n_items)

            # take the last batch as validation
            lo, hi = start_idx[-1], end_idx[-1]
            A_vad = a_row_batch(XT[idx[lo:hi]], self.beta[idx[lo:hi]],
                                self.theta, pi[idx[lo:hi]], nu_old,
                                alpha_old.T, self.lam_y).T
            pred_vad = get_mu(self.nu, pi[idx[lo:hi]], self.alpha)
            init_loss = -np.sum(A_vad * np.log(pred_vad) + (1 - A_vad) *
                                np.log(1 - pred_vad)) / self.batch_sgd + \
                self.lam_nu / 2 * np.sum(self.nu**2)
            if self.verbose:
                print('\t\tEpoch #%d: initial validation loss = %.3f' %
                      (epoch, init_loss))
                sys.stdout.flush()

            for lo, hi in zip(start_idx[:-1], end_idx[:-1]):
                A_batch = a_row_batch(XT[idx[lo:hi]], self.beta[idx[lo:hi]],
                                      self.theta, pi[idx[lo:hi]], nu_old,
                                      alpha_old.T, self.lam_y).T
                pred_batch = get_mu(self.nu, pi[idx[lo:hi]], self.alpha)
                diff = A_batch - pred_batch
                grad_nu = 1. / self.batch_sgd * diff.dot(pi[idx[lo:hi]])\
                    - self.lam_nu * self.nu
                self.nu += self.learning_rate * grad_nu
                self.alpha += self.learning_rate * diff.mean(axis=1,
                                                             keepdims=True)

            lo, hi = start_idx[-1], end_idx[-1]
            pred_vad = get_mu(self.nu, pi[idx[lo:hi]], self.alpha)
            loss = -np.sum(A_vad * np.log(pred_vad) + (1 - A_vad) *
                           np.log(1 - pred_vad)) / self.batch_sgd + \
                self.lam_nu / 2 * np.sum(self.nu**2)
            if self.verbose:
                print('\t\tEpoch #%d: validation loss = %.3f' % (epoch, loss))
                sys.stdout.flush()
                # It seems that after a few epochs the validation loss will
                # not decrease. However, we empirically found that it is still
                # better to train for more epochs, instead of stop the SGD
        pass

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'ExpoMF_cov_K%d_mu%.1e_iter%d.npz' % (self.n_components,
                                                         self.init_mu, iter)
        np.savez(os.path.join(self.save_dir, filename), U=self.theta,
                 V=self.beta, nu=self.nu, alpha=self.alpha)


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


def inv_logit(x):
    return 1. / (1 + np.exp(-x))


def get_mu(nu, pi, alpha):
    ''' \mu_{ui} = inv_logit(nu_u * pi_i + alpha_u)'''
    return inv_logit(nu.dot(pi.T) + alpha)


def a_row_batch(Y_batch, theta_batch, beta, nu_batch, pi, alpha_batch, lam_y):
    '''Compute the posterior of exposure latent variables A by batch

    When updating users:
        Y_batch: (batch_users, n_items)
        theta_batch: (batch_users, n_components)
        beta: (n_items, n_components)
        nu_batch: (batch_users, n_components)
        pi: (n_items, n_components)
        alpha_batch: (batch_users, 1)

    When updating items:
        Y_batch: (batch_item, n_users)
        theta_batch: (batch_item, n_components)
        beta: (n_users, n_components)
        nu_batch: (batch_item, n_components)
        pi: (n_users, n_components)
        alpha_batch: (1, n_users)
    '''
    pEX = sqrt(lam_y / 2 * np.pi) * \
        np.exp(-lam_y * theta_batch.dot(beta.T)**2 / 2)
    mu = get_mu(nu_batch, pi, alpha_batch)
    A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
    A[Y_batch.nonzero()] = 1.
    return A


def _solve(k, A_k, X, Y, f, lam, lam_y):
    '''Update one single factor'''
    s_u, i_u = get_row(Y, k)
    a = np.dot(s_u * A_k[i_u], X[i_u])
    B = X.T.dot(A_k[:, np.newaxis] * X) + lam * np.eye(f)
    return LA.solve(B, a)


def _solve_batch(lo, hi, X, X_old_batch, S, T_batch, alpha, Y, m, f, lam,
                 lam_y):
    '''Update factors by batch, will eventually call _solve() on each factor to
    keep the parallel process busy'''
    assert X_old_batch.shape[0] == hi - lo
    assert T_batch.shape[0] == hi - lo
    X_batch = np.empty_like(X_old_batch, dtype=X_old_batch.dtype)

    if X.shape[0] == alpha.shape[0]:  # update item
        A_batch = a_row_batch(Y[lo:hi], X_old_batch, X, T_batch, S, alpha.T,
                              lam_y)
    else:  # update user
        A_batch = a_row_batch(Y[lo:hi], X_old_batch, X, T_batch, S,
                              alpha[lo:hi], lam_y)

    for ib, k in enumerate(xrange(lo, hi)):
        X_batch[ib] = _solve(k, A_batch[ib], X, Y, f, lam, lam_y)
    return X_batch


def recompute_factors(X, X_old, S, T, alpha, Y, lam, lam_y, n_jobs,
                      batch_size=1000):
    '''Regress X to Y with exposure matrix (computed on-the-fly with X_old) and
    ridge term lam by embarrassingly parallelization. All the comments below
    are in the view of computing user factors

    When updating users:
        X: item CF factors (beta)
        X_old: user CF factors (theta)
        S: content topic proportions (pi)
        T: user consideration factors (nu)
        alpha: user consideration bias (alpha)

    When updating items:
        X: user CF factors (theta)
        X_old: item CF factors (beta)
        S: user consideration factors (nu)
        T: content topic proportions (pi)
        alpha: user consideration bias (alpha)
    '''
    m, n = Y.shape  # m = number of users, n = number of items
    assert X.shape[0] == n
    assert X_old.shape[0] == m
    assert X.shape == S.shape
    assert X_old.shape == T.shape
    f = X.shape[1]  # f = number of factors

    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(delayed(_solve_batch)(
        lo, hi, X, X_old[lo:hi], S, T[lo:hi], alpha, Y, m, f, lam, lam_y)
        for lo, hi in zip(start_idx, end_idx))
    X_new = np.vstack(res)
    return X_new
