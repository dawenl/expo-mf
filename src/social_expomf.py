"""

Social Exposure Matrix Factorization for collaborative filtering

CREATED: 2016-03-30 18:47:20 by Dawen Liang <dliang@ee.columbia.edu>

"""


import os
import sys
import time
import numpy as np
from numpy import linalg as LA

from joblib import Parallel, delayed
from math import sqrt
from scipy import sparse, weave
from sklearn.base import BaseEstimator, TransformerMixin

import rec_eval

floatX = np.float32
EPS = 1e-8


class SocialExpoMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, max_iter=10, batch_size=1000,
                 batch_sgd=10, max_epoch=10, init_std=0.01, n_jobs=8,
                 random_state=None, save_params=False, save_dir='.',
                 early_stopping=False, verbose=False, **kwargs):
        '''
        Social Exposure matrix factorization

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
        self.lam_tau = float(kwargs.get('lambda_tau', 1e-5))
        self.lam_y = float(kwargs.get('lam_y', 1.0))
        self.learning_rate = float(kwargs.get('learning_rate', 0.1))
        self.init_mu = float(kwargs.get('init_mu', 0.01))

    def _init_CF_params(self, n_users, n_items):
        ''' Initialize all the CF latent factors '''
        # user CF factors
        self.theta = self.init_std * \
            np.random.randn(n_users, self.n_components).astype(floatX)
        # item CF factors
        self.beta = self.init_std * \
            np.random.randn(n_items, self.n_components).astype(floatX)

    def _init_expo_params(self, N, n_users):
        # initialize exposure parameters
        # social-network exposure influence, which has the same sparsity
        # pattern as the network graph
        self.tau = sparse.csr_matrix(N)
        self.tau.data = self.init_std * \
            np.random.randn(*self.tau.data.shape).astype(floatX)
        self.tauT = self.tau.T.tocsr()
        # user exposure bias
        self.alpha = np.log(self.init_mu / (1 - self.init_mu)) * \
            np.ones((n_users, 1), dtype=floatX)

    def fit(self, X, N, vad_data=None, **kwargs):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape (n_users, n_items)
            Training data.

        N : scipy.sparse.csr_matrix, shape (n_users, n_users)
            Social networks.

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
        self._init_CF_params(n_users, n_items)
        self._init_expo_params(N, n_users)

        self._update(X, N, vad_data, **kwargs)
        return self

    def transform(self, X):
        pass

    def _update(self, X, N, vad_data, **kwargs):
        '''Model training and evaluation on validation set'''
        XT = X.T.tocsr()  # pre-compute this
        self.vad_ndcg = -np.inf

        for i in xrange(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            self._update_factors(X, XT)
            if self.verbose:
                start_t = _writeline_and_time(
                    '\tUpdating user exposure factors...\n')
            self._update_expo(XT, N)
            if self.verbose:
                print('\tUpdating user consideration factors: time=%.2f'
                      % (time.time() - start_t))
                sys.stdout.flush()

            if vad_data is not None:
                self._validate(X, N, vad_data)

            if self.save_params:
                self._save_params(i)
        pass

    def _update_factors(self, X, XT):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating user factors...')
        self.theta = recompute_factors(self.beta, self.theta, self.tau,
                                       self.alpha, X,
                                       self.lam_theta / self.lam_y,
                                       self.lam_y,
                                       self.n_jobs,
                                       batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating user factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating item factors...')
        self.beta = recompute_factors(self.theta, self.beta,
                                      self.tauT, self.alpha, XT,
                                      self.lam_beta / self.lam_y,
                                      self.lam_y,
                                      self.n_jobs,
                                      batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating item factors: time=%.2f'
                  % (time.time() - start_t))
        pass

    def _update_expo(self, XT, N):
        '''Update user exposure factors and bias with mini-batch SGD'''
        alpha_old = self.alpha.copy()

        n_items = XT.shape[0]
        start_idx = range(0, n_items, self.batch_sgd)
        end_idx = start_idx[1:] + [n_items]

        # run SGD to learn nu and alpha
        for epoch in xrange(self.max_epoch):
            idx = np.random.permutation(n_items)

            # take the last batch as validation
            lo, hi = start_idx[-1], end_idx[-1]
            A_vad = a_row_batch(XT[idx[lo:hi]].nonzero(),
                                self.beta[idx[lo:hi]], self.theta,
                                XT[idx[lo:hi]], self.tauT,
                                alpha_old.T, self.lam_y).T

            pred_vad = get_mu(self.tau, XT[idx[lo:hi]].T.tocsr(), self.alpha)
            init_loss = -np.sum(A_vad * np.log(pred_vad) + (1 - A_vad) *
                                np.log(1 - pred_vad)) / self.batch_sgd + \
                self.lam_nu / 2 * np.sum(self.tau.data**2)
            if self.verbose:
                print('\t\tEpoch #%d: initial validation loss = %.3f' %
                      (epoch, init_loss))
                sys.stdout.flush()

            for lo, hi in zip(start_idx[:-1], end_idx[:-1]):
                A_batch = a_row_batch(XT[idx[lo:hi]].nonzero(),
                                      self.beta[idx[lo:hi]], self.theta,
                                      XT[idx[lo:hi]], self.tauT,
                                      alpha_old.T, self.lam_y).T
                pred_batch = get_mu(self.tau, XT[idx[lo:hi]].T.tocsr(),
                                    self.alpha)
                diff = A_batch - pred_batch   # (n_users, batch_sgd)
                assert diff.shape == (N.shape[0], self.batch_sgd)
                #grad_nu = 1. / self.batch_sgd * diff.dot(XT[idx[lo:hi]])\
                #    - self.lam_nu * self.nu
                grad_nz = 1. / self.batch_sgd * _inner(diff, XT[idx[lo:hi]].toarray(), *N.nonzero()) - self.lam_tau * self.tau.data
                #self.nu += self.learning_rate * grad_nu
                self.tau.data += self.learning_rate * grad_nz
                self.alpha += self.learning_rate * diff.mean(axis=1,
                                                             keepdims=True)

            lo, hi = start_idx[-1], end_idx[-1]
            pred_vad = get_mu(self.tau, XT[idx[lo:hi]].T.tocsr(), self.alpha)
            loss = -np.sum(A_vad * np.log(pred_vad) + (1 - A_vad) *
                           np.log(1 - pred_vad)) / self.batch_sgd + \
                self.lam_nu / 2 * np.sum(self.tau.data**2)
            if self.verbose:
                print('\t\tEpoch #%d: validation loss = %.3f' % (epoch, loss))
                sys.stdout.flush()
                # It seems that after a few epochs the validation loss will
                # not decrease. However, we empirically found that it is still
                # better to train for more epochs, instead of stop the SGD
        self.tauT = self.tau.T.tocsr()
        pass

    def _validate(self, X, vad_data, **kwargs):
        mu = dict(params=[self.tau, X, self.alpha],
                  func=get_mu)
        vad_ndcg = rec_eval.normalized_dcg_at_k(X, vad_data,
                                                self.theta,
                                                self.beta,
                                                mu=mu,
                                                **kwargs)
        if self.verbose:
            print('\tValidation NDCG@k: %.4f' % vad_ndcg)
            sys.stdout.flush()
        if self.early_stopping and self.vad_ndcg > vad_ndcg:
            break  # we will not save the parameter for this iteration
        self.vad_ndcg = vad_ndcg

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'ExpoMF_cov_K%d_mu%.1e_iter%d.npz' % (self.n_components,
                                                         self.init_mu, iter)
        np.savez(os.path.join(self.save_dir, filename), U=self.theta,
                 V=self.beta, tau=self.tau.data, alpha=self.alpha)


# Utility functions #

def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def _inner(X, Y, rows, cols):
    '''
    Sparse dot product. Given matrices X and Y, and indices (rows, cols),
    compute the entries of X.dot(Y)[rows, cols]

    X: (m, k)
    Y: (k, n)
    '''
    n_out = rows.size
    k, n = Y.shape
    out = np.empty(n_out, dtype=X.dtype)
    code = r"""
    for (int i = 0; i < n_out; i++) {
       out[i] = 0.0;
       for (int j = 0; j < k; j++) {
           out[i] += X[rows[i] * k + j] * Y[j * n + cols[i]];
       }
    }
    """
    weave.inline(code, ['out', 'X', 'Y', 'rows', 'cols',
                        'n', 'k', 'n_out'])
    return out


def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]


def inv_logit(x):
    return 1. / (1 + np.exp(-x))


def get_mu(tau, Y, alpha):
    ''' \mu_{ui} = inv_logit(\sum_{v \in N(u)} tau_{uv} y_{vi} + alpha_u)

    When updating users:
        tau: (batch_users, n_users) sparse social network influence matrix
        Y: (n_users, n_items) sparse click matrix
        alpha: (batch_users, 1) per-user bias
        return: (batch_users, n_items) dense ndarray

    When updating items:
        tau: (batch_items, n_users) sparse click matrix
        Y: (n_users, n_users) sparse social network influence matrix^T
        alpha: (1, n_users)
        return: (batch_items, n_users) dense ndarray

    '''
    return inv_logit(tau.dot(Y).toarray() + alpha)


def a_row_batch(nz_idx, theta_batch, beta, tau_batch, Y, alpha_batch,
                lam_y):
    '''Compute the posterior of exposure latent variables A by batch

    When updating users:
        theta_batch: (batch_users, n_components)
        beta: (n_items, n_components)
        tau_batch: (batch_users, n_users)
        Y: (n_users, n_items)
        alpha_batch: (batch_users, 1)

    When updating items:
        theta_batch: (batch_item, n_components)
        beta: (n_users, n_components)
        tau_batch: (batch_items, n_users)
        Y: (n_users, n_users)
        alpha_batch: (1, n_users)
    '''
    pEX = sqrt(lam_y / 2 * np.pi) * \
        np.exp(-lam_y * theta_batch.dot(beta.T)**2 / 2)
    mu = get_mu(tau_batch, Y, alpha_batch)
    A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
    A[nz_idx] = 1.
    return A


def _solve(k, A_k, X, Y, f, lam, lam_y):
    '''Update one single factor'''
    s_u, i_u = get_row(Y, k)
    a = np.dot(s_u * A_k[i_u], X[i_u])
    B = X.T.dot(A_k[:, np.newaxis] * X) + lam * np.eye(f)
    return LA.solve(B, a)


def _solve_batch(lo, hi, X, X_old_batch, S, alpha, Y, m, f, lam, lam_y):
    '''Update factors by batch, will eventually call _solve() on each factor to
    keep the parallel process busy

    When updating users:
        X: (n_items, n_components)
        X_old_batch: (batch_users, n_components)
        S: (n_users, n_users)
        alpha: (n_users, n_users)
        Y: (n_users, n_items)

    When updating items:
        X: (n_users, n_components)
        X_old_batch: (batch_items, n_components)
        S: (n_users, n_users) social influence matrix^T
        alpha: (n_users, 1)
        Y: (n_items, n_users) click matrix^T
    '''
    assert X_old_batch.shape[0] == hi - lo
    X_batch = np.empty_like(X_old_batch, dtype=X_old_batch.dtype)

    if X.shape[0] == alpha.shape[0]:  # update item
        A_batch = a_row_batch(Y[lo:hi].nonzero(), X_old_batch, X, Y[lo:hi], S,
                              alpha.T, lam_y)
    else:  # update user
        A_batch = a_row_batch(Y[lo:hi].nonzero(), X_old_batch, X, S[lo:hi], Y,
                              alpha[lo:hi], lam_y)

    for ib, k in enumerate(xrange(lo, hi)):
        X_batch[ib] = _solve(k, A_batch[ib], X, Y, f, lam, lam_y)
    return X_batch


def recompute_factors(X, X_old, S, alpha, Y, lam, lam_y, n_jobs,
                      batch_size=1000):
    '''Regress X to Y with exposure matrix (computed on-the-fly with X_old) and
    ridge term lam by embarrassingly parallelization. All the comments below
    are in the view of computing user factors

    When updating users:
        X: item CF factors (beta: (n_items, n_components))
        X_old: user CF factors (theta: (n_users, n_components))
        S: social influence factors (tau: (n_users, n_users))
        alpha: user exposure bias (alpha: (n_users, 1))

    When updating items:
        X: user CF factors (theta: (n_users, n_components))
        X_old: item CF factors (beta: (n_items, n_components))
        S: social influence factors^T (tarT: (n_users, n_users))
        alpha: user exposure bias (alpha: (n_users, 1))
    '''
    m, n = Y.shape  # m = number of users, n = number of items
    assert X.shape[0]
    assert X_old.shape[0] == S.shape[0] == S.shape[1] == m
    f = X.shape[1]  # f = number of factors

    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(delayed(_solve_batch)(
        lo, hi, X, X_old[lo:hi], S, alpha, Y, m, f, lam, lam_y)
        for lo, hi in zip(start_idx, end_idx))
    X_new = np.vstack(res)
    return X_new
