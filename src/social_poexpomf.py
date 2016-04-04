"""

Poisson Social Exposure Matrix Factorization for collaborative filtering

CREATED: 2016-04-03 17:52:11 by Dawen Liang <dliang@ee.columbia.edu>

"""


import os
import sys
import time
import numpy as np
from numpy import linalg as LA

from joblib import Parallel, delayed
from math import log, sqrt
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

import rec_eval

floatX = np.float32
EPS = 1e-8


class PoissonSocialExpoMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, max_iter=10, batch_size=500,
                 init_std=0.01, n_jobs=8, random_state=None, save_params=False,
                 save_dir='.', early_stopping=False, verbose=False, **kwargs):
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
            Regularization parameter for user (lambda_theta) and item CF
            factors (lambda_beta). Default value 1e-5. Since in implicit
            feedback all the n_users-by-n_items data points are used for
            training, overfitting is almost never an issue
        lambda_y: float
            inverse variance on the observational model. Default value 1.0
        init_mu: float
            init_mu is used to initalize the user exposure bias (alpha) such
            that all the \mu_{ui} = inv_logit(nu_u * pi_i + alpha_u) is roughly
            init_mu. Default value is 0.01. This number should change according
            to the sparsity of the data (sparser data with smaller init_mu).
            In the experiment, we select the value from validation set
        '''
        self.lam_theta = float(kwargs.get('lambda_theta', 1e-5))
        self.lam_beta = float(kwargs.get('lambda_beta', 1e-5))
        self.lam_y = float(kwargs.get('lam_y', 1.0))
        self.init_mu = float(kwargs.get('init_mu', 0.01))

    def _init_CF_params(self, n_users, n_items):
        ''' Initialize all the CF latent factors '''
        # user CF factors
        self.theta = self.init_std * \
            np.random.randn(n_users, self.n_components).astype(floatX)
        # item CF factors
        self.beta = self.init_std * \
            np.random.randn(n_items, self.n_components).astype(floatX)

    def _init_expo_params(self, N, n_users, n_items):
        # initialize exposure parameters
        # social-network exposure influence, which has the same sparsity
        # pattern as the network graph
        self.tau = sparse.csr_matrix(N)
        self.tau.data = self.init_std * \
            np.random.rand(*self.tau.data.shape).astype(floatX)
        self.tauT = self.tau.T.tocsr()
        # user exposure bias (the scale should be multiplied by 2, but we
        # remove it to compensate for the magnitude added from tau)
        scale = sqrt(-log(1 - self.init_mu))
        self.gamma = scale * np.random.rand(n_users, 1).astype(floatX)
        # item exposure bias
        self.alpha = scale * np.random.rand(n_items, 1).astype(floatX)

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
        self._init_expo_params(N, n_users, n_items)

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
            self._update_expo(X, XT, N)
            if self.verbose:
                print('\tUpdating user consideration factors: time=%.2f'
                      % (time.time() - start_t))
                sys.stdout.flush()

            if vad_data is not None:
                vad_ndcg = self._validate(X, vad_data, **kwargs)
                if self.early_stopping and self.vad_ndcg > vad_ndcg:
                    break  # we will not save the parameter for this iteration
                self.vad_ndcg = vad_ndcg

            if self.save_params:
                self._save_params(i)
        pass

    def _update_factors(self, X, XT):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating user CF factors...')
        self.theta = recompute_factors_ALS(self.beta, self.theta, self.tau,
                                           self.alpha, self.gamma, X,
                                           self.lam_theta / self.lam_y,
                                           self.lam_y,
                                           self.n_jobs, update_user=True,
                                           batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating user CF factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating item CF factors...')
        self.beta = recompute_factors_ALS(self.theta, self.beta, self.tauT,
                                          self.gamma, self.alpha, XT,
                                          self.lam_beta / self.lam_y,
                                          self.lam_y,
                                          self.n_jobs, update_user=False,
                                          batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating item CF factors: time=%.2f'
                  % (time.time() - start_t))
        pass

    def _update_expo(self, X, XT, N):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating network influence...')
        self.tau.data = update_tau(self.tau, self.alpha, self.gamma,
                                   self.beta, self.theta, X, N,
                                   self.lam_y, self.n_jobs,
                                   batch_size=self.batch_size)
        self.tauT = self.tau.T.tocsr()

        if self.verbose:
            print('\r\tUpdating network influence: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating user exposure bias...')
        self.gamma = recompute_factors_multi(self.tau, self.alpha, self.gamma,
                                             self.beta, self.theta, X,
                                             self.lam_y, self.n_jobs,
                                             batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating user exposure bias: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating item exposure bias...')
        self.alpha = recompute_factors_multi(self.tauT, self.gamma, self.alpha,
                                             self.theta, self.beta, XT,
                                             self.lam_y, self.n_jobs,
                                             batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating item exposure bias: time=%.2f'
                  % (time.time() - start_t))
        pass

    def _validate(self, X, vad_data, **kwargs):
        mu = dict(params=[self.tau, X, self.gamma, self.alpha],
                  func=get_mu)
        vad_ndcg = rec_eval.normalized_dcg_at_k(X, vad_data,
                                                self.theta,
                                                self.beta,
                                                mu=mu,
                                                **kwargs)
        if self.verbose:
            print('\tValidation NDCG@k: %.4f' % vad_ndcg)
            sys.stdout.flush()
        return vad_ndcg

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'SocialPoissonExpoMF_K%d_mu%.1e_iter%d.npz' % (
            self.n_components, self.init_mu, iter)
        np.savez(os.path.join(self.save_dir, filename), U=self.theta,
                 V=self.beta, tau=self.tau.data, gamma=self.gamma,
                 alpha=self.alpha)


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


def get_mu(tau, Y, gamma, alpha):
    '''
    \lambda_{ui} = \sum_{v \in N(u)} tau_{uv} y_{vi} + gamma_u + alpha_i
    \mu_{ui} = 1 - exp(-\lambda_{ui})

    When updating users:
        tau: (batch_users, n_users) sparse social network influence matrix
        Y: (n_users, n_items) sparse click matrix
        gamma: (batch_users, 1) per-user bias
        alpha: (n_items, 1) per-item bias
        return: (batch_users, n_items) dense ndarray

    When updating items:
        tau: (batch_items, n_users) sparse click matrix
        Y: (n_users, n_users) sparse social network influence matrix^T
        gamma: (batch_items, 1) per-item bias
        alpha: (n_users, 1) per-user bias
        return: (batch_items, n_users) dense ndarray

    '''
    return 1 - np.exp(-tau.dot(Y).toarray() + gamma + alpha.T)


def a_row_batch(nz_idx, theta_batch, beta, tau_batch, Y, gamma_batch, alpha,
                lam_y):
    '''Compute the posterior of exposure latent variables A by batch

    When updating users:
        theta_batch: (batch_users, n_components)
        beta: (n_items, n_components)
        tau_batch: (batch_users, n_users)
        Y: (n_users, n_items)
        gamma_batch: (batch_users, 1)
        alpha: (n_items, 1)

    When updating items:
        theta_batch: (batch_item, n_components)
        beta: (n_users, n_components)
        tau_batch: (batch_items, n_users)
        Y: (n_users, n_users) tau^T
        gamma_batch: (batch_items, 1)
        alpha: (n_users, 1)
    '''
    pEX = sqrt(lam_y / 2 * np.pi) * \
        np.exp(-lam_y * theta_batch.dot(beta.T)**2 / 2)
    mu = get_mu(tau_batch, Y, gamma_batch, alpha)
    A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
    A[nz_idx] = 1.
    return A


def _solve(k, A_k, X, Y, f, lam, lam_y):
    '''Update one single factor'''
    s_u, i_u = get_row(Y, k)
    a = np.dot(s_u * A_k[i_u], X[i_u])
    B = X.T.dot(A_k[:, np.newaxis] * X) + lam * np.eye(f)
    return LA.solve(B, a)


def _solve_batch(lo, hi, X, X_old_batch, S, alpha, gamma_batch, Y, m, f,
                 lam, lam_y, update_user):
    '''Update factors by batch, will eventually call _solve() on each factor to
    keep the parallel process busy

    When updating users:
        X: (n_items, n_components)
        X_old_batch: (batch_users, n_components)
        S: (n_users, n_users)
        alpha: (n_items, 1)
        gamma_batch: (batch_users, 1)
        Y: (n_users, n_items)

    When updating items:
        X: (n_users, n_components)
        X_old_batch: (batch_items, n_components)
        S: (n_users, n_users) social influence matrix^T
        alpha: (n_users, 1)
        gamma_batch: (batch_items, 1)
        Y: (n_items, n_users) click matrix^T
    '''
    assert X_old_batch.shape[0] == hi - lo
    X_batch = np.empty_like(X_old_batch, dtype=X_old_batch.dtype)

    if update_user:  # update user
        A_batch = a_row_batch(Y[lo:hi].nonzero(), X_old_batch, X, S[lo:hi], Y,
                              gamma_batch, alpha, lam_y)
    else:  # update item
        A_batch = a_row_batch(Y[lo:hi].nonzero(), X_old_batch, X, Y[lo:hi], S,
                              gamma_batch, alpha, lam_y)

    for ib, k in enumerate(xrange(lo, hi)):
        X_batch[ib] = _solve(k, A_batch[ib], X, Y, f, lam, lam_y)
    return X_batch


def recompute_factors_ALS(X, X_old, S, alpha, gamma, Y, lam, lam_y, n_jobs,
                          update_user=True, batch_size=1000):
    '''Regress X to Y with exposure matrix (computed on-the-fly with X_old) and
    ridge term lam by embarrassingly parallelization. All the comments below
    are in the view of computing user factors

    When updating users:
        X: item CF factors (beta: (n_items, n_components))
        X_old: user CF factors (theta: (n_users, n_components))
        S: social influence factors (tau: (n_users, n_users))
        alpha: item exposure bias (alpha: (n_items, 1))
        gamma: user exposure bias (gamma: (n_users, 1))
        Y: click matrix (X: (n_users, n_items))

    When updating items:
        X: user CF factors (theta: (n_users, n_components))
        X_old: item CF factors (beta: (n_items, n_components))
        S: social influence factors^T (tauT: (n_users, n_users))
        alpha: user exposure bias (gamma: (n_users, 1))
        gamma: item exposure bias (alpha: (n_items, 1))
        Y: click matrix^T (XT: (n_items, n_users))
    '''
    m, n = Y.shape  # m = number of users, n = number of items
    f = X.shape[1]  # f = number of factors

    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(delayed(_solve_batch)(
        lo, hi, X, X_old[lo:hi], S, alpha, gamma[lo:hi], Y, m, f, lam, lam_y,
        update_user)
        for lo, hi in zip(start_idx, end_idx))
    X_new = np.vstack(res)
    return X_new


def update_tau(S_old, alpha, gamma, beta, theta, Y, N, lam_y, n_jobs,
               batch_size=1000):
    m, _ = Y.shape[0]  # m: n_users
    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]

    res = Parallel(n_jobs=n_jobs)(delayed(_multi_tau_batch)(
        lo, hi, S_old[lo:hi], alpha, gamma[lo:hi], beta, theta[lo:hi],
        Y, N[lo:hi], lam_y)
        for lo, hi in zip(start_idx, end_idx))
    tau_new_data = np.hstack(res)
    return tau_new_data


def _multi_tau_batch(lo, hi, S_old_batch, alpha, gamma_batch, beta,
                     theta_batch, Y, N_batch, lam_y):
    assert S_old_batch.shape[0] == gamma_batch.shape[0] \
        == theta_batch.shape[0] == N_batch.shape[0] == hi - lo

    A_batch = a_row_batch(Y[lo:hi].nonzero(), theta_batch, beta, S_old_batch,
                          Y, gamma_batch, alpha, lam_y)
    Ah_batch = S_old_batch.dot(Y).toarray() + gamma_batch + alpha.T
    # Ah_batch.shape = (batch_users, n_items)

    rho_batch = A_batch * Ah_batch
    Et_batch = 1. / rho_batch - 1. / np.expm1(rho_batch)
    # for x very close to 0, we directly take the limit
    Et_batch[Et_batch < .5] = .5

    num = Y.dot((A_batch / Ah_batch).T).T
    den = Y.dot((1 - A_batch * (1 - Et_batch)).T).T
    S_batch_data = (num/den)[N_batch.nonzero()]
    return S_batch_data


def recompute_factors_multi(S, X, X_old, B, T, Y, lam_y, n_jobs,
                            update_user=True, batch_size=1000):
    '''
    multiplicative updates
    all the comments below are in the view of computing user factors

    When updating users bias:
        S: social influence (tau: (n_users, n_users))
        X: item exposure bias (alpha: (n_items, 1))
        X_old: user exposure bias (gamma: (n_users, 1))
        B: item CF factors (beta: (n_items, n_components))
        T: user CF factors (theta: (n_users, n_components))
        Y: click data (X: (n_users, n_items))
        return: X_new: (n_users, 1)

    When updating items bias:
        S: social influence^T (tauT: (n_users, n_users))
        X: user exposure bias (gamma: (n_users, 1))
        X_old: item exposure bias (alpha: (n_items, 1))
        B: user CF factors (theta: (n_users, n_components))
        T: item CF factors (beta: (n_items, n_components))
        Y: click data^T (YT: (n_items, n_users))
        return: X_new: (n_items, 1)
    '''
    m, _ = Y.shape[0]  # m: n_users
    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]

    res = Parallel(n_jobs=n_jobs)(delayed(_multi_batch)(
        lo, hi, S, X, X_old[lo:hi], B, T[lo:hi], Y, lam_y, update_user)
        for lo, hi in zip(start_idx, end_idx))
    X_new = np.vstack(res)
    return X_new


def _multi_batch(lo, hi, S, X, X_old_batch, B, T_batch, Y, lam_y, update_user):
    assert X_old_batch.shape[0] == T_batch.shape[0] == hi - lo

    if update_user:  # update user
        A_batch = a_row_batch(Y[lo:hi].nonzero(), T_batch, B, S[lo:hi], Y,
                              X_old_batch, X, lam_y)
        Ah_batch = S[lo:hi].dot(Y).toarray() + X_old_batch + X.T
        # Ah_batch.shape = (batch_users, n_items)
    else:
        A_batch = a_row_batch(Y[lo:hi].nonzero(), T_batch, B, Y[lo:hi], S,
                              X_old_batch, X, lam_y)
        Ah_batch = Y[lo:hi].dot(S).toarray() + X_old_batch + X.T
        # Ah_batch.shape = (batch_items, n_users)

    rho_batch = A_batch * Ah_batch
    Et_batch = 1. / rho_batch - 1. / np.expm1(rho_batch)
    # for x very close to 0, we directly take the limit
    Et_batch[Et_batch < .5] = .5

    X_batch = X_old_batch * (A_batch / Ah_batch).sum(axis=1, keepdims=True) \
        / (1 - A_batch * (1 - Et_batch)).sum(axis=1, keepdims=True)
    return X_batch
