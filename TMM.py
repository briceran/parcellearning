# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:16:31 2017

@author: kristianeschenburg
"""

import numpy as np

import scipy
from scipy import special,optimize

from sklearn import neighbors,cluster


class StudentsTMixtureModel(object):
    
    """
    Class to represent a Student's T mixture model probability distribution
    This class estimates the parameters of the Student's-T mixture distribution.
    """
    
    def __init__(self,n_components = 1,covariance_type='full',
                 init_state='kmeans',random_state = np.random.RandomState(),
                 df=10,reg_covar=1e-6,tol=1e-3,fix_dof=True):
        
        self.covariance_type_ = covariance_type
        self.df_ = df
        self.fix_dof_ = fix_dof
        self.init_state_ = init_state
        self.n_components_ = n_components
        self.random_state_ = random_state
        self.reg_covar_ = reg_covar
        self.tol_ = tol
        
        self._check_parameters()
        
    def _check_parameters(self):
        
        """
        Check legality of parameters.
        """
        
        if self.n_components_ < 1 or self.n_components_ % 1 > 0:
            raise ValueError('n_components must be positive integer value.')
        
        if self.covariance_type_ not in ['full','tied']:
            raise ValueError('covariance_type is invalid.')

        if self.init_state_ not in ['kmeans','random']:
            raise ValueError('init_state type is invalid.')
            
        if self.reg_covar_ <= 0:
            raise ValueError('reg_covar must be greater than 0.')
            
        if self.df_ <1:
            raise ValueError('df must be greater than or equal to 1.')
        
        if not isinstance(self.fix_dof_,bool):
            raise ValueError('fix_dof must be boolean.')
            
    def fit(self,X):
        
        """
        Method to fit the Student's T Mixture Model.
        
        Parameters:
        - - - - -
            X : 
        """
        
        pass
    
    def _eStep(self,X):
        
        """
        Expectation step of the T-Mixture Model.
        """
        
        self.resp_ = updatePosteriors(X,self.mu_,self.mixtures_,
                                      self.covariances_,
                                      self.covariance_type_,
                                      self.df_,self.n_components_)
        
        self.weights_ = updateWeights(X,self.mu_,self.covariances_,
                                      self.covariance_type_,
                                      self.df_,self.n_components_)
        
    def _mStep(self,X):
        
        """
        Maximization step of the T-Mixture Model.
        """
        
        if np.ndim(X) == 1:
            X.shape += (1,)
        
        n_samples,n_dim = X.shape
        
        self.mixtures_ = updateMixtures(self.resp_)
        self.mu_ = updateMean(X,self.resp_,self.weights_)
        
        w = self.resp_ * self.weights_
        
        self.covariances_ = updateCovariances(X,self.mu_,w,
                                              self.covariance_type_,
                                              self.reg_covar_)
        
        # if we update the degrees of freedom at each iteration
        if not self.fix_dof_:
            self.df_ = updateDOF(self.df_,self.resp_,self.weights_,n_dim)
            
        
    def _initialize(self,X):
        
        """
        Initialize model parameters with either k-means or randomly.
        
        Parameters:
        - - - - -
            X : input data array
        """
        
        self.resp_ = self._initializePosteriors(X)
        self.mixtures_ = updateMixtures(self.resp_)
        
        _,mu,covariances = estimate_t_parameters(X,self.resp_,
                                                   self.covariance_type_,
                                                   self.reg_covar_)
        
        self.mu_ = mu
        self.covariances_ = covariances
        
        if isinstance(self.df_,int):
            self.df_ = np.squeeze(1.*np.ones(shape=(1,self.n_components_))*self.df_)
        
        self.weights_ = updateWeights(X,self.mu_,self.covariances_,
                                self.covariance_type_,self.df_,
                                self.n_components_)
        
    def _initializePosteriors(self,X):
        
        """
        Initialize posterior probabilities.
        
        Parameters:
        - - - - -
            X : input data array
        Returns:
            initialized posterior probabilities
        """
        
        n_samples, p = X.shape
        self.p_ = p
        
        if self.init_state_ == 'kmeans':
            resp = np.zeros(shape=(n_samples,self.n_components_))
            label = cluster.KMeans(n_clusters=self.n_components_,
                                   n_init=1).fit(X).labels_
            resp[np.arange(n_samples),label] = 1
        elif self.init_state_ == 'random':
            resp = self.random_state.rand(n_samples,self.n_components)
            resp /= resp.sum(axis=1)[:,np.newaxis]
            
        return resp

def _compute_precision_cholesky(covariances,covariance_type):
    
    """
    Method to compute the Cholesky factorization of a precision matrix.
    
    Parameters:
    - - - - -
    
        covariance : n x n covariance matrix
        covariance_type : indicator of type of matrix to expect -- options
                            'full' or 'tied'
    Returns:
    - - - -
        precisions_chol : cholesky factorization of the precision matrix
    """
    
    if covariance_type == 'full':
        n_components, n_features,_ = covariances.shape
        precisions_chol = np.empty((n_components,n_features,n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = scipy.linalg.cholesky(covariance,lower=True)
            except np.linalg.LinAlgError:
                raise ValueError('Cannot compute Cholesky factorization of ' \
                                 'covariance matrix.')
            precisions_chol[k] = scipy.linalg.solve_triangular(cov_chol,
                           np.eye(n_features),lower=True).T
                           
    elif covariance_type == 'tied':
        _,n_features = covariances.shape
        try:
            cov_chol = scipy.linalg.cholesky(covariance,lower=True)
        except np.linalg.LinAlgError:
            raise ValueError('Cannot compute Cholesky factorization of ' \
                            'covariance matrix.')
        precision_chol = scipy.linalg.solve_triangular(cov_chol,
                                                       np.eye(n_features),
                                                       lower=True).T
    else:
        print('covariance_type is incorrect.')
    
    return precision_chol
    

def estimate_t_parameters(X,resp,covariance_type,reg_covar):
    
    """
    Method to intialize the expected counts, means, and covariance matrices 
    for a T-Mixture Model.
    
    Parameters:
    - - - - -
    
    """
        
    nk = resp.sum(axis=0) + 10*np.finfo(resp.dtype).eps
    means = np.dot(resp.T,X) / nk[:,np.newaxis]
    
    covariances = {'full': estimate_t_covariances_full,
                   'tied': estimate_t_covariances_tied}[covariance_type](X,
                   resp,nk,means,reg_covar)

    return nk,means,covariances

def estimate_t_covariances_tied(X,resp,nk,means,reg_covar):
        
    """
    Estimate tied covariance values.
    
    Parameters:
    - - - - -
        X : input data array (n-samples by p-features)
        resp : estimated posterior probabilities (n-samples by k-components)
        nk : estimated count per component (1 by k-components)
        means : current estimate of components means (k-components by p-features)
        reg_covar : covariance regulatization
    
    Returns:
    - - - -
        covariance : tied estimate of covariance matrices
    """
    
    avg_X2 = np.dot(X.T,X)
    avg_means2 = np.dot(nk*means.T,means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[::len(covariance)+1] += reg_covar
    return covariance

def estimate_t_covariances_full(X,resp,nk,means,reg_covar):
        
    """
    Estimate full covariance values.
    
    Parameters:
    - - - - -
        X : input data array (n-samples by p-features)
        resp : estimated posterior probabilities (n-samples by k-components)
        nk : estimated count per component (1 by k-components)
        means : current estimate of components means (k-components by p-features)
        reg_covar : covariance regulatization
    
    Returns:
    - - - -
        covariance : full estimate of covariance matrices
    """
    
    n_components, n_features = means.shape
    covariance = np.empty((n_components,n_features,n_features))
    
    for k in range(n_components):
        diff = X-means[k]
        covariance[k] = np.dot(resp[:,k]*diff.T,diff)/nk[k]
        covariance[k].flat[::n_features+1] += reg_covar
    return covariance

def weighted_logLikelihood_t(X,mu,dof,mixtures,covariances,covariance_type):
    
    """
    Computes the weighted log-likelihood of the data with the current
    parameters.
    
    Parameters:
    - - - - -
        X : input data array (n-samples by p-features)
        mu : mean vectors for each component (k-components by p features)
        dof : current degrees of freedom estimates
        mixtures : mixing fractions
        covariances : current covariance matrix estimates
        covariance_type : 'full' or 'tied'
    """
    
    wLogLik = 0
    n_components,_ = mu.shape
    
    for k in n_components:
        
        mu_k = mu[k]
        dof_k = dof[k]
        mix_k = mixtures[k]
        
        if covariance_type == 'full':
            sigma = covariances[k]
        elif covariance_type == 'tied':
            sigma = covariances
            
        wLogLik += _component_logLikehood_t(X,mu_k,sigma,dof_k,mix_k)
        
    return wLogLik

def _component_logLikehood_t(X,mu,Sigma,dof,fraction):
    
    """
    Computes the loglikelihood of the data with the current parameters for a 
    single component.
    
    Parameters:
    - - - - -
        X : input data array (n-samples by p-features)
        mu : mean vector for single component
        Sigma : covariance matrix for single component
        dof : current degrees of freedom estimate for a single component
        fraction : mixing fraction for single component
    """
    
    return fraction*np.mean(np.log(multivariate_t_pdf(X,mu,Sigma,dof)))


def multivariate_t_pdf(X,mu,Sigma,dof):
    
    """
    Probability density function of the multivariate student's t distribution.
    
    Parameters:
    - - - - -
        X : input data vector or array (n-samples x p features)
        mu : mean vector (1 x p features)
        Sigma : covariance matrix estimate (p x p features)
        dof : degrees of freedom
        
    Returns:
    - - - -
        density : density of x2
    """
    
    X = np.atleast_2d(X)
    p = Sigma.shape[0]

    numerator = special.gamma(1.0 * (p + dof) / 2.0)

    denominator = (
            special.gamma(1.0 * dof / 2.0) * 
            np.power(dof * np.pi, 1.0 * p / 2.0) *  
            np.power(np.linalg.det(Sigma), 1.0 / 2.0) * 
            np.power(
                1.0 + (1.0 / dof) *
                np.diagonal(
                    np.dot( np.dot(X - mu, np.linalg.inv(Sigma)), (X - mu).T)
                ), 
                1.0 * (p + dof) / 2.0
                )
            )
                
    density = 1.*numerator/denominator

    return density

def updateCovariances(X,mu,z,covariance_type,reg_covar):
    
    """
    Method to update the covariance matrices.
    
    Parameters:
    - - - - -
        X : input data vector or array (n-samples x p features)
        mu : mean vector (1 x p features)
        z : posterior probability of each data observation for each 
                        component (n samples by k components)
        covariance_type : 'full' or 'tied'
        reg_covar : 
        
        
    """
    
    nk = np.sum(z,axis=0)
    
    covariances = {'full': estimate_t_covariances_full,
                   'tied': estimate_t_covariances_tied}[covariance_type](X,
                   z,nk,mu,reg_covar)
                    
    return covariances

def updateDOF(dof,z,u,p):
    
    """
    Solve for updated degrees of freedom for each component.
    
    Parameters:
    - - - - -
        dof : degrees of freedom
        z : posterior probability of each data observation for each 
                        component (n samples by k components)
        u : expected weight matrix (n samples by k components)
        p : data dimensionality
    """
    
    if np.ndim(z) == 1:
        z.shape += (1,)
        
    n_samples,n_components = z.shape
    updated = np.zeros(shape=(1,n_components))
    
    for k in np.arange(0,n_components):
        
        if n_components == 1:
            dof_k = dof
            z_k = z
            u_k = u
        else:
            dof_k = dof[k]
            z_k = z[:,k]
            u_k = u[:,k]
        
        updated[0,k] = _newtonsDOF(dof_k,p,z_k,u_k)

    return np.squeeze(updated)

def _newtonsDOF(vi,p,z,u):
        
        """
        Method to implement Newton's method to update the degrees of freedom
    
        Parameters:
        - - - - -
            vi : current estimate of degrees of freedom for given component
            p : data dimensionality
            z : posterior probability of samples in given component
            u : weights of samples in given component
        """
        
        t1 = np.log(1.*(vi+p)/2)
        t2 = np.dot(z.T,(np.log(u)-u))/np.sum(z,axis=0)
        t3 = scipy.special.digamma(1.*(vi+p)/2)
        
        rem = 1 - t1 + t2 + t3
        
        vts = optimize.newton(func=_order_0,x0=vi,
                              fprime=_order_1,fprime2=_order_2,args=(rem,))
        
        return vts
    
def _order_0(vi,rt):
    
    """
    0th order (original) function.
    """
    
    return np.log(vi) - np.log(2) + special.digamma(1.*vi/2) - rt

def _order_1(vi,rt):
    
    """
    First derivative.
    """
    
    return (1./vi) + (1./2)*special.polygamma(1,1.*vi/2)

def _order_2(vi,rt):
    
    """
    Second derivative.
    """
    
    return (-1./(vi**2)) + (1./4)*special.polygamma(2,(1.*vi/2))

def updateMean(X,z,weights):
        
        """
        Update array of component mean estimates.
        
        Parameters:
        - - - - -
            X : data vector or array (n samples by p features)
            z : array of posterior probabilities for each sample for each
                        component (n samples by k components)   
            weights : array of weights for each sample (n samples by k components)
            
        Returns:
        - - - -
            array of updated means (k components by p features)
            
        """
        
        scaled = z*weights
        
        numer = np.dot(scaled.T,X)
        denom = np.sum(scaled,axis=0)[:,np.newaxis]

        return 1.*numer/denom

def updateMixtures(z):
        
        """
        Update mixing fracture of each component.
        
        Parameters:
        - - - - -
            z : posterior probability of each data observation for each 
                        component (n samples by k components)
        Returns:
        - - - -
            mixing fractions for each component
        """
        
        return np.mean(z,axis=0)

def updatePosteriors(X,mu,pi,covariances,covariance_type,
                     dof,n_components):
        
    """
    Compute posterior probabilities of data observations.
    
    Parameters:
    - - - - -
        X : input data vector or array (n-samples x p-features)
        mu : array of means (n_components x p features)            
        pi : component probabilities
        covariances = covariance estimates
        covariance_type = 'tied' or 'full'
        dof : degrees of freedom
        n_components : number of mixture components

    Returns:
    - - - -        
        Z : posterior probability array (n-samples by n_components)
    """
    
    X = np.atleast_2d(X)
    r,_ = X.shape
    K = n_components
    
    Z = np.zeros(shape=(r,K))
    
    for k in np.arange(0,K):
        
        if n_components == 1:
            dof_k = dof
            mu_k = mu
            pi_k = pi
        else:
            mu_k = mu[k,:]
            dof_k = dof[k]
            pi_k = pi[k]
        
        if covariance_type == 'full':
            sigma = covariances[k]
        elif covariance_type == 'tied':
            sigma = covariances

        posteriors = pi_k*multivariate_t_pdf(X,mu_k,sigma,dof_k)
        Z[:,k] = list(posteriors)
    
    Z = Z/np.sum(Z,axis=1)[:,np.newaxis]
    
    return Z

def updateWeights(X,mu,covariances,covariance_type,dof,n_components):
        
    """
    Compute the expected weight for each observation.
    
    Parameters:
    - - - - -
    
        X : input data vector or array (n-samples by p features)
        mu : array of means (k components by p features)
        covariances : covariance estimates
        covariance_type : 'tied' or 'full'
        dof : degrees of freedom
        n_components : number of mixture components
        
    Returns:
    - - - -
        U : expected weight matrix (n samples by k components)
    """
    
    r,c = X.shape
    K = n_components
    U = np.zeros(shape=(r,K))
    
    for k in np.arange(0,K):
        
        if n_components == 1:
            dof_k = dof
            mu_k = mu
        else:
            dof_k = dof[k]
            
            mu_k = mu[k]
            mu_k.shape += (1,)
            
            [xs,ys] = mu_k.shape
            if xs > ys:
                mu_k = mu_k.T
            
        
        if covariance_type == 'full':
            sigma = covariances[k]
        elif covariance_type == 'tied':
            sigma = covariances

        p = np.linalg.inv(sigma)
        
        dist = neighbors.DistanceMetric.get_metric('mahalanobis',VI=p)
        delta = dist.pairwise(X,mu_k)
        
        numer = dof_k + c
        denom = dof_k + delta
        
        U[:,k] = list(1.*(numer)/denom)
        
    return U