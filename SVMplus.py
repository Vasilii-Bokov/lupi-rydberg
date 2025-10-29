import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
import warnings


class SVMplus(BaseEstimator, ClassifierMixin):
    """
    SVM+ (Support Vector Machine with Privileged Information) implementation
    
    This implementation follows the formulation from Vapnik's paper:
    "Learning Using Privileged Information: Similarity Control and Knowledge Transfer"
    """
    
    def __init__(self, C=1.0, gamma=1.0, kernel='rbf', kernel_star='rbf', 
                 kernel_params=None, kernel_star_params=None, tol=1e-6, max_iter=1000):
        """
        Initialize SVM+ classifier
        
        Parameters:
        -----------
        C : float, default=1.0
            Regularization parameter
        gamma : float, default=1.0
            Regularization parameter for privileged information
        kernel : str or callable, default='rbf'
            Kernel function for regular features
        kernel_star : str or callable, default='rbf'
            Kernel function for privileged features
        kernel_params : dict, default=None
            Parameters for regular kernel
        kernel_star_params : dict, default=None
            Parameters for privileged kernel
        tol : float, default=1e-6
            Tolerance for optimization
        max_iter : int, default=1000
            Maximum iterations for optimization
        """
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.kernel_star = kernel_star
        self.kernel_params = kernel_params or {}
        self.kernel_star_params = kernel_star_params or {}
        self.tol = tol
        self.max_iter = max_iter
        
    def _get_kernel_function(self, kernel_type, params):
        """Get kernel function based on type and parameters"""
        if kernel_type == 'linear':
            return lambda X, Y: linear_kernel(X, Y, **params)
        elif kernel_type == 'rbf':
            return lambda X, Y: rbf_kernel(X, Y, **params)
        elif kernel_type == 'poly':
            return lambda X, Y: polynomial_kernel(X, Y, **params)
        elif callable(kernel_type):
            return kernel_type
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def _compute_kernel_matrix(self, X, Y, kernel_func):
        """Compute kernel matrix between X and Y"""
        return kernel_func(X, Y)
    
    def fit(self, X, X_star, y):
        """
        Fit SVM+ model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Regular training features
        X_star : array-like, shape (n_samples, n_privileged_features)
            Privileged training features
        y : array-like, shape (n_samples,)
            Target values (-1 or 1)
        """
        X = np.array(X)
        X_star = np.array(X_star)
        y = np.array(y)
        
        # Ensure y is in {-1, 1}
        unique_y = np.unique(y)
        if set(unique_y) != {-1, 1}:
            # Convert to {-1, 1} if needed
            y_binary = np.where(y == unique_y[0], -1, 1)
            self.classes_ = unique_y
            y = y_binary
        else:
            self.classes_ = np.array([-1, 1])
        
        self.n_samples_ = X.shape[0]
        self.X_train_ = X
        self.X_star_train_ = X_star
        self.y_train_ = y
        
        # Get kernel functions
        self.kernel_func_ = self._get_kernel_function(self.kernel, self.kernel_params)
        self.kernel_star_func_ = self._get_kernel_function(self.kernel_star, self.kernel_star_params)
        
        # Compute kernel matrices
        self.K_ = self._compute_kernel_matrix(X, X, self.kernel_func_)
        self.K_star_ = self._compute_kernel_matrix(X_star, X_star, self.kernel_star_func_)
        
        # Solve dual optimization problem
        self._solve_dual()
        
        # Compute bias terms
        self._compute_bias()
        
        return self
    
    def _solve_dual(self):
        """Solve the dual optimization problem"""
        n = self.n_samples_
        
        # Initial guess for alpha and beta
        x0 = np.zeros(2 * n)
        
        # Bounds: alpha_i >= 0, 0 <= beta_i <= C
        bounds = [(0, None) for _ in range(n)] + [(0, self.C) for _ in range(n)]
        
        # Constraints
        constraints = [
            # Sum(alpha_i * y_i) = 0
            {
                'type': 'eq',
                'fun': lambda x: np.sum(x[:n] * self.y_train_),
                'jac': lambda x: np.concatenate([self.y_train_, np.zeros(n)])
            },
            # Sum(alpha_i - beta_i) = 0
            {
                'type': 'eq',
                'fun': lambda x: np.sum(x[:n] - x[n:]),
                'jac': lambda x: np.concatenate([np.ones(n), -np.ones(n)])
            }
        ]
        
        # Solve optimization problem
        result = minimize(
            fun=self._dual_objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Extract alpha and beta
        self.alpha_ = result.x[:n]
        self.beta_ = result.x[n:]
        
        # Find support vectors
        self.support_mask_ = self.alpha_ > self.tol
        self.support_ = np.where(self.support_mask_)[0]
        self.n_support_ = len(self.support_)
        
    def _dual_objective(self, x):
        """Dual objective function to minimize (negative of the original maximization)"""
        n = self.n_samples_
        alpha = x[:n]
        beta = x[n:]
        
        # First term: -sum(alpha_i)
        term1 = -np.sum(alpha)
        
        # Second term: 1/2 * sum_ij(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
        term2 = 0.5 * np.sum(alpha[:, np.newaxis] * alpha[np.newaxis, :] * 
                            self.y_train_[:, np.newaxis] * self.y_train_[np.newaxis, :] * 
                            self.K_)
        
        # Third term: 1/(2*gamma) * sum_ij((alpha_i - beta_i) * (alpha_j - beta_j) * K*(x*_i, x*_j))
        alpha_minus_beta = alpha - beta
        term3 = (1.0 / (2.0 * self.gamma)) * np.sum(
            alpha_minus_beta[:, np.newaxis] * alpha_minus_beta[np.newaxis, :] * self.K_star_)
        
        return term1 + term2 + term3
    
    def _compute_bias(self):
        """Compute bias terms b and b*"""
        if self.n_support_ == 0:
            self.b_ = 0.0
            self.b_star_ = 0.0
            return
        
        # For support vectors with 0 < alpha_i < C, we can compute b
        # yi * (sum(alpha_j * yj * K(xi, xj)) + b) = 1 - (w*, z*i) - b*
        
        # Find support vectors that are not on the boundary
        free_sv_mask = (self.alpha_ > self.tol) & (self.alpha_ < self.C - self.tol)
        free_sv_indices = np.where(free_sv_mask)[0]
        
        if len(free_sv_indices) > 0:
            # Use the first free support vector to compute b
            i = free_sv_indices[0]
            
            # Compute decision function without bias
            decision_no_bias = np.sum(self.alpha_ * self.y_train_ * self.K_[i, :])
            
            # Compute privileged part
            privileged_part = np.sum((self.alpha_ - self.beta_) * self.K_star_[i, :]) / self.gamma
            
            # b = yi * (1 - privileged_part) - decision_no_bias
            self.b_ = self.y_train_[i] * (1.0 - privileged_part) - decision_no_bias
            
            # For b*, we use the relation: xi = (w*, z*i) + b*
            # From the slack function definition
            self.b_star_ = privileged_part - np.sum((self.alpha_ - self.beta_) * self.K_star_[i, :]) / self.gamma
        else:
            # Fallback: use average over all support vectors
            self.b_ = 0.0
            self.b_star_ = 0.0
    
    def decision_function(self, X):
        """
        Compute the decision function for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        decision : array, shape (n_samples,)
            Decision function values
        """
        X = np.array(X)
        
        # Compute kernel between test and training samples
        K_test = self._compute_kernel_matrix(X, self.X_train_, self.kernel_func_)
        
        # Decision function: sum(alpha_i * y_i * K(x, x_i)) + b
        # K_test shape: (n_test_samples, n_train_samples)
        # alpha_ and y_train_ shape: (n_train_samples,)
        # We need to compute: K_test @ (alpha_ * y_train_)
        decision = K_test @ (self.alpha_ * self.y_train_) + self.b_
        
        return decision
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        decision = self.decision_function(X)
        predictions = np.sign(decision)
        
        # Convert back to original class labels if needed
        if hasattr(self, 'classes_') and len(self.classes_) == 2:
            predictions = np.where(predictions == -1, self.classes_[0], self.classes_[1])
        
        return predictions
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'C': self.C,
            'gamma': self.gamma,
            'kernel': self.kernel,
            'kernel_star': self.kernel_star,
            'kernel_params': self.kernel_params,
            'kernel_star_params': self.kernel_star_params,
            'tol': self.tol,
            'max_iter': self.max_iter
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for param, value in params.items():
            setattr(self, param, value)
        return self
    

class MulticlassSVMplus(BaseEstimator, ClassifierMixin):
    """
    Multiclass SVM+ using One-vs-One strategy
    
    Extends SVM+ to handle multiclass problems by training binary SVM+ classifiers
    for each pair of classes and using voting for prediction.
    """
    
    def __init__(self, C=1.0, gamma=1.0, kernel='rbf', kernel_star='rbf', 
                 kernel_params=None, kernel_star_params=None, tol=1e-6, max_iter=1000):
        """
        Initialize Multiclass SVM+ classifier
        
        Parameters:
        -----------
        C : float, default=1.0
            Regularization parameter
        gamma : float, default=1.0
            Regularization parameter for privileged information
        kernel : str or callable, default='rbf'
            Kernel function for regular features
        kernel_star : str or callable, default='rbf'
            Kernel function for privileged features
        kernel_params : dict, default=None
            Parameters for regular kernel
        kernel_star_params : dict, default=None
            Parameters for privileged kernel
        tol : float, default=1e-6
            Tolerance for optimization
        max_iter : int, default=1000
            Maximum iterations for optimization
        """
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.kernel_star = kernel_star
        self.kernel_params = kernel_params or {}
        self.kernel_star_params = kernel_star_params or {}
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, X, X_star, y):
        """
        Fit Multiclass SVM+ model using One-vs-One strategy
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Regular training features
        X_star : array-like, shape (n_samples, n_privileged_features)
            Privileged training features
        y : array-like, shape (n_samples,)
            Target values (multiclass labels)
        """
        X = np.array(X)
        X_star = np.array(X_star)
        y = np.array(y)
        
        # Store unique classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ == 2:
            # Binary case - use single SVM+
            self.binary_classifiers_ = [SVMplus(
                C=self.C, gamma=self.gamma, kernel=self.kernel, 
                kernel_star=self.kernel_star, kernel_params=self.kernel_params,
                kernel_star_params=self.kernel_star_params, tol=self.tol, max_iter=self.max_iter
            )]
            self.binary_classifiers_[0].fit(X, X_star, y)
            self.class_pairs_ = [(self.classes_[0], self.classes_[1])]
        else:
            # Multiclass case - One-vs-One strategy
            self.binary_classifiers_ = []
            self.class_pairs_ = []
            
            # Train binary classifier for each pair of classes
            for i in range(self.n_classes_):
                for j in range(i + 1, self.n_classes_):
                    class_i, class_j = self.classes_[i], self.classes_[j]
                    
                    # Get samples for this pair of classes
                    pair_mask = (y == class_i) | (y == class_j)
                    X_pair = X[pair_mask]
                    X_star_pair = X_star[pair_mask]
                    y_pair = y[pair_mask]
                    
                    # Convert to binary labels {-1, 1}
                    y_binary = np.where(y_pair == class_i, -1, 1)
                    
                    # Train binary SVM+ classifier
                    clf = SVMplus(
                        C=self.C, gamma=self.gamma, kernel=self.kernel,
                        kernel_star=self.kernel_star, kernel_params=self.kernel_params,
                        kernel_star_params=self.kernel_star_params, tol=self.tol, max_iter=self.max_iter
                    )
                    clf.fit(X_pair, X_star_pair, y_binary)
                    
                    self.binary_classifiers_.append(clf)
                    self.class_pairs_.append((class_i, class_j))
        
        return self
    
    def decision_function(self, X):
        """
        Compute decision function values for all class pairs
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        decision : array, shape (n_samples, n_classifiers)
            Decision function values for each binary classifier
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_classifiers = len(self.binary_classifiers_)
        
        decisions = np.zeros((n_samples, n_classifiers))
        
        for i, clf in enumerate(self.binary_classifiers_):
            decisions[:, i] = clf.decision_function(X)
            
        return decisions
    
    def predict(self, X):
        """
        Predict class labels using majority voting
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            # Binary case
            return self.binary_classifiers_[0].predict(X)
        
        # Multiclass case - voting
        votes = np.zeros((n_samples, self.n_classes_))
        
        for i, (clf, (class_i, class_j)) in enumerate(zip(self.binary_classifiers_, self.class_pairs_)):
            predictions = clf.predict(X)
            
            # Convert binary predictions back to original classes
            for k, pred in enumerate(predictions):
                if pred == -1:  # Predicted class_i
                    class_idx = np.where(self.classes_ == class_i)[0][0]
                    votes[k, class_idx] += 1
                else:  # Predicted class_j
                    class_idx = np.where(self.classes_ == class_j)[0][0]
                    votes[k, class_idx] += 1
        
        # Return class with most votes
        predicted_indices = np.argmax(votes, axis=1)
        return self.classes_[predicted_indices]
    
    def predict_proba(self, X):
        """
        Predict class probabilities using normalized votes
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        proba : array, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            # Binary case - use decision function
            decisions = self.binary_classifiers_[0].decision_function(X)
            # Convert to probabilities using sigmoid-like function
            proba_pos = 1 / (1 + np.exp(-decisions))
            proba_neg = 1 - proba_pos
            return np.column_stack([proba_neg, proba_pos])
        
        # Multiclass case - use votes as proxy for probabilities
        votes = np.zeros((n_samples, self.n_classes_))
        
        for i, (clf, (class_i, class_j)) in enumerate(zip(self.binary_classifiers_, self.class_pairs_)):
            decisions = clf.decision_function(X)
            
            # Use decision function values as confidence
            for k, decision in enumerate(decisions):
                if decision < 0:  # Favors class_i
                    class_idx = np.where(self.classes_ == class_i)[0][0]
                    votes[k, class_idx] += abs(decision)
                else:  # Favors class_j
                    class_idx = np.where(self.classes_ == class_j)[0][0]
                    votes[k, class_idx] += abs(decision)
        
        # Normalize to get probabilities
        vote_sums = votes.sum(axis=1, keepdims=True)
        vote_sums[vote_sums == 0] = 1  # Avoid division by zero
        probabilities = votes / vote_sums
        
        return probabilities
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'C': self.C,
            'gamma': self.gamma,
            'kernel': self.kernel,
            'kernel_star': self.kernel_star,
            'kernel_params': self.kernel_params,
            'kernel_star_params': self.kernel_star_params,
            'tol': self.tol,
            'max_iter': self.max_iter
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for param, value in params.items():
            setattr(self, param, value)
        return self




