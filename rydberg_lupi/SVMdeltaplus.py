import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
import warnings


class SVMdeltaplus(BaseEstimator, ClassifierMixin):
    """
    SVMdelta+ (SVM with Privileged Information for Similarity Control) implementation
    
    This implementation follows the formulation from section 4.1 of Vapnik's paper:
    "Learning Using Privileged Information: Similarity Control and Knowledge Transfer"
    
    SVMdelta+ learns a similarity measure using privileged information to control
    the metric in the decision space.
    """
    
    def __init__(self, C=1.0, gamma=1.0, delta=1.0, kernel='rbf', kernel_star='rbf', 
                 kernel_params=None, kernel_star_params=None, similarity_type='linear',
                 tol=1e-6, max_iter=1000):
        """
        Initialize SVMdelta+ classifier
        
        Parameters:
        -----------
        C : float, default=1.0
            Regularization parameter for margin violations
        gamma : float, default=1.0
            Regularization parameter for privileged information
        delta : float, default=1.0
            Regularization parameter for similarity control
        kernel : str or callable, default='rbf'
            Kernel function for regular features
        kernel_star : str or callable, default='rbf'
            Kernel function for privileged features  
        kernel_params : dict, default=None
            Parameters for regular kernel
        kernel_star_params : dict, default=None
            Parameters for privileged kernel
        similarity_type : str, default='linear'
            Type of similarity transformation ('linear', 'quadratic')
        tol : float, default=1e-6
            Tolerance for optimization
        max_iter : int, default=1000
            Maximum iterations for optimization
        """
        self.C = C
        self.gamma = gamma
        self.delta = delta
        self.kernel = kernel
        self.kernel_star = kernel_star
        self.kernel_params = kernel_params or {}
        self.kernel_star_params = kernel_star_params or {}
        self.similarity_type = similarity_type
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
    
    def _compute_similarity_matrix(self, K_star):
        """
        Compute similarity transformation matrix based on privileged information
        
        Parameters:
        -----------
        K_star : array-like, shape (n_samples, n_samples)
            Kernel matrix for privileged features
            
        Returns:
        --------
        S : array-like, shape (n_samples, n_samples)
            Similarity transformation matrix
        """
        n = K_star.shape[0]
        
        if self.similarity_type == 'linear':
            # Linear similarity: S_ij = exp(-delta * ||z*_i - z*_j||^2)
            # Using kernel trick: ||z*_i - z*_j||^2 = K*(z*_i, z*_i) + K*(z*_j, z*_j) - 2*K*(z*_i, z*_j)
            diag_K = np.diag(K_star)
            dist_matrix = diag_K[:, np.newaxis] + diag_K[np.newaxis, :] - 2 * K_star
            S = np.exp(-self.delta * dist_matrix)
            
        elif self.similarity_type == 'quadratic':
            # Quadratic similarity using privileged kernel directly
            # S_ij = K*(z*_i, z*_j) / sqrt(K*(z*_i, z*_i) * K*(z*_j, z*_j))
            diag_K = np.diag(K_star)
            normalizer = np.sqrt(diag_K[:, np.newaxis] * diag_K[np.newaxis, :])
            normalizer[normalizer == 0] = 1  # Avoid division by zero
            S = K_star / normalizer
            
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
            
        return S
    
    def fit(self, X, X_star, y):
        """
        Fit SVMdelta+ model
        
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
        
        # Compute similarity matrix
        self.S_ = self._compute_similarity_matrix(self.K_star_)
        
        # Apply similarity control to the decision kernel
        self.K_controlled_ = self.S_ * self.K_  # Element-wise multiplication
        
        # Solve dual optimization problem
        self._solve_dual()
        
        # Compute bias terms
        self._compute_bias()
        
        return self
    
    def _solve_dual(self):
        """Solve the dual optimization problem for SVMdelta+"""
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
        """
        Dual objective function for SVMdelta+ (negative of maximization problem)
        
        The objective incorporates similarity-controlled kernel matrix
        """
        n = self.n_samples_
        alpha = x[:n]
        beta = x[n:]
        
        # First term: -sum(alpha_i)
        term1 = -np.sum(alpha)
        
        # Second term: 1/2 * sum_ij(alpha_i * alpha_j * y_i * y_j * S_ij * K(x_i, x_j))
        # Using similarity-controlled kernel matrix
        term2 = 0.5 * np.sum(alpha[:, np.newaxis] * alpha[np.newaxis, :] * 
                            self.y_train_[:, np.newaxis] * self.y_train_[np.newaxis, :] * 
                            self.K_controlled_)
        
        # Third term: 1/(2*gamma) * sum_ij((alpha_i - beta_i) * (alpha_j - beta_j) * K*(x*_i, x*_j))
        alpha_minus_beta = alpha - beta
        term3 = (1.0 / (2.0 * self.gamma)) * np.sum(
            alpha_minus_beta[:, np.newaxis] * alpha_minus_beta[np.newaxis, :] * self.K_star_)
        
        return term1 + term2 + term3
    
    def _compute_bias(self):
        """Compute bias terms for SVMdelta+"""
        if self.n_support_ == 0:
            self.b_ = 0.0
            self.b_star_ = 0.0
            return
        
        # Find support vectors that are not on the boundary
        free_sv_mask = (self.alpha_ > self.tol) & (self.alpha_ < self.C - self.tol)
        free_sv_indices = np.where(free_sv_mask)[0]
        
        if len(free_sv_indices) > 0:
            # Use the first free support vector to compute b
            i = free_sv_indices[0]
            
            # Compute decision function without bias using similarity-controlled kernel
            decision_no_bias = np.sum(self.alpha_ * self.y_train_ * self.K_controlled_[i, :])
            
            # Compute privileged part
            privileged_part = np.sum((self.alpha_ - self.beta_) * self.K_star_[i, :]) / self.gamma
            
            # b = yi * (1 - privileged_part) - decision_no_bias
            self.b_ = self.y_train_[i] * (1.0 - privileged_part) - decision_no_bias
            
            # For b*, compute using the relation from the correcting function
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
        
        # For test samples, we need to compute similarity with training samples
        # This requires privileged information at test time, which is typically not available
        # In practice, we use the average similarity or identity transformation
        
        # Using identity similarity for test samples (standard approach when X_star not available)
        # Decision function: sum(alpha_i * y_i * K(x, x_i)) + b
        decision = K_test @ (self.alpha_ * self.y_train_) + self.b_
        
        return decision
    
    def decision_function_with_privileged(self, X, X_star):
        """
        Compute the decision function for samples with privileged information
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        X_star : array-like, shape (n_samples, n_privileged_features)
            Test privileged samples
            
        Returns:
        --------
        decision : array, shape (n_samples,)
            Decision function values with similarity control
        """
        X = np.array(X)
        X_star = np.array(X_star)
        
        # Compute kernel matrices between test and training samples
        K_test = self._compute_kernel_matrix(X, self.X_train_, self.kernel_func_)
        K_star_test = self._compute_kernel_matrix(X_star, self.X_star_train_, self.kernel_star_func_)
        
        # Compute similarity matrix between test and training samples
        n_test = X.shape[0]
        n_train = self.X_train_.shape[0]
        
        if self.similarity_type == 'linear':
            # Compute distances between test and training privileged features
            diag_test = np.sum(X_star**2, axis=1)  # Assuming RBF kernel for simplicity
            diag_train = np.diag(self.K_star_)
            
            # Distance matrix: ||z*_test_i - z*_train_j||^2
            dist_matrix = (diag_test[:, np.newaxis] + diag_train[np.newaxis, :] - 
                          2 * K_star_test)
            S_test = np.exp(-self.delta * dist_matrix)
            
        elif self.similarity_type == 'quadratic':
            # Normalized kernel similarity
            diag_test = np.sum(X_star**2, axis=1)  # Simplified for RBF
            diag_train = np.diag(self.K_star_)
            normalizer = np.sqrt(diag_test[:, np.newaxis] * diag_train[np.newaxis, :])
            normalizer[normalizer == 0] = 1
            S_test = K_star_test / normalizer
        
        # Apply similarity control
        K_controlled_test = S_test * K_test
        
        # Decision function with similarity control
        decision = np.sum(K_controlled_test * (self.alpha_ * self.y_train_)[np.newaxis, :], 
                         axis=1) + self.b_
        
        return decision
    
    def predict(self, X):
        """
        Predict class labels for samples in X (without privileged information)
        
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
    
    def predict_with_privileged(self, X, X_star):
        """
        Predict class labels for samples with privileged information
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        X_star : array-like, shape (n_samples, n_privileged_features)
            Test privileged samples
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels with similarity control
        """
        decision = self.decision_function_with_privileged(X, X_star)
        predictions = np.sign(decision)
        
        # Convert back to original class labels if needed
        if hasattr(self, 'classes_') and len(self.classes_) == 2:
            predictions = np.where(predictions == -1, self.classes_[0], self.classes_[1])
        
        return predictions
    
    def get_similarity_matrix(self):
        """
        Get the computed similarity matrix
        
        Returns:
        --------
        S : array, shape (n_samples, n_samples)
            Similarity matrix computed from privileged information
        """
        return self.S_
    
    def get_controlled_kernel_matrix(self):
        """
        Get the similarity-controlled kernel matrix
        
        Returns:
        --------
        K_controlled : array, shape (n_samples, n_samples)
            Kernel matrix with similarity control applied
        """
        return self.K_controlled_
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'C': self.C,
            'gamma': self.gamma,
            'delta': self.delta,
            'kernel': self.kernel,
            'kernel_star': self.kernel_star,
            'kernel_params': self.kernel_params,
            'kernel_star_params': self.kernel_star_params,
            'similarity_type': self.similarity_type,
            'tol': self.tol,
            'max_iter': self.max_iter
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for param, value in params.items():
            setattr(self, param, value)
        return self


class MulticlassSVMdeltaplus(BaseEstimator, ClassifierMixin):
    """
    Multiclass SVMdelta+ using One-vs-One strategy
    
    Extends SVMdelta+ to handle multiclass problems by training binary SVMdelta+ classifiers
    for each pair of classes.
    """
    
    def __init__(self, C=1.0, gamma=1.0, delta=1.0, kernel='rbf', kernel_star='rbf', 
                 kernel_params=None, kernel_star_params=None, similarity_type='linear',
                 tol=1e-6, max_iter=1000):
        """
        Initialize Multiclass SVMdelta+ classifier
        """
        self.C = C
        self.gamma = gamma
        self.delta = delta
        self.kernel = kernel
        self.kernel_star = kernel_star
        self.kernel_params = kernel_params or {}
        self.kernel_star_params = kernel_star_params or {}
        self.similarity_type = similarity_type
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, X, X_star, y):
        """
        Fit Multiclass SVMdelta+ model using One-vs-One strategy
        """
        X = np.array(X)
        X_star = np.array(X_star)
        y = np.array(y)
        
        # Store unique classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ == 2:
            # Binary case
            self.binary_classifiers_ = [SVMdeltaplus(
                C=self.C, gamma=self.gamma, delta=self.delta, kernel=self.kernel, 
                kernel_star=self.kernel_star, kernel_params=self.kernel_params,
                kernel_star_params=self.kernel_star_params, 
                similarity_type=self.similarity_type, tol=self.tol, max_iter=self.max_iter
            )]
            self.binary_classifiers_[0].fit(X, X_star, y)
            self.class_pairs_ = [(self.classes_[0], self.classes_[1])]
        else:
            # Multiclass case - One-vs-One strategy
            self.binary_classifiers_ = []
            self.class_pairs_ = []
            
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
                    
                    # Train binary SVMdelta+ classifier
                    clf = SVMdeltaplus(
                        C=self.C, gamma=self.gamma, delta=self.delta, kernel=self.kernel,
                        kernel_star=self.kernel_star, kernel_params=self.kernel_params,
                        kernel_star_params=self.kernel_star_params,
                        similarity_type=self.similarity_type, tol=self.tol, max_iter=self.max_iter
                    )
                    clf.fit(X_pair, X_star_pair, y_binary)
                    
                    self.binary_classifiers_.append(clf)
                    self.class_pairs_.append((class_i, class_j))
        
        return self
    
    def predict(self, X):
        """Predict class labels using majority voting (without privileged info)"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            return self.binary_classifiers_[0].predict(X)
        
        # Multiclass case - voting
        votes = np.zeros((n_samples, self.n_classes_))
        
        for i, (clf, (class_i, class_j)) in enumerate(zip(self.binary_classifiers_, self.class_pairs_)):
            predictions = clf.predict(X)
            
            for k, pred in enumerate(predictions):
                if pred == -1:
                    class_idx = np.where(self.classes_ == class_i)[0][0]
                    votes[k, class_idx] += 1
                else:
                    class_idx = np.where(self.classes_ == class_j)[0][0]
                    votes[k, class_idx] += 1
        
        predicted_indices = np.argmax(votes, axis=1)
        return self.classes_[predicted_indices]
    
    def predict_with_privileged(self, X, X_star):
        """Predict class labels using privileged information"""
        X = np.array(X)
        X_star = np.array(X_star)
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            return self.binary_classifiers_[0].predict_with_privileged(X, X_star)
        
        # Multiclass case - voting with privileged info
        votes = np.zeros((n_samples, self.n_classes_))
        
        for i, (clf, (class_i, class_j)) in enumerate(zip(self.binary_classifiers_, self.class_pairs_)):
            predictions = clf.predict_with_privileged(X, X_star)
            
            for k, pred in enumerate(predictions):
                if pred == -1:
                    class_idx = np.where(self.classes_ == class_i)[0][0]
                    votes[k, class_idx] += 1
                else:
                    class_idx = np.where(self.classes_ == class_j)[0][0]
                    votes[k, class_idx] += 1
        
        predicted_indices = np.argmax(votes, axis=1)
        return self.classes_[predicted_indices]
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'C': self.C,
            'gamma': self.gamma,
            'delta': self.delta,
            'kernel': self.kernel,
            'kernel_star': self.kernel_star,
            'kernel_params': self.kernel_params,
            'kernel_star_params': self.kernel_star_params,
            'similarity_type': self.similarity_type,
            'tol': self.tol,
            'max_iter': self.max_iter
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for param, value in params.items():
            setattr(self, param, value)
        return self