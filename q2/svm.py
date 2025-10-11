# import cvxopt
# import numpy as np

# class SupportVectorMachine:
#     '''
#     Binary Classifier using Support Vector Machine
#     '''
#     def __init__(self):
#         pass
        
#     def fit(self, X, y, kernel = 'linear', C = 1.0, gamma = 0.001):
#         '''
#         Learn the parameters from the given training data
#         Classes are 0 or 1
        
#         Args:
#             X: np.array of shape (N, D) 
#                 where N is the number of samples and D is the flattened dimension of each image
                
#             y: np.array of shape (N,)
#                 where N is the number of samples and y[i] is the class of the ith sample
                
#             kernel: str
#                 The kernel to be used. Can be 'linear' or 'gaussian'
                
#             C: float
#                 The regularization parameter
                
#             gamma: float
#                 The gamma parameter for gaussian kernel, ignored for linear kernel
#         '''
#         pass

#     def predict(self, X):
#         '''
#         Predict the class of the input data
        
#         Args:
#             X: np.array of shape (N, D) 
#                 where N is the number of samples and D is the flattened dimension of each image
                
#         Returns:
#             np.array of shape (N,)
#                 where N is the number of samples and y[i] is the class of the
#                 ith sample (0 or 1)
#         '''
        
#         pass


import cvxopt
import numpy as np

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    def __init__(self):
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        self.b = None
        self.w = None
        self.kernel_type = None
        self.gamma = None
        self.X_train = None
        
    def _linear_kernel(self, x1, x2):
        """Linear kernel function"""
        return np.dot(x1, x2.T)
    
    def _gaussian_kernel(self, x1, x2):
        """Gaussian (RBF) kernel function"""
        if len(x1.shape) == 1:
            x1 = x1.reshape(1, -1)
        if len(x2.shape) == 1:
            x2 = x2.reshape(1, -1)
        
        # Compute pairwise squared Euclidean distances
        sq_dists = np.sum(x1**2, axis=1).reshape(-1, 1) + \
                   np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
        return np.exp(-self.gamma * sq_dists)
    
    def _compute_kernel_matrix(self, X1, X2):
        """Compute kernel matrix between X1 and X2"""
        if self.kernel_type == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel_type == 'gaussian':
            return self._gaussian_kernel(X1, X2)
        
    def fit(self, X, y, kernel='linear', C=1.0, gamma=0.001):
        '''
        Learn the parameters from the given training data
        Classes are 0 or 1
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
                
            kernel: str
                The kernel to be used. Can be 'linear' or 'gaussian'
                
            C: float
                The regularization parameter
                
            gamma: float
                The gamma parameter for gaussian kernel, ignored for linear kernel
        '''
        self.kernel_type = kernel
        self.gamma = gamma
        self.X_train = X
        
        # Convert labels to -1 and 1
        y = np.where(y == 0, -1, 1).astype(float)
        
        n_samples, n_features = X.shape
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(X, X)
        
        # Formulate QP problem in cvxopt format
        # min (1/2) * alpha^T * P * alpha + q^T * alpha
        # subject to: G * alpha <= h
        #            A * alpha = b
        
        # P matrix: P[i,j] = y_i * y_j * K(x_i, x_j)
        P = cvxopt.matrix(np.outer(y, y) * K)
        
        # q vector: all -1s
        q = cvxopt.matrix(-np.ones(n_samples))
        
        # Inequality constraints: -alpha_i <= 0 and alpha_i <= C
        G1 = np.eye(n_samples) * -1  # -alpha_i <= 0
        G2 = np.eye(n_samples)        # alpha_i <= C
        G = cvxopt.matrix(np.vstack([G1, G2]))
        
        h1 = np.zeros(n_samples)
        h2 = np.ones(n_samples) * C
        h = cvxopt.matrix(np.hstack([h1, h2]))
        
        # Equality constraint: sum(alpha_i * y_i) = 0
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(0.0)
        
        # Suppress cvxopt output
        cvxopt.solvers.options['show_progress'] = False
        
        # Solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Extract alphas
        alphas = np.array(solution['x']).flatten()
        
        # Support vectors have non-zero alphas (with tolerance)
        sv_threshold = 1e-5
        sv_indices = alphas > sv_threshold
        
        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        
        # Compute bias term b
        # Use support vectors with C > alpha > 0 for numerical stability
        margin_sv = (alphas > sv_threshold) & (alphas < C - sv_threshold)
        
        if np.sum(margin_sv) > 0:
            # Compute b using margin support vectors
            margin_indices = np.where(margin_sv)[0]
            b_values = []
            
            for idx in margin_indices:
                kernel_vals = self._compute_kernel_matrix(
                    X[sv_indices], X[idx].reshape(1, -1)
                ).flatten()
                b_val = y[idx] - np.sum(self.alphas * self.support_vector_labels * kernel_vals)
                b_values.append(b_val)
            
            self.b = np.mean(b_values)
        else:
            # Fallback: use all support vectors
            kernel_vals = self._compute_kernel_matrix(
                self.support_vectors, self.support_vectors[0].reshape(1, -1)
            ).flatten()
            self.b = self.support_vector_labels[0] - \
                     np.sum(self.alphas * self.support_vector_labels * kernel_vals)
        
        # For linear kernel, compute weight vector w
        if self.kernel_type == 'linear':
            self.w = np.sum(
                (self.alphas * self.support_vector_labels).reshape(-1, 1) * self.support_vectors,
                axis=0
            )

    def predict(self, X):
        '''
        Predict the class of the input data
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the
                ith sample (0 or 1)
        '''
        if self.kernel_type == 'linear':
            # For linear kernel: y = sign(w^T * x + b)
            decision = np.dot(X, self.w) + self.b
        else:
            # For non-linear kernel: y = sign(sum(alpha_i * y_i * K(x_i, x) + b))
            kernel_vals = self._compute_kernel_matrix(self.support_vectors, X)
            decision = np.sum(
                (self.alphas * self.support_vector_labels).reshape(-1, 1) * kernel_vals,
                axis=0
            ) + self.b
        
        # Convert to 0/1 labels
        predictions = np.where(decision >= 0, 1, 0)
        return predictions