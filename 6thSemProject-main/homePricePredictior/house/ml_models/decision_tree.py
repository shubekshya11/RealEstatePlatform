# import numpy as np
# import pandas as pd

# class TreeNode:
#     def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None, mse_reduction=0): #added mse_reduction=0
#         self.feature_index = feature_index
#         self.threshold = threshold
#         self.left = left
#         self.right = right
#         self.value = value
        
#         #added code
#         self.mse_reduction = mse_reduction  # Store MSE reduction for feature importance


# class DecisionTreeRegressor:
#     def __init__(self, max_depth=15, min_samples_split=2,min_samples_leaf=3,random_state=None):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf 
#         self.random_state = random_state
#         self.root = None

#         #added code
#         self.feature_importances_ = None  #Will store importance scores


#     #added code for normalizing importance values
#     def _normalize_importance(self):
#     ##Normalize feature importance scores to sum to 1. """
#         total_importance = np.sum(self.feature_importances_)
#         if total_importance > 0:
#             self.feature_importances_ /= total_importance
#     #added code ends here
    

#     def fit(self, X, y):
#         X = np.array(X)
#         y = np.array(y)

#         #added code
#         self.feature_importances_ = np.zeros(X.shape[1])  # Initialize feature importance array

#         self.root = self._build_tree(X, y, depth=0)

#         #added code
#         self.feature_importances_ /= np.sum(self.feature_importances_)  # Normalize
#         # Normalize feature importances
#         total_importance = np.sum(self.feature_importances_)
#         if total_importance > 0:
#             self.feature_importances_ /= total_importance  
#         #added code ends here

#     def _build_tree(self, X, y, depth):
#         num_samples, num_features = X.shape

#         if num_samples < self.min_samples_split or depth >= self.max_depth:
#             return TreeNode(value=np.mean(y))

#         best_feature, best_thresh, best_mse = None, None, float("inf")
#         best_splits = None
#         #added code
#         best_mse_reduction = 0  # Track best MSE reduction
#         mse_parent = np.var(y) * len(y)  # MSE before splitting


#         for feature_index in range(num_features):
#             thresholds = np.unique(X[:, feature_index])
#             for threshold in thresholds:
#                 left_mask = X[:, feature_index] <= threshold
#                 right_mask = X[:, feature_index] > threshold

#                 if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
#                     continue

#                 y_left = y[left_mask]
#                 y_right = y[right_mask]
#                 #commented code uncommented for feature importance (this was original code)
#                 # mse = np.var(y_left) * len(y_left) + np.var(y_right) * len(y_right)

#                 #added code
#                 mse_left = np.var(y_left) * len(y_left)
#                 mse_right = np.var(y_right) * len(y_right)
#                 mse_split = mse_left + mse_right

#                 mse_reduction = mse_parent - mse_split  # Calculate MSE reduction
#                 #added code ends here

# #ORIGINAL CODE
#                 # if mse < best_mse:
#                 #     best_mse = mse
#                 #     best_feature = feature_index
#                 #     best_thresh = threshold
#                 #     best_splits = (X[left_mask], y_left, X[right_mask], y_right)

#                 #added code
#                 if mse_split < best_mse:
#                     best_mse = mse_split
#                     best_feature = feature_index
#                     best_thresh = threshold
#                     best_splits = (X[left_mask], y_left, X[right_mask], y_right)
#                     best_mse_reduction = mse_reduction  # Store MSE reduction
#                 #added code ends here

#         # if best_feature is None:
#         #     return TreeNode(value=np.mean(y))
        
#         # added code --- Add MSE reduction to feature importance

#         if best_feature is None or best_splits is None:
#             return TreeNode(value=np.mean(y))
#         self.feature_importances_[best_feature] += best_mse_reduction  # Add MSE reduction to feature importance
#         # added code ends here

#         left_node = self._build_tree(best_splits[0], best_splits[1], depth + 1)
#         right_node = self._build_tree(best_splits[2], best_splits[3], depth + 1)
#         return TreeNode(feature_index=best_feature, threshold=best_thresh, left=left_node, right=right_node)

#     def _predict_sample(self, x, node):
#         if node.value is not None:
#             return node.value
#         if x[node.feature_index] <= node.threshold:
#             return self._predict_sample(x, node.left) 
#         else:
#             return self._predict_sample(x, node.right)

#     def predict(self, X):
#         X = np.array(X)
#         return np.array([self._predict_sample(x, self.root) for x in X])



import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None, mse_reduction=0):
        self.feature_index = feature_index  # Index of feature used for splitting
        self.threshold = threshold          # Threshold value for the split
        self.left = left                    # Left subtree
        self.right = right                  # Right subtree
        self.value = value                  # Value if leaf node
        self.mse_reduction = mse_reduction  # MSE reduction from this split

class DecisionTreeRegressor:
    def __init__(self, max_depth=15, min_samples_split=2, min_samples_leaf=3, random_state=None):
        self.max_depth = max_depth          # Maximum tree depth
        self.min_samples_split = min_samples_split  # Min samples to split
        self.min_samples_leaf = min_samples_leaf    # Min samples in leaf
        self.random_state = random_state    # Random seed
        self.root = None                    # Root of the tree
        self.feature_importances_ = None    # Feature importance scores
        self.floor_feature_index = None     # Track floor feature index

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.array(X)
        y = np.array(y)
        
        # Initialize feature importance array
        self.feature_importances_ = np.zeros(X.shape[1])
        
        # Try to identify floor feature automatically
        if isinstance(X, pd.DataFrame) and 'Floors' in X.columns:
            self.floor_feature_index = X.columns.get_loc('Floors')
        elif isinstance(X, np.ndarray):
            # Look for floor-related features in column names if available
            pass
            
        self.root = self._build_tree(X, y, depth=0)
        self._normalize_importance()
        
        return self

    def _normalize_importance(self):
        """Normalize feature importance scores to sum to 1."""
        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ /= total

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            num_samples < self.min_samples_split or 
            np.var(y) < 1e-6):
            return TreeNode(value=np.mean(y))

        best = {
            'feature': None,
            'threshold': None,
            'mse': float('inf'),
            'mse_reduction': 0,
            'splits': None
        }

        mse_parent = np.var(y) * len(y)
        
        # Special handling for floor feature if identified
        feature_indices = list(range(num_features))
        if self.floor_feature_index is not None:
            # Try floor feature first to encourage its selection
            feature_indices = [self.floor_feature_index] + [
                i for i in range(num_features) if i != self.floor_feature_index
            ]

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                
                # Check minimum samples in each child
                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                    
                y_left, y_right = y[left_mask], y[right_mask]
                mse = np.var(y_left) * len(y_left) + np.var(y_right) * len(y_right)
                mse_reduction = mse_parent - mse
                
                # Prefer splits with higher MSE reduction
                if mse < best['mse'] and mse_reduction > 0:
                    best.update({
                        'feature': feature_index,
                        'threshold': threshold,
                        'mse': mse,
                        'mse_reduction': mse_reduction,
                        'splits': (
                            X[left_mask], y_left,
                            X[right_mask], y_right
                        )
                    })

        if best['feature'] is None:
            return TreeNode(value=np.mean(y))
            
        # Update feature importance
        self.feature_importances_[best['feature']] += best['mse_reduction']
        
        # Build subtrees
        left_node = self._build_tree(
            best['splits'][0], best['splits'][1], depth + 1
        )
        right_node = self._build_tree(
            best['splits'][2], best['splits'][3], depth + 1
        )
        
        return TreeNode(
            feature_index=best['feature'],
            threshold=best['threshold'],
            left=left_node,
            right=right_node,
            mse_reduction=best['mse_reduction']
        )

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def get_feature_importance(self):
        """Return normalized feature importance scores."""
        return self.feature_importances_