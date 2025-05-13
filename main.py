import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class DecisionTree:
    def __init__(self) -> None:
        """
        Initialize a Decision Tree
        """
        self.tree = None

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """
        Train the Decision Tree model
        
        Parameters:
        X: Training features
        y: Training labels
        """
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X:np.ndarray, y:np.ndarray):
        """
        Build Decision Tree recursively
        
        Parameters:
        X: Feature matrix
        y: Label array
        """
        # Check for leaf node condition: if all samples have the same label, return the label as a leaf node
        if len(set(y)) == 1:
            return y[0]
        
        # Find the optimal split attribute and split value
        split_attribute, split_value = self.find_split(X, y)

        # Based on the split value, samples that less than split value go to the left subtree
        left_indices = X[:, split_attribute] < split_value
        # Based on the split value, samples that greater than split value go to the left subtree
        right_indices = X[:, split_attribute] >= split_value

        # Recursively build the left and right subtrees with split data
        left_branch = self._build_tree(X[left_indices], y[left_indices])
        right_branch = self._build_tree(X[right_indices], y[right_indices])

        # Return the split results and subtrees as a tuple
        return (split_attribute, split_value, left_branch, right_branch)

    def find_split(self, X:np.ndarray, y:np.ndarray) -> tuple:
        """
        Find best split based on information gain
        
        Parameters:
        X: Feature matrix
        y: Label array
        """
        best_gain = 0.0  # Changed to float to fix type error
        best_attribute = None
        best_value = None
        
        # Iterate over each attribute in the dataset to find potential splits
        for attribute in range(X.shape[1]):

            values = np.unique(X[:, attribute])  # Get unique feature values for the current attribute
            
            # Iterate over each unique values to find the best split
            for value in values:

                # Split the data into left and right subsets and prepare for current split
                # This step can be replaced by sorting as described in the spec
                left_indices = X[:, attribute] < value
                right_indices = X[:, attribute] >= value

                # Skip invalid splits if one side has no data
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                # Calculate the information gain of the current split
                gain = self.information_gain(y, y[left_indices], y[right_indices])
                
                # Update the best gain and split parameters if current split is the best
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = attribute
                    best_value = value

        # Return the best split attribute and value
        return best_attribute, best_value

    def information_gain(self, parent:np.ndarray, left_child:np.ndarray, right_child:np.ndarray) -> float:
        """
        Calculate information gain for a split
        
        Parameters:
        parent: Parent node data
        left_child: Left child data
        right_child: Right child data
        """

        # Correspond to the \frac{ \abs{S_{left}} }{ \abs{s_left} + \abs{s_right} } term in spec
        p = float(len(left_child)) / len(parent)

        # Return the information gain of this split
        return self.entropy(parent) - p * self.entropy(left_child) - (1 - p) * self.entropy(right_child)

    def entropy(self, y:np.ndarray) -> float:
        """
        Calculate entropy of label distribution
        
        Parameters:
        y: Label array
        """

        # Count unique labels in y
        value, counts = np.unique(y, return_counts=True)

        # Calculate the probability of each label
        probabilities = counts / counts.sum()

        return -np.sum(probabilities * np.log(probabilities + 1e-10))

    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Make predictions using trained tree
        
        Parameters:
        X: Test features
        """
        predictions = []

        for sample in X:
            # Iterates over each test sample to get predictions
            predictions.append(self._predict_sample(sample, self.tree))

        return np.array(predictions)

    def _predict_sample(self, sample:np.ndarray, tree):
        """
        Predict single sample by traversing tree
        
        Parameters:
        sample: Single test instance
        tree: Current tree node
        """

        # If it's a leaf node, return its label
        if not isinstance(tree, tuple):
            return tree
        
        # If it's a tree node, unpack the split parameters and child branches
        attribute, value, left_branch, right_branch = tree

        # Determine whether to traverse the left or right branch based on the split parameters
        if sample[attribute] < value:
            return self._predict_sample(sample, left_branch)
        else:
            return self._predict_sample(sample, right_branch)

    def nested_cross_validation(self, X: np.ndarray, y: np.ndarray, outer_k: int = 10, inner_k: int = 10, use_pruning: bool = False) -> tuple:
        """
        Perform nested cross-validation
        
        Parameters:
        X: Feature matrix
        y: Label array
        outer_k: Number of outer folds
        inner_k: Number of inner folds
        use_pruning: Whether to use pruning
        """
        outer_fold_size = len(X) // outer_k   # Calculate number of data in each outer folds given outer k
        all_confusion_matrix = []   # Store the all confusion matrix
    
        best_f1 = -1
        best_tree = None

        for i in range(outer_k):  # Iterate over each outer fold to cross-validate model performance 
            val_indices = range(i * outer_fold_size, (i + 1) * outer_fold_size)
            train_indices = np.array([x for x in range(len(X)) if x not in val_indices])

            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[val_indices], y[val_indices]

            inner_fold_size = len(X_train) // inner_k   # Calculate number of data in each inner folds given inner k

            for j in range(inner_k):
                inner_val_indices = range(j * inner_fold_size, (j + 1) * inner_fold_size)
                inner_train_indices = np.array([x for x in range(len(X_train)) if x not in inner_val_indices])

                X_inner_train, y_inner_train = X_train[inner_train_indices], y_train[inner_train_indices]
                X_inner_val, y_inner_val = X_train[inner_val_indices], y_train[inner_val_indices]

                # Fit model in an inner fold
                self.fit(X_inner_train, y_inner_train)
            
                # Apply pruning if specified
                if use_pruning:
                    self.prune(X_inner_val, y_inner_val)

                # Evaluate this model in an inner fold
                y_pred = self.predict(X_test)

                inner_metrics = np.zeros((4, 4))
                for true, pred in zip(y_test, y_pred):
                    inner_metrics[int(true) - 1, int(pred) - 1] += 1
                _, _, inner_f1 = self.evaluate(inner_metrics)

                # Store the best Decision Tree by F1-score
                if inner_f1 > best_f1:
                    best_f1 = inner_f1
                    best_tree = deepcopy(self.tree)

                all_confusion_matrix.append(inner_metrics)

        # Calculate average confusion matrix
        avg_confusion_matrix = np.mean(all_confusion_matrix, axis=0)

        # Calculate average performance metrics
        precision, recall, f1 = self.evaluate(avg_confusion_matrix)

        self.tree = best_tree

        return precision, recall, f1, avg_confusion_matrix

    def evaluate(self, confusion_matrix:np.ndarray) -> tuple:
        """
        Calculate evaluation metrics
        
        Parameters:
        confusion_matrix: Confusion matrix of predictions
        """

        # The per-label metrics are calculated as follows
        # Here we add a small value (1e-10) to avoid nan (divide by 0)
        precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-10)
        recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Return a Macro-Averaging across labels (simple average of per-label metrics)
        return np.nanmean(precision), np.nanmean(recall), np.nanmean(f1)

    def plot_confusion_matrix(self, confusion_matrix) -> None:
        """
        Visualize confusion matrix
        
        Parameters:
        confusion_matrix: Matrix to plot
        """
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        # Dynamically determine ticks based on the shape of the confusion matrix
        tick_marks = np.arange(1, confusion_matrix.shape[0]+1)
        plt.xticks(tick_marks-1, tick_marks)
        plt.yticks(tick_marks-1, tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Adding formatted results into each entry of confusion matrix 
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], '.2f'),
                         ha='center', va='center',
                         color='white' if confusion_matrix[i, j] > np.max(confusion_matrix) / 2 else 'black')

        plt.tight_layout()
        plt.show()

    def plot_tree(self, node, depth:int=0, pos:tuple=None, ax:plt.Axes=None, width:float=2.0, vertical_gap:float=0.2):
        """
        Visualize decision tree structure
        
        Parameters:
        node: Current tree node
        depth: Current depth in tree
        pos: Position coordinates (x,y)
        ax: Matplotlib axes object
        width: Width between sibling nodes
        vertical_gap: Vertical space between levels
        """
        # Create a new image if no existed axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(60, 60))  # Set figure size
            ax.set_xlim(-2, 2)
            ax.set_ylim(0, 1)
            ax.axis('off')  # Remove coordinate axes for better visualization

        # Set position of figure
        if pos is None:
            pos = (0, 1) 

        if isinstance(node, tuple):
            attribute, value, left_branch, right_branch = node
            # Plot nodes and its values
            ax.text(pos[0], pos[1], f"[X{attribute} < {value:.2f}]", ha='center',
                    bbox=dict(facecolor='white', edgecolor='lightblue', boxstyle='round,pad=0.3'), color='black')

            # Set the position of left and right subtrees
            left_pos = (pos[0] - width / 2, pos[1] - vertical_gap)
            right_pos = (pos[0] + width / 2, pos[1] - vertical_gap)

            # Plot lines connecting nodes
            ax.plot([pos[0], left_pos[0]], [pos[1], left_pos[1]], color='black', lw=1)
            ax.plot([pos[0], right_pos[0]], [pos[1], right_pos[1]], color='black', lw=1)

            # Plot left and right subtrees recusively
            self.plot_tree(left_branch, depth + 1, pos=left_pos, ax=ax, width=width / 2, vertical_gap=vertical_gap)
            self.plot_tree(right_branch, depth + 1, pos=right_pos, ax=ax, width=width / 2, vertical_gap=vertical_gap)
        else:
            # Denote the leaf nodes as rect boxes with different colors
            ax.text(pos[0], pos[1], f"[Leaf: {node}]", ha='center',
                    bbox=dict(facecolor='lightgreen', edgecolor='lightblue', boxstyle='round,pad=0.3'), color='black')

        if depth == 0: # Depth = 0 means every node is plotted and returned 
            plt.tight_layout()
            plt.show()

    def prune(self, X_val:np.ndarray, y_val:np.ndarray):
        """
        Prune tree using validation data
        
        Parameters:
        X_val: Validation features
        y_val: Validation labels
        """
        # return number of misclassifications at a node
        def _get_node_error(node, X, y):
            predictions = np.array([self._predict_sample(x, node) for x in X])
            return np.sum(predictions != y)
    
        # return true if not a leaf
        def _is_decision_node(node):
            return isinstance(node, tuple)
    
        def _prune_node(node, X, y):
            # if node is already a leaf, return it
            if not _is_decision_node(node):
                return node
            
            # Recursively prune children first
            attribute, value, left, right = node
            left = _prune_node(left, X[X[:, attribute] < value], y[X[:, attribute] < value])
            right = _prune_node(right, X[X[:, attribute] >= value], y[X[:, attribute] >= value])
        
            # If both children are leaves, consider pruning this node
            if not _is_decision_node(left) and not _is_decision_node(right):
                # Calculate error before pruning
                error_before = _get_node_error(node, X, y)
            
                # Calculate error after pruning
                values, counts = np.unique(y, return_counts=True)
                majority_class = values[np.argmax(counts)]
                error_after = np.sum(y != majority_class)
            
                # Prune if it doesn't increase error
                if error_after <= error_before:
                    return majority_class
                
            return (attribute, value, left, right)

        self.tree = _prune_node(self.tree, X, y)


def load_data(filepath:str) -> tuple:
    """
    Load and preprocess data
    """
    data = np.loadtxt(filepath)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


if __name__ == "__main__":

    # Load datasets, change the path to your target dataset
    data = np.loadtxt('data/source/noisy_dataset.txt')

    # Separate data into features and labels and convert labels to integers
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    dt = DecisionTree()

    # Perform nested cross-validation
    precision, recall, f1, confusion_matrix = dt.nested_cross_validation(X, y)
    # Plot Decision Tree
    print('tree')
    dt.plot_tree(dt.tree)

    # Plot confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix)
    dt.plot_confusion_matrix(confusion_matrix)


    print('-' * 50)
    #dt.prune(X, y)
    precision, recall, f1, confusion_matrix = dt.nested_cross_validation(X, y, use_pruning=True)
    # Plot Decision Tree
    print('pruning tree')
    dt.plot_tree(dt.tree)

    # Plot confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix)
    dt.plot_confusion_matrix(confusion_matrix)
