import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving/loading the model
import numpy as np

class XGBoostClassifier:
    def __init__(self, params=None, n_classes=3):
        """
        Initialize the XGBoostClassifier class for multi-class classification.

        Parameters:
        - params (dict): Hyperparameters for XGBoost. If None, defaults are used.
        - test_size (float): Proportion of data to be used for testing.
        - random_state (int): Random seed for reproducibility.
        - n_classes (int): Number of target classes (e.g., 3 for pre-tremor, tremor, and control).
        """
        # Default parameters for XGBoost if none are provided
        if params is None:
            self.params = {
                'objective': 'multi:softmax',
                # NOTE: 'merror tracks the number of wrong predictions and 'mlogloss' tracks the probability of each class
                'eval_metric': ['merror', 'mlogloss'],
                'num_class': n_classes, 
                'max_depth': 6,
                'learning_rate': 0.1,
                'verbosity': 1,
            }
        else:
            self.params = params
        
        self.n_classes = n_classes
        self.model = None
    
    def fit(self, X_train, y_train, X_test, y_test, label_mapping=True):
        """
        Train the XGBoost model for multi-class classification. Saves after training.

        Parameters:
        - X_train (np.ndarray or pd.DataFrame): Training feature matrix (n_epochs, n_channels, n_features).
        - y_train (np.ndarray or pd.Series): Training target labels (n_epochs,).
        - X_test (np.ndarray or pd.DataFrame): Testing feature matrix (n_epochs, n_channels, n_features).
        - y_test (np.ndarray or pd.Series): Testing target labels (n_epochs,).
        - label_mapping (bool): Whether to encode the target labels using LabelEncoder. Default this to true
            due to the setup of the dataset. Set to False if the labels are already encoded.
        """

        assert self.n_classes == len(np.unique(y_train)), "Number of classes does not match the target labels."

        if label_mapping:
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train) # {1, 2, 3} -> {0, 1, 2}
            y_test = label_encoder.transform(y_test)

        # Flattening the feature matrix (n_epochs, n_channels * n_features) for input to the model
        n_epochs, n_channels, n_features = X_train.shape
        X_train_flat = X_train.reshape(n_epochs, n_channels * n_features)
        X_test_flat = X_test.reshape(X_test.shape[0], n_channels * n_features)
        
        # Convert data to DMatrix format, which is the format XGBoost prefers
        dtrain = xgb.DMatrix(X_train_flat, label=y_train)
        dtest = xgb.DMatrix(X_test_flat, label=y_test)
        
        # Train the XGBoost model
        self.model = xgb.train(self.params, dtrain, num_boost_round=100, evals=[(dtest, 'test')])
        
        # Predict on the test set
        y_pred = self.predict(X_test)
        
        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    
    def predict(self, X):
        """
        Make predictions using the trained XGBoost model.

        Parameters:
        - X (np.ndarray or pd.DataFrame): Feature matrix (n_epochs, n_channels, n_features) to predict on.

        Returns:
        - Predicted labels (np.ndarray): Predicted class labels (0, 1, or 2 for each epoch).
        """
        # Flatten the feature matrix (n_epochs, n_channels * n_features)
        n_epochs, n_channels, n_features = X.shape
        X_flat = X.reshape(n_epochs, n_channels * n_features)

        dtest = xgb.DMatrix(X_flat)
        
        return self.model.predict(dtest)
    
    def save_model(self, filename):
        """
        Save the trained model to a file.

        Parameters:
        - filename (str): Path where the model will be saved.
        """
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load a pre-trained model from a file.

        Parameters:
        - filename (str): Path to the saved model file.
        """
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.

        Returns:
        - Feature importance (np.ndarray).
        """
        if self.model is not None:
            return self.model.get_score(importance_type='weight')
        else:
            print("Model is not trained yet.")
            return None