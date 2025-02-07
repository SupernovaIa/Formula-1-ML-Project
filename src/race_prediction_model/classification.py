# Data processing  
# -----------------------------------------------------------------------  
import pandas as pd  
import numpy as np  

# Visualization  
# -----------------------------------------------------------------------  
import seaborn as sns  
import matplotlib.pyplot as plt  

# Model training and evaluation  
# -----------------------------------------------------------------------  
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve
)
from sklearn.model_selection import train_test_split, GridSearchCV  
import xgboost as xgb  

# Utility modules  
# -----------------------------------------------------------------------  
import pickle  
import shap  
import time

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score



class ClassificationModels:
    """
    A class to manage the training, evaluation, and analysis of multiple classification models.
    """
    
    def __init__(self, df, tv, seed=42, train_prop=0.8):
        """
        Initializes the class with a dataset, splits it into training and testing sets, and sets up machine learning models for evaluation.

        Parameters:
        - df (pd.DataFrame): The dataset to be used for training and testing.
        - tv (str): The name of the target variable column in the dataset.
        - seed (int, optional): Random seed for reproducibility during dataset splitting. Defaults to 42.
        - train_prop (float, optional): Proportion of the dataset to be used for training. Defaults to 0.8.
        """

        # Dataset split
        self.df = df
        self.target_variable = tv
        self.X = df.drop(columns=tv)
        self.y = df[tv]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=train_prop, random_state=seed)

        # Models and results
        self.models = {
            "logistic_regression": LogisticRegression(),
            "tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "xgboost": xgb.XGBClassifier()
        }
        self.results = {model_name: {"best_model": None, "pred_train": None, "pred_test": None, "time": None} for model_name in self.models}
   

    def fit_model(self, model_name, param_grid=None, cross_validation=5, file_name='best_model'):
        """
        Fits a specified machine learning model using grid search for hyperparameter tuning, evaluates it, and saves the best model.

        Parameters:
        - model_name (str): The name of the model to fit. Must be one of the predefined models.
        - param_grid (dict, optional): Custom hyperparameter grid for grid search. If not provided, a default grid is used based on the model.
        - cross_validation (int, optional): Number of cross-validation folds. Defaults to 5.
        - file_name (str, optional): Name of the file to save the trained model. Defaults to 'best_model'.

        Raises:
        - ValueError: If the specified `model_name` is not found in the predefined models.

        Updates:
        - self.results[model_name]["time"] (float): Training time in seconds.
        - self.results[model_name]["best_model"] (object): The best model identified during grid search.
        - self.results[model_name]["pred_train"] (array): Predictions on the training set.
        - self.results[model_name]["pred_test"] (array): Predictions on the testing set.

        Saves:
        - Trained model to a pickle file in the '../model/' directory with the specified `file_name`.
        """

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]

        # Default params
        default_params = {
            "logistic_regression": {
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga'],
                'max_iter': [500, 1000]
            },
            "tree": {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            "gradient_boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 1.0]
            },
            "xgboost": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }

        if param_grid is None:
            param_grid = default_params.get(model_name, {})

        # Get training time
        start_time = time.time()

        # Model fit
        grid_search = GridSearchCV(estimator=model, 
                                   param_grid=param_grid, 
                                   cv=cross_validation,
                                   n_jobs=-1)
        
        grid_search.fit(self.X_train, self.y_train)

        # End training time
        elapsed_time = time.time() - start_time

        # Save results
        self.results[model_name]["time"] = elapsed_time
        self.results[model_name]["best_model"] = grid_search.best_estimator_
        self.results[model_name]["pred_train"] = grid_search.best_estimator_.predict(self.X_train)
        self.results[model_name]["pred_test"] = grid_search.best_estimator_.predict(self.X_test)

        # Save model
        with open(f'../model/{file_name}.pkl', 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)


    def get_metrics(self, model_name):
        """
        Computes and returns evaluation metrics for a specified model on both training and testing datasets.

        Parameters:
        - model_name (str): The name of the model for which to compute metrics. The model must already be fitted.

        Raises:
        - ValueError: If the specified `model_name` is not found in the results or if the model has not been fitted.

        Returns:
        - (pd.DataFrame): A DataFrame containing metrics for both training and testing data.
        """

        if model_name not in self.results:
            raise ValueError(f"'{model_name}' model not found.")
               
        pred_train = self.results[model_name]["pred_train"]
        pred_test = self.results[model_name]["pred_test"]

        if pred_train is None or pred_test is None:
            raise ValueError(f"Must fit '{model_name}' model before getting metrics.")
        
        model = self.results[model_name]["best_model"]

        if hasattr(model, "predict_proba"):
            self.results[model_name]["prob_train"] = model.predict_proba(self.X_train)[:, 1]
            self.results[model_name]["prob_test"] = model.predict_proba(self.X_test)[:, 1]
        else:
            self.results[model_name]["prob_train"] = self.results[model_name]["prob_test"] = None

        # Training data metrics
        metrics_train = {
            "accuracy": accuracy_score(self.y_train, pred_train),
            "precision": precision_score(self.y_train, pred_train),
            "recall": recall_score(self.y_train, pred_train),
            "f1": f1_score(self.y_train, pred_train),
            "kappa": cohen_kappa_score(self.y_train, pred_train),
            "auc": roc_auc_score(self.y_train, self.results[model_name]["prob_train"]) if self.results[model_name]["prob_train"] is not None else None,
            "average_precision": average_precision_score(self.y_train, self.results[model_name]["prob_train"]) if self.results[model_name]["prob_train"] is not None else None,
            "time_seconds": self.results[model_name]["time"]
        }

        # Test data metrics
        metrics_test = {
            "accuracy": accuracy_score(self.y_test, pred_test),
            "precision": precision_score(self.y_test, pred_test),
            "recall": recall_score(self.y_test, pred_test),
            "f1": f1_score(self.y_test, pred_test),
            "kappa": cohen_kappa_score(self.y_test, pred_test),
            "auc": roc_auc_score(self.y_test, self.results[model_name]["prob_test"]) if self.results[model_name]["prob_test"] is not None else None,
            "average_precision": average_precision_score(self.y_test, self.results[model_name]["prob_test"]) if self.results[model_name]["prob_test"] is not None else None,
            "time_seconds": self.results[model_name]["time"]
        }

        # Return metrics
        return pd.DataFrame({"train": metrics_train, "test": metrics_test}).T


    def plot_confusion_matrix(self, model_name, size=(8, 6)):
        """
        Plots the confusion matrix for a specified model on the testing dataset.

        Parameters:
        - model_name (str): The name of the model for which to plot the confusion matrix. The model must already be fitted.
        - size (tuple, optional): The size of the plot as (width, height). Defaults to (8, 6).

        Raises:
        - ValueError: If the specified `model_name` is not found in the results or if the model has not been fitted.

        Displays:
        - A heatmap of the confusion matrix with annotations for actual versus predicted values.
        """

        if model_name not in self.results:
            raise ValueError(f"'{model_name}' model not found.")

        pred_test = self.results[model_name]["pred_test"]

        if pred_test is None:
            raise ValueError(f"Must fit '{model_name}' before getting confusion matrix.")

        # Confusion matrix
        matrix = confusion_matrix(self.y_test, pred_test)
        
        plt.figure(figsize=size)
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues')
        plt.title(f"Confusion matrix ({model_name})")
        plt.xlabel("Prediction")
        plt.ylabel("Actual value")
        plt.show()


    def plot_predictors_importance(self, model_name, size=(10, 6)):
        """
        Plots the feature importance for a specified model and returns the importance values as a DataFrame.

        Parameters:
        - model_name (str): The name of the model for which to plot feature importance. The model must already be fitted.
        - size (tuple, optional): The size of the plot as (width, height). Defaults to (10, 6).

        Raises:
        - ValueError: If the specified `model_name` is not found in the results or if the model has not been fitted.

        Returns:
        - (pd.DataFrame): A DataFrame containing feature names and their corresponding importance values, sorted in descending order.

        Displays:
        - A horizontal bar plot of feature importances with features on the y-axis and importance values on the x-axis.
        """

        if model_name not in self.results:
            raise ValueError(f"'{model_name}' model not found.")
        
        model = self.results[model_name]["best_model"]

        if model is None:
            raise ValueError(f"Must fit '{model_name}' before getting confusion matrix.")
        
        # Check if feature importances are available
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif model_name == "logistic_regression" and hasattr(model, "coef_"):
            importance = model.coef_[0]
        else:
            print(f"'{model_name}' model does not support feature importances.")
            return
        
        # Plot
        importances_df = pd.DataFrame({
            "Feature": self.X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=size)
        sns.barplot(x="Importance", y="Feature", data=importances_df, palette="viridis")
        plt.title(f"Feature importance ({model_name})")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()

        return importances_df
    

    def plot_shap_summary(self, model_name):
        """
        Generates a SHAP summary plot to visualize feature contributions for a specified model.

        Parameters:
        - model_name (str): The name of the model for which to generate the SHAP summary plot. The model must already be fitted.

        Raises:
        - ValueError: If the specified `model_name` is not found in the results or if the model has not been fitted.

        Displays:
        - A SHAP summary plot showing the magnitude and direction of feature contributions to the model predictions.

        Notes:
        - Tree-based models (e.g., Decision Tree, Random Forest, Gradient Boosting, XGBoost) use the `shap.TreeExplainer`.
        - Other models use a generic `shap.Explainer`.
        - For multi-class models, SHAP values for the positive class are selected for visualization.
        """

        if model_name not in self.results:
            raise ValueError(f"'{model_name}' model not found.")

        model = self.results[model_name]["best_model"]

        if model is None:
            raise ValueError(f"Must fit '{model_name}' before getting shap plot.")

        # Use TreeExplainer for tree-based models
        if model_name in ["tree", "random_forest", "gradient_boosting", "xgboost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_test)

            # Check if the SHAP values have multiple classes (dimension 3).
            if isinstance(shap_values, list):
                # For binary models, select SHAP values for the positive class.
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                # For Decision Trees, select SHAP values for the positive class.
                shap_values = shap_values[:, :, 1]

        # Use generic explainer for other models
        else:
            explainer = shap.Explainer(model, self.X_test, check_additivity=False)
            shap_values = explainer(self.X_test).values

        # Generate summary plot
        shap.summary_plot(shap_values, self.X_test, feature_names=self.X.columns)


    def plot_roc_curve(self, model_name):
        """
        Plots the ROC curve for a specified model, comparing its performance to a random classifier and a perfect classifier.

        Parameters:
        - model_name (str): The name of the model for which the ROC curve will be plotted. It must match a key in `self.results`.

        Returns:
        - None: The function directly displays the ROC curve plot.
        """

        fpr, tpr, thresholds =  roc_curve(self.y_test, self.results[model_name]["prob_test"])
    
        plt.figure(figsize=(6,4))

        sns.lineplot(x = fpr, y = tpr, color = "deepskyblue", label = "ROC Curve")

        plt.fill_between(fpr, tpr, color = "deepskyblue", alpha = 0.2, interpolate=False, label = f'AUC : {self.get_metrics(model_name).loc['test']['auc']:.3f}')
        plt.plot([0,1],[0,1], color = "red", ls = "--", label = "Random Classifier")
        plt.plot([0,0,1], [0,1,1], color = "darkorange", lw = 1.5, label = "Perfect Classifier")

        plt.xlabel("False Positive Ratio")
        plt.ylabel("True Positive Ratio")
        plt.grid(ls = "--", lw = 0.6, alpha = 0.6)
        plt.title("ROC Curve")
        plt.legend()
        plt.show()




    def plot_precision_recall_curve(self, model_name):
        """
        Plots the Precision-Recall curve for a specified model, comparing its performance.

        Parameters:
        - model_name (str): The name of the model for which the Precision-Recall curve will be plotted.
                            It must match a key in `self.results`.

        Returns:
        - None: The function directly displays the Precision-Recall curve plot.
        """

        precision, recall, thresholds = precision_recall_curve(self.y_test, self.results[model_name]["prob_test"])
        
        plt.figure(figsize=(6,4))

        sns.lineplot(x=recall, y=precision, color="deepskyblue", label="Precision-Recall Curve")

        plt.fill_between(recall, precision, color="deepskyblue", alpha=0.2, interpolate=False, 
                        label=f'AP : {self.get_metrics(model_name).loc["test"]["average_precision"]:.3f}')

        plt.plot([0,1], [self.y_test.mean(), self.y_test.mean()], color="red", ls="--", 
                label="Random Classifier (Baseline)")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(ls="--", lw=0.6, alpha=0.6)
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()
