import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List
import joblib

class DelayModel:

    def __init__(self):
        # I
        self._model = None  

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepares the data for training or prediction.

        Args:
            data (pd.DataFrame): Raw data.
            target_column (str, optional): If specified, returns the target.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and target.
            or
            pd.DataFrame: Only the features.
        """
        # Create dummy variables
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)
        
        
        if hasattr(self._model, "feature_names_in_"):
            # Fill missing columns with zeros
            features = features.reindex(columns=self._model.feature_names_in_, fill_value=0)

        if target_column:
            target = data[target_column]
            return features, target

        return features

    def load(self, file_path: str) -> None:
        """
        Loads a trained model from a file.

        Args:
            file_path (str): Path to the file from which the model will be loaded.
        """
        self._model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predicts delays for new flights.

        Args:
            features (pd.DataFrame): Preprocessed data.
        
        Returns:
            List[int]: Predictions.
        """
        if not self._model:
            raise ValueError("The model is not loaded.")
        
        return self._model.predict(features).tolist()
