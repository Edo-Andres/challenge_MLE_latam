import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List
import joblib
import numpy as np
from datetime import datetime

class DelayModel:

    def __init__(self):
        self._model = None  # Modelo inicializado como None
        self.FEATURES_COLS = None  # Atributo para almacenar las columnas esperadas

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepara los datos para entrenamiento o predicción.

        Args:
            data (pd.DataFrame): Datos crudos.
            target_column (str, optional): Si se especifica, devuelve el target.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Características y target.
            o
            pd.DataFrame: Solo las características.
        """
        # Crear variables dummy
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        # Asegurar que las columnas generadas coincidan con las esperadas
        if self.FEATURES_COLS is not None:
            features = features.reindex(columns=self.FEATURES_COLS, fill_value=0)

        # Validar si se necesita target_column
        if target_column:
            if target_column not in data.columns:
                # Calcular min_diff si no existe
                if 'Fecha-O' in data.columns and 'Fecha-I' in data.columns:
                    data['min_diff'] = data.apply(
                        lambda row: ((datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S') - 
                                      datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')).total_seconds()) / 60,
                        axis=1
                    )
                else:
                    raise KeyError("Las columnas 'Fecha-O' y 'Fecha-I' son necesarias para calcular 'delay'.")

                # Crear la columna delay
                print(f"El target_column '{target_column}' no está presente. Calculando dinámicamente.")
                data[target_column] = np.where(data['min_diff'] > 15, 1, 0)
            target = data[[target_column]]
            return features, target

        return features

    def load(self, file_path: str) -> None:
        """
        Carga un modelo entrenado desde un archivo.

        Args:
            file_path (str): Ruta al archivo desde donde se cargará el modelo.
        """
        file_path =  r'challenge\reg_model_2.pkl'
        self._model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")

        # Actualizar FEATURES_COLS con las columnas usadas en el modelo
        if hasattr(self._model, "feature_names_in_"):
            self.FEATURES_COLS = list(self._model.feature_names_in_)

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Entrena el modelo con los datos proporcionados.

        Args:
            features (pd.DataFrame): Características preprocesadas.
            target (pd.DataFrame): Variable objetivo.
        """
        self._model = LogisticRegression()
        self._model.fit(features, target.values.ravel())

        # Guardar las columnas usadas para el modelo
        self.FEATURES_COLS = list(features.columns)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predice los retrasos para nuevos vuelos.

        Args:
            features (pd.DataFrame): Datos preprocesados.
        
        Returns:
            List[int]: Predicciones.
        """
        if not self._model:
            raise ValueError("El modelo no ha sido entrenado o cargado.")
        
        return self._model.predict(features).tolist()
