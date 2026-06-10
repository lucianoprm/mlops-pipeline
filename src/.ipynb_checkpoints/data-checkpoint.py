import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_data(filepath: str) -> np.ndarray:
    """
    Carrega pontos lat/lng de um arquivo CSV.

    O CSV deve ter colunas 'lat' e 'lng'.
    Retorna um array numpy de shape (n, 2).

    Parâmetros:
        filepath: caminho para o arquivo CSV

    Retorna:
        np.ndarray de shape (n, 2)
    """
    # O Pandas vai receber o texto "data/raw/points.csv" enviado pelo main.py
    points = pd.read_csv(filepath, usecols=['lat', 'lng'])
    
    # Retorna a matriz numpy
    return points.to_numpy()


def prepare_features(points: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normaliza os features usando StandardScaler.

    Parâmetros:
        points: array numpy de shape (n, 2) com lat/lng

    Retorna:
        Tupla (features_normalizados, scaler)
        O scaler deve ser guardado para inverter a transformação depois.
    """
    scaler = StandardScaler()
    features = scaler.fit_transform(points)
    return features, scaler
