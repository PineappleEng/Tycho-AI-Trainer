#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras


# In[2]:


class ExoplanetTabularCNN:
    def __init__(self, input_shape, num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.scaler = None
        self.feature_columns = None
    
    def predict(self, X):
        return self.model.predict(X)


# In[3]:


def load_saved_model(model_number):
    carpeta = f"Modelos/exoplanet_cnn_model_{model_number:03d}"
    
    model_path = os.path.join(carpeta, "model.keras")
    scaler_path = os.path.join(carpeta, "scaler.pkl")
    features_path = os.path.join(carpeta, "features.json")
    metadata_path = os.path.join(carpeta, "metadata.json")

    for file_path in [model_path, scaler_path, features_path, metadata_path]:
        if not os.path.exists(file_path):
            print(f"Archivo faltante: {file_path}")
            return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        model = ExoplanetTabularCNN(
            input_shape=tuple(metadata['input_shape']),
            num_classes=metadata['num_classes']
        )

        model.model = keras.models.load_model(model_path)

        with open(scaler_path, 'rb') as f:
            model.scaler = pickle.load(f)

        with open(features_path, 'r') as f:
            model.feature_columns = json.load(f)

        print(f"Modelo {model_number:03d} cargado correctamente desde '{carpeta}'")
        return model

    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None


# In[7]:


def predict_from_csv(csv_file_path, modelo_num=2, output_file="predicciones_exoplanetas.csv"):
    
    print("Cargando modelo entrenado...")
    loaded_model = load_saved_model(modelo_num)
    
    if loaded_model is None:
        print("No se pudo cargar el modelo.")
        return
    
    print("Modelo cargado.")
    
    print(f"\nCargando datos desde: {csv_file_path}")
    try:
        new_df = pd.read_csv(csv_file_path)
        print(f"Datos cargados: {new_df.shape}")
    except Exception as e:
        print(f"Error cargando el CSV: {e}")
        return
    
    columns_to_keep = loaded_model.feature_columns
    missing_columns = [col for col in columns_to_keep if col not in new_df.columns]
    
    if missing_columns:
        print(f"Columnas faltantes: {missing_columns}")
        return
    
    processed_df = new_df[columns_to_keep].copy()

    if processed_df.isnull().values.any():
        print("Valores faltantes detectados se rellenan con la mediana...")
        processed_df = processed_df.fillna(processed_df.median())
    
    print("Escalando datos...")
    try:
        new_data_scaled = loaded_model.scaler.transform(processed_df)
    except Exception as e:
        print(f"Error en el preprocesamiento: {e}")
        return
    
    print("Realizando predicciones...")
    try:
        predictions = loaded_model.predict(new_data_scaled)
        predicted_classes = np.argmax(predictions, axis=1)
        prediction_probs = predictions[:, 1]
    except Exception as e:
        print(f"Error en las predicciones: {e}")
        return
    
    results_df = pd.DataFrame({
        "Indice": np.arange(1, len(predicted_classes) + 1),
        "Probabilidad": prediction_probs,
        "Resultado": np.where(predicted_classes == 1, "Exoplaneta", "No exoplaneta")
    })

    try:
        results_df.to_csv(output_file, index=False)
        print(f"\nResultados guardados en: {output_file}")
    except Exception as e:
        print(f"Error guardando resultados: {e}")
    if (len(results_df) > 10):
        print("\nPrimeras 10 predicciones:")
    else:
        print("\nPredicciones:")
    print(results_df.head(10))
        
    return results_df


# Receives: csv path, model number
def evaluate(csv_path, model_num):
    if os.path.exists(csv_path):
        resultados = predict_from_csv(csv_path, model_num)
        if resultados is not None:
            print(f"\nPredicciones completadas. Revisa el archivo 'predicciones_exoplanetas.csv'")
    else:
        print("El archivo CSV no existe")

