import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pickle
import os
import json
import glob
import webbrowser

last_created_model = ""

class ExoplanetTabularCNN:
    def __init__(self, input_shape, num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.scaler = None
        self.feature_columns = None
    
    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Reshape((self.input_shape[0], 1)),
            
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        callbacks = [
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def evaluate(self, X_test, y_test):
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = np.mean(y_pred == y_test)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def predict(self, X):
        return self.model.predict(X)

    def _get_incremental_folder_name(self, base_filepath):
        # Base de todos los modelos
        model_base_dir = os.path.join('./', 'Modelos')
        os.makedirs(model_base_dir, exist_ok=True)
        
        # Nombre base 
        base_name = os.path.basename(base_filepath)
        
        pattern = os.path.join(model_base_dir, f"{base_name}_*")
        existing_folders = [f for f in glob.glob(pattern) if os.path.isdir(f)]
    
        # Encontrar número máximo
        max_number = 0
        for folder in existing_folders:
            folder_name = os.path.basename(folder)
            number_part = folder_name.replace(f"{base_name}_", "")
            if number_part.isdigit():
                max_number = max(max_number, int(number_part))
        
        new_number = max_number + 1
        new_folder_name = f"{base_name}_{new_number:03d}"
        
        return os.path.join(model_base_dir, new_folder_name)

    def save_model(self, filepath, create_folder=True):
        model_base_dir = os.path.join('../..', 'Modelos')
        os.makedirs(model_base_dir, exist_ok=True)
        folder_path = self._get_incremental_folder_name(filepath)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Carpeta creada:  {folder_path}")

        file_names = {
            'model': 'model.keras',
            'scaler': 'scaler.pkl', 
            'features': 'features.json',
            'metadata': 'metadata.json'
        }
        
        model_files = []
        
        # Guardar modelo
        model_path = os.path.join(folder_path, file_names['model'])
        self.model.save(model_path)
        model_files.append(model_path)
        last_created_model = model_path
        print(f"Modelo Keras guardado en: {model_path}")
        
        # Guardar scaler
        if self.scaler is not None:
            scaler_path = os.path.join(folder_path, file_names['scaler'])
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            model_files.append(scaler_path)
            print(f"Scaler guardado en: {scaler_path}")
        
        # Guardar features
        if self.feature_columns is not None:
            features_path = os.path.join(folder_path, file_names['features'])
            with open(features_path, 'w') as f:
                json.dump(self.feature_columns, f)
            model_files.append(features_path)
            print(f"Feature columns guardadas en: {features_path}")
        
        # Guardar metadata
        metadata = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'feature_columns': self.feature_columns
        }
        metadata_path = os.path.join(folder_path, file_names['metadata'])
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        model_files.append(metadata_path)
        print(f"Metadata guardada en: {metadata_path}")
        
        print("Modelo guardado exitosamente en la carpeta 'Modelos'!")
        return model_files, folder_path if create_folder else None



def load_and_preprocess_koi_data(csv_file_path):
    print("Cargando datos...")
    df = pd.read_csv(csv_file_path)
    
    print(f"Dataset original: {df.shape}")
    print(f"Columnas originales: {len(df.columns)}")
    
    columns_to_keep = [
        'loc_rowid', 'koi_disposition', 'koi_period', 'koi_period_err1', 'koi_period_err2',
        'koi_duration', 'koi_duration_err1', 'koi_duration_err2', 'koi_depth', 'koi_depth_err1', 'koi_depth_err2',
        'koi_ror', 'koi_ror_err1', 'koi_ror_err2', 'koi_srho', 'koi_srho_err1', 'koi_srho_err2',
        'koi_model_snr', 'koi_num_transits', 'koi_bin_oedp_sig', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2',
        'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2',
        'koi_smass', 'koi_smass_err1', 'koi_smass_err2', 'koi_jmag', 'koi_hmag', 'koi_kmag'
    ]
    
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    
    print(f"Columnas encontradas: {len(existing_columns)}")
    if missing_columns:
        print(f"Columnas faltantes: {missing_columns}")
    
    df = df[existing_columns]
    
    print(f"Dataset después de filtrar columnas: {df.shape}")
    
    X, y, feature_columns, scaler = preprocess_koi_data(df)
    
    print(f"\nDatos preprocesados - X: {X.shape}, y: {y.shape}")
    print(f"Proporción de clases: {np.bincount(y) / len(y)}")
    
    return X, y, feature_columns, scaler


def preprocess_koi_data(df):
    data = df.copy()
    
    print("Distribución original de koi_disposition:")
    print(data['koi_disposition'].value_counts())
    
    disposition_mapping = {
        'CONFIRMED': 1,
        'CANDIDATE': 0,
        'FALSE POSITIVE': 0,
        'NO DISPOSITIONED': 0
    }
    
    data['target'] = data['koi_disposition'].map(disposition_mapping)
    
    print("\nDistribución después del mapeo:")
    print(data['target'].value_counts())
    
    exclude_columns = ['loc_rowid', 'koi_disposition', 'target']
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    
    print(f"\nCaracterísticas utilizadas ({len(feature_columns)}): {feature_columns}")
    
    numerical_features = data[feature_columns].select_dtypes(include=[np.number]).columns
    
    print("\nValores missing por columna:")
    missing_info = data[feature_columns].isnull().sum()
    print(missing_info[missing_info > 0])
    
    imputer = SimpleImputer(strategy='median')
    data[numerical_features] = imputer.fit_transform(data[numerical_features])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[feature_columns])
    
    X = X_scaled
    y = data['target'].values
    
    return X, y, feature_columns, scaler


def main():
    with open('params.json', 'r') as f:
        data = json.load(f)
        BATCH_SIZE = data['batch_size']
        EPOCHS = data['epochs']
        TEST_SIZE = data['test_size']
        VAL_SIZE = data['val_size']

    try:
        X, y, feature_columns, scaler = load_and_preprocess_koi_data(data['dataset'])
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return
    
    # Dividir datos
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42, stratify=y_temp
    )
    
    print(f"\nDivisión de datos:")
    print(f"Entrenamiento: {X_train.shape}")
    print(f"Validación: {X_val.shape}")
    print(f"Prueba: {X_test.shape}")
    print(f"Balance en entrenamiento: {np.bincount(y_train) / len(y_train)}")
    print(f"Balance en prueba: {np.bincount(y_test) / len(y_test)}")
    
    input_shape = (X_train.shape[1],)
    cnn_model = ExoplanetTabularCNN(input_shape=input_shape, num_classes=2)
    cnn_model.scaler = scaler
    cnn_model.feature_columns = feature_columns
    
    cnn_model.compile_model(learning_rate=0.001)
    
    print("\nArquitectura del modelo:")
    cnn_model.model.summary()
    
    print("\nEntrenando modelo...")
    history = cnn_model.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    metrics = cnn_model.evaluate(X_test, y_test)
    
    print("\n" + "="*50)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("="*50)
    metricsjson = json.dumps(metrics, indent=4)
    print(metricsjson)

    with open('metrics.json', 'w') as f:
        f.write(metricsjson)
    if history is not None:
        # Primera figura: Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        plt.title('Accuracy durante el entrenamiento', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('accuracy_entrenamiento.png')  # Guardar primera imagen
        plt.show()

        # Segunda figura: Loss
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        plt.title('Loss durante el entrenamiento', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('loss_entrenamiento.png')  # Guardar segunda imagen
        plt.show()

    print("\nGUARDANDO MODELO...")
    
    cnn_model.save_model("exoplanet_cnn_model")
    
    print("Entrenamiento completado y modelo guardado!")

    webbrowser.open_new_tab("127.0.0.1:5001/ModGenFin")
    
    return cnn_model, history, metrics
