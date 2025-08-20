import pickle
import pandas as pd
import numpy as np
import os
import warnings
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from prophet.serialize import model_from_json
import json

warnings.filterwarnings('ignore')

# Clase del modelo LSTM (necesaria para cargar el modelo)
class LSTMModel(nn.Module):
    """Modelo LSTM usando PyTorch."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Capas LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Capa de salida
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        device = x.device
        # Inicializar estados ocultos
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward pass LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Tomar la última salida temporal
        out = out[:, -1, :]
        
        # Aplicar dropout
        out = self.dropout(out)
        
        # Capa lineal final
        out = self.linear(out)
        
        return out

def create_lstm_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Crea secuencias temporales para predicción LSTM.
    
    Args:
        data: Datos de entrada (features)
        sequence_length: Longitud de la secuencia temporal
        
    Returns:
        Array con secuencias para predicción
    """
    X = []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
    
    return np.array(X)

def hacer_predicciones_todos_productos(df_test_completo, modelo_mapping):
    """
    Hace predicciones para todos los productos usando sus mejores modelos.
    Ahora identifica el tipo de modelo a partir del nombre del archivo e incluye LSTM.
    
    Parameters:
    - df_test_completo: DataFrame con los datos de test
    - modelo_mapping: DataFrame con columnas 'id_producto' y 'mejor_modelo'
    
    Returns:
    - DataFrame con columnas: date, id_producto, demanda
    """
    
    # Lista para almacenar todas las predicciones
    todas_las_predicciones = []
    
    print("Iniciando predicciones para todos los productos...")
    print("-" * 60)
    
    # Iterar por cada producto en el mapeo
    for idx, row in modelo_mapping.iterrows():
        id_producto = row['id_producto']
        nombre_modelo = row['mejor_modelo']
        
        print(f"Procesando producto {id_producto} con modelo: {nombre_modelo}")
        
        try:
            # Construir la ruta completa del modelo
            model_path = os.path.join('Modelos registrados', nombre_modelo)
            
            # Verificar que el archivo del modelo existe
            if not os.path.exists(model_path):
                print(f"  ❌ ERROR: No se encontró el archivo {model_path}")
                continue
                
            # ==========================================
            # DETECCIÓN DEL TIPO DE MODELO POR NOMBRE Y EXTENSIÓN
            # ==========================================
            
            print(f"  🔍 Analizando archivo: {nombre_modelo}")
            print(f"      - Termina en .json: {nombre_modelo.endswith('.json')}")
            print(f"      - Contiene 'prophet': {'prophet' in nombre_modelo.lower()}")
            
            # Extraer el tipo de modelo del nombre del archivo
            # Formato esperado: best_model_producto_{id}_{tipo}.pkl o prophet_model_producto_{id}.json
            if nombre_modelo.endswith('.json') and 'prophet' in nombre_modelo.lower():
                tipo_modelo = 'prophet'
                print(f"  ✅ Detectado como: {tipo_modelo}")
            elif 'arima' in nombre_modelo.lower():
                tipo_modelo = 'arima'
                print(f"  ✅ Detectado como: {tipo_modelo}")
            elif 'ridge' in nombre_modelo.lower():
                tipo_modelo = 'ridge'
                print(f"  ✅ Detectado como: {tipo_modelo}")
            elif 'lasso' in nombre_modelo.lower():
                tipo_modelo = 'lasso'
                print(f"  ✅ Detectado como: {tipo_modelo}")
            elif 'lstm' in nombre_modelo.lower():
                tipo_modelo = 'lstm'
                print(f"  ✅ Detectado como: {tipo_modelo}")
            elif 'prophet' in nombre_modelo.lower():
                tipo_modelo = 'prophet'
                print(f"  ✅ Detectado como: {tipo_modelo} (fallback)")
            else:
                print(f"  ⚠️  WARNING: Tipo de modelo no reconocido en {nombre_modelo}")
                print(f"      Se intentará detectar por estructura interna...")
                tipo_modelo = 'desconocido'
            
            # ==========================================
            # CARGA DEL MODELO SEGÚN TIPO
            # ==========================================
            
            if tipo_modelo == 'prophet':
                # Prophet usa archivos JSON, no pickle
                model_dict = None  # No se usa para Prophet
                print(f"  📈 Modelo Prophet identificado, se cargará directamente desde JSON")
            else:
                # Cargar modelo pickle para otros tipos
                print(f"  📦 Cargando modelo pickle para tipo: {tipo_modelo}")
                with open(model_path, 'rb') as file:
                    model_dict = pickle.load(file)
            
            # Filtrar datos para el producto actual
            df_test_producto = df_test_completo[df_test_completo['id_producto'] == id_producto].copy()
            
            if len(df_test_producto) == 0:
                print(f"  ⚠️  WARNING: No hay datos de test para el producto {id_producto}")
                continue
            
            # ==========================================
            # PROCESAMIENTO SEGÚN TIPO DE MODELO
            # ==========================================
            
            if tipo_modelo in ['ridge', 'lasso']:
                # ==========================================
                # MODELO LINEAL (Ridge/Lasso)
                # ==========================================
                print(f"  🔧 Procesando modelo lineal ({tipo_modelo.upper()})")
                
                # Extraer el pipeline del diccionario
                if 'pipeline' in model_dict:
                    model_pipeline = model_dict['pipeline']
                else:
                    print(f"  ❌ ERROR: No se encontró 'pipeline' en modelo {tipo_modelo}")
                    continue
                
                # Preparar las características (X) para predicción - quitar date e id_producto
                X_test_producto = df_test_producto.drop(columns=['id_producto', 'date'])
                
                # Hacer predicciones
                predicciones = model_pipeline.predict(X_test_producto)
                
                # Crear DataFrame con los resultados
                resultado_producto = pd.DataFrame({
                    'date': df_test_producto['date'].values,
                    'id_producto': id_producto,
                    'demanda': predicciones
                })
                
                print(f"  ✅ Predicciones {tipo_modelo} completadas: {len(predicciones)} registros")
                
            elif tipo_modelo == 'arima':
                # ==========================================
                # MODELO ARIMA
                # ==========================================
                print(f"  🎯 Procesando modelo ARIMA")
                
                # Extraer el modelo ARIMA entrenado
                if 'trained_model' in model_dict:
                    arima_model = model_dict['trained_model']
                else:
                    print(f"  ❌ ERROR: No se encontró 'trained_model' en modelo ARIMA")
                    continue
                
                # Para ARIMA, necesitamos el número de periodos a predecir
                n_periods = len(df_test_producto)
                
                # Hacer predicciones ARIMA (out-of-sample)
                predicciones_arima, conf_int = arima_model.predict(
                    n_periods=n_periods, 
                    return_conf_int=True
                )
                
                # Crear DataFrame con los resultados
                resultado_producto = pd.DataFrame({
                    'date': df_test_producto['date'].values,
                    'id_producto': id_producto,
                    'demanda': predicciones_arima
                })
                
                print(f"  ✅ Predicciones ARIMA completadas: {len(predicciones_arima)} registros")
                
            elif tipo_modelo == 'lstm':
                # ==========================================
                # MODELO LSTM
                # ==========================================
                print(f"  🧠 Procesando modelo LSTM")
                
                try:
                    # Extraer información del modelo LSTM
                    if not all(key in model_dict for key in ['trained_model', 'model_architecture', 'scaler_X', 'scaler_y', 'sequence_length', 'feature_columns']):
                        print(f"  ❌ ERROR: Estructura incompleta en modelo LSTM")
                        continue
                    
                    # Extraer componentes del modelo
                    model_state_dict = model_dict['trained_model']
                    model_architecture = model_dict['model_architecture']
                    scaler_X = model_dict['scaler_X']
                    scaler_y = model_dict['scaler_y']
                    sequence_length = model_dict['sequence_length']
                    feature_columns = model_dict['feature_columns']
                    
                    # Verificar que tenemos las columnas necesarias
                    missing_cols = [col for col in feature_columns if col not in df_test_producto.columns]
                    if missing_cols:
                        print(f"  ❌ ERROR: Faltan columnas en datos de test: {missing_cols}")
                        continue
                    
                    # Preparar datos de test
                    X_test_producto = df_test_producto[feature_columns].values
                    
                    # Verificar que tenemos suficientes datos para crear secuencias
                    if len(X_test_producto) <= sequence_length:
                        print(f"  ❌ ERROR: Datos insuficientes para crear secuencias LSTM ({len(X_test_producto)} < {sequence_length})")
                        continue
                    
                    # Escalar los datos usando el scaler entrenado
                    X_test_scaled = scaler_X.transform(X_test_producto)
                    
                    # Crear secuencias temporales
                    X_test_sequences = create_lstm_sequences(X_test_scaled, sequence_length)
                    
                    # Recrear el modelo LSTM con la arquitectura guardada
                    lstm_model = LSTMModel(
                        input_size=model_architecture['input_size'],
                        hidden_size=model_architecture['hidden_size'],
                        num_layers=model_architecture['num_layers'],
                        dropout_rate=model_architecture['dropout_rate']
                    )
                    
                    # Cargar los pesos entrenados
                    lstm_model.load_state_dict(model_state_dict)
                    lstm_model.eval()
                    
                    # Hacer predicciones
                    with torch.no_grad():
                        # Convertir a tensor
                        X_test_tensor = torch.FloatTensor(X_test_sequences)
                        
                        # Hacer predicción
                        predictions_scaled = lstm_model(X_test_tensor)
                        predictions_scaled = predictions_scaled.squeeze().numpy()
                        
                        # Asegurar que sea un array 1D
                        if predictions_scaled.ndim == 0:
                            predictions_scaled = np.array([predictions_scaled])
                    
                    # Desescalar las predicciones
                    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
                    
                    # Ajustar las fechas (perdemos sequence_length observaciones al inicio)
                    dates_ajustadas = df_test_producto['date'].iloc[sequence_length:].values
                    
                    # Verificar que las dimensiones coincidan
                    if len(predictions) != len(dates_ajustadas):
                        print(f"  ⚠️  WARNING: Dimensiones no coinciden - predicciones: {len(predictions)}, fechas: {len(dates_ajustadas)}")
                        # Tomar el mínimo
                        min_len = min(len(predictions), len(dates_ajustadas))
                        predictions = predictions[:min_len]
                        dates_ajustadas = dates_ajustadas[:min_len]
                    
                    # Crear DataFrame con los resultados
                    resultado_producto = pd.DataFrame({
                        'date': dates_ajustadas,
                        'id_producto': id_producto,
                        'demanda': predictions
                    })
                    
                    print(f"  ✅ Predicciones LSTM completadas: {len(predictions)} registros")
                    
                except Exception as lstm_error:
                    print(f"  ❌ ERROR específico LSTM: {str(lstm_error)}")
                    continue
                    
            elif tipo_modelo == 'prophet':
                # ==========================================
                # MODELO PROPHET
                # ==========================================
                print(f"  📈 Procesando modelo Prophet")
                
                try:
                    # Para Prophet, el archivo es JSON, no pickle
                    with open(model_path, 'r') as file:
                        model_json = file.read()
                    
                    # Deserializar el modelo Prophet
                    prophet_model = model_from_json(model_json)
                    
                    # Para Prophet, necesitamos crear un dataframe con fechas futuras
                    # Prophet requiere columna 'ds' para fechas
                    future_dates = pd.DataFrame({
                        'ds': df_test_producto['date'].values
                    })
                    
                    # Agregar variables exógenas si las hay en el modelo
                    # Verificar qué regressors tiene el modelo Prophet
                    if hasattr(prophet_model, 'extra_regressors') and prophet_model.extra_regressors:
                        regressor_names = list(prophet_model.extra_regressors.keys())
                        print(f"    🔧 Modelo Prophet con regressors: {regressor_names}")
                        
                        # Agregar regressors disponibles
                        for regressor in regressor_names:
                            if regressor in df_test_producto.columns:
                                future_dates[regressor] = df_test_producto[regressor].values
                                print(f"    ✅ Agregado regressor: {regressor}")
                            else:
                                # Si no está disponible, usar un valor por defecto
                                print(f"    ⚠️  Regressor {regressor} no disponible, usando valor por defecto")
                                future_dates[regressor] = 0  # O algún valor por defecto apropiado
                    
                    # Hacer predicciones con Prophet
                    forecast = prophet_model.predict(future_dates)
                    
                    # Extraer las predicciones (columna 'yhat')
                    predicciones_prophet = forecast['yhat'].values
                    
                    # Crear DataFrame con los resultados
                    resultado_producto = pd.DataFrame({
                        'date': df_test_producto['date'].values,
                        'id_producto': id_producto,
                        'demanda': predicciones_prophet
                    })
                    
                    print(f"  ✅ Predicciones Prophet completadas: {len(predicciones_prophet)} registros")
                    
                except Exception as prophet_error:
                    print(f"  ❌ ERROR específico Prophet: {str(prophet_error)}")
                    continue
                    
            else:
                # ==========================================
                # DETECCIÓN POR ESTRUCTURA (FALLBACK)
                # ==========================================
                print(f"  🔍 Intentando detección por estructura interna...")
                
                if model_dict is None:
                    print(f"  ❌ ERROR: No se pudo cargar el modelo para análisis de estructura")
                    continue
                
                if 'pipeline' in model_dict:
                    # Modelo lineal
                    print(f"  🔧 Detectado modelo lineal por estructura")
                    
                    model_pipeline = model_dict['pipeline']
                    X_test_producto = df_test_producto.drop(columns=['id_producto', 'date'])
                    predicciones = model_pipeline.predict(X_test_producto)
                    
                    resultado_producto = pd.DataFrame({
                        'date': df_test_producto['date'].values,
                        'id_producto': id_producto,
                        'demanda': predicciones
                    })
                    
                    print(f"  ✅ Predicciones lineales completadas: {len(predicciones)} registros")
                    
                elif 'trained_model' in model_dict and 'model_architecture' in model_dict:
                    # Posible modelo LSTM por estructura
                    print(f"  🧠 Detectado posible modelo LSTM por estructura")
                    # Redirigir al procesamiento LSTM
                    tipo_modelo = 'lstm'
                    continue  # Esto hará que se procese como LSTM en la siguiente iteración
                    
                elif 'trained_model' in model_dict:
                    # Modelo ARIMA
                    print(f"  🎯 Detectado modelo ARIMA por estructura")
                    
                    arima_model = model_dict['trained_model']
                    n_periods = len(df_test_producto)
                    predicciones_arima, conf_int = arima_model.predict(
                        n_periods=n_periods, 
                        return_conf_int=True
                    )
                    
                    resultado_producto = pd.DataFrame({
                        'date': df_test_producto['date'].values,
                        'id_producto': id_producto,
                        'demanda': predicciones_arima
                    })
                    
                    print(f"  ✅ Predicciones ARIMA completadas: {len(predicciones_arima)} registros")
                    
                else:
                    print(f"  ❌ ERROR: Estructura de modelo desconocida para {nombre_modelo}")
                    print(f"      Claves disponibles: {list(model_dict.keys())}")
                    continue
            
            # Agregar a la lista de todas las predicciones
            todas_las_predicciones.append(resultado_producto)
            
        except Exception as e:
            print(f"  ❌ ERROR procesando producto {id_producto}: {str(e)}")
            print(f"      Tipo de error: {type(e).__name__}")
            continue
    
    print("-" * 60)
    
    # Consolidar todas las predicciones en un solo DataFrame
    if todas_las_predicciones:
        # Concatenar todos los DataFrames de predicciones
        df_resultado_final = pd.concat(todas_las_predicciones, ignore_index=True)
        
        # Ordenar por producto y fecha
        df_resultado_final = df_resultado_final.sort_values(['id_producto', 'date']).reset_index(drop=True)
        
        # Crear el directorio output/pred_out_of_sample si no existe
        output_dir = 'output/pred_out_of_sample'
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar las predicciones en el archivo CSV
        output_file = os.path.join(output_dir, 'resultado_prueba.csv')
        df_resultado_final.to_csv(output_file, index=False)
        
        print(f"✅ Predicciones guardadas exitosamente en: {output_file}")
        print(f"📊 Resumen del archivo generado:")
        print(f"   - Total de registros: {len(df_resultado_final):,}")
        print(f"   - Productos incluidos: {df_resultado_final['id_producto'].nunique()}")
        print(f"   - Rango de fechas: {df_resultado_final['date'].min()} a {df_resultado_final['date'].max()}")
        print(f"   - Columnas: {list(df_resultado_final.columns)}")
        
        return df_resultado_final
        
    else:
        print("❌ No se generaron predicciones para ningún producto")
        return None