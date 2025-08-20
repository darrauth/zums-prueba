"""
Herramientas para modelos LSTM con optimización bayesiana usando Optuna y validación temporal.
Este módulo contiene funciones para entrenar modelos LSTM sobre datos de series temporales,
con búsqueda bayesiana de hiperparámetros y evaluación completa usando PyTorch.
"""

import pandas as pd
import numpy as np
import os
import warnings
from typing import Dict, Tuple, List, Any
from datetime import datetime

# Machine Learning con PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optimización Bayesiana
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

# Configurar PyTorch para CPU
torch.set_default_dtype(torch.float32)
device = torch.device('cpu')

class TimeSeriesDataset(Dataset):
    """Dataset personalizado para series temporales con PyTorch."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de evaluación para series de tiempo.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Diccionario con métricas calculadas
    """
    # Evitar división por cero y valores infinitos
    y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)
    y_pred_safe = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_safe))
    mae = mean_absolute_error(y_true, y_pred_safe)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred_safe) / y_true_safe)) * 100
    
    # R² Score
    r2 = r2_score(y_true, y_pred_safe)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de evaluación para series de tiempo.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Diccionario con métricas calculadas
    """
    # Evitar división por cero y valores infinitos
    y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)
    y_pred_safe = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_safe))
    mae = mean_absolute_error(y_true, y_pred_safe)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred_safe) / y_true_safe)) * 100
    
    # R² Score
    r2 = r2_score(y_true, y_pred_safe)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def create_lstm_sequences(data: np.ndarray, target: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea secuencias temporales para entrenamiento LSTM.
    
    Args:
        data: Datos de entrada (features)
        target: Variable objetivo
        sequence_length: Longitud de la secuencia temporal
        
    Returns:
        Tupla con (X_sequences, y_sequences)
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(target[i])
    
    return np.array(X), np.array(y)

def train_pytorch_lstm_model(model, train_loader, val_loader, num_epochs, learning_rate, patience=5):
    """
    Entrena un modelo LSTM con PyTorch de forma rápida.
    
    Args:
        model: Modelo LSTM de PyTorch
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        num_epochs: Número máximo de epochs
        learning_rate: Tasa de aprendizaje
        patience: Paciencia para early stopping
        
    Returns:
        Modelo entrenado
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    model.train()
    
    for epoch in range(num_epochs):
        # Entrenamiento
        train_loss = 0.0
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validación cada 2 epochs para ahorrar tiempo
        if epoch % 2 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Guardar mejor modelo
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    # Cargar el mejor modelo si existe
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def optimize_lstm_model_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sequence_length: int = 30,
    n_trials: int = 15,  # Reducido para 1.5 min por producto
    cv_folds: int = 2,
    timeout: float = 60.0,  # 1 minuto para optimización
    random_state: int = 42
) -> Tuple[Dict[str, Any], float]:
    """
    Optimiza hiperparámetros de modelos LSTM usando optimización bayesiana con Optuna.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Variable objetivo de entrenamiento
        sequence_length: Longitud de secuencias temporales
        n_trials: Número de evaluaciones para la optimización
        cv_folds: Número de folds para TimeSeriesSplit
        timeout: Tiempo máximo en segundos
        random_state: Semilla aleatoria
        
    Returns:
        Tupla con (mejores_parametros, mejor_score)
    """
    
    # Configurar TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    def objective(trial):
        """
        Función objetivo para Optuna.
        """
        try:
            # Definir hiperparámetros (simplificados para velocidad)
            params = {
                'hidden_size': trial.suggest_int('hidden_size', 32, 64),
                'num_layers': trial.suggest_int('num_layers', 1, 2),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32])
            }
            
            # Validación cruzada temporal
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                # Normalizar datos
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                
                X_tr_scaled = scaler_X.fit_transform(X_tr)
                y_tr_scaled = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
                
                X_val_scaled = scaler_X.transform(X_val)
                y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
                
                # Crear secuencias
                X_tr_seq, y_tr_seq = create_lstm_sequences(X_tr_scaled, y_tr_scaled, sequence_length)
                X_val_seq, y_val_seq = create_lstm_sequences(X_val_scaled, y_val_scaled, sequence_length)
                
                if len(X_tr_seq) == 0 or len(X_val_seq) == 0:
                    cv_scores.append(float('inf'))
                    continue
                
                # Crear datasets y dataloaders
                train_dataset = TimeSeriesDataset(X_tr_seq, y_tr_seq)
                val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
                
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
                val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
                
                # Crear modelo
                input_size = X_tr_scaled.shape[1]
                model = LSTMModel(
                    input_size=input_size,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    dropout_rate=params['dropout_rate']
                ).to(device)
                
                # Entrenar modelo (epochs muy reducidos)
                model = train_pytorch_lstm_model(
                    model, train_loader, val_loader, 
                    num_epochs=10,  # Muy reducido para velocidad
                    learning_rate=params['learning_rate'],
                    patience=3  # Paciencia muy reducida
                )
                
                # Evaluar
                model.eval()
                y_pred_scaled = []
                with torch.no_grad():
                    for batch_X, _ in val_loader:
                        batch_X = batch_X.to(device)
                        outputs = model(batch_X)
                        y_pred_scaled.extend(outputs.squeeze().cpu().numpy())
                
                y_pred_scaled = np.array(y_pred_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                y_val_original = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1)).ravel()
                
                # Calcular RMSE
                rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
                cv_scores.append(rmse)
            
            # Score promedio (Optuna minimiza por defecto)
            mean_score = np.mean(cv_scores)
            return mean_score
            
        except Exception as e:
            print(f"Error en trial {trial.number}: {e}")
            return np.inf
    
    try:
        # Configurar Optuna para suprimir logs verbosos
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Crear estudio de optimización
        sampler = TPESampler(seed=random_state)
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler
        )
        
        # Ejecutar optimización con timeout
        study.optimize(
            objective, 
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False
        )
        
        # Obtener mejores parámetros y score
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"Optuna completó {len(study.trials)} trials")
        
        return best_params, best_score
        
    except Exception as e:
        print(f"Error en optimización con Optuna: {e}")
        # Usar parámetros por defecto como fallback
        best_params = {
            'hidden_size': 50,
            'num_layers': 1,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        return best_params, np.inf

def train_and_evaluate_lstm_model(
    producto_data: Dict[str, pd.DataFrame],
    producto_id: int,
    sequence_length: int = 30,
    output_dir: str = 'output',
    target_col: str = 'demanda'
) -> Dict[str, Any]:
    """
    Entrena y evalúa un modelo LSTM para un producto específico usando PyTorch.
    
    Args:
        producto_data: Diccionario con datos 'train' y 'test' del producto
        producto_id: ID del producto
        sequence_length: Longitud de secuencias temporales
        output_dir: Directorio para guardar resultados
        target_col: Nombre de la columna objetivo
        
    Returns:
        Diccionario con resultados del modelo
    """
    try:
        # Extraer datos
        train_data = producto_data['train'].copy()
        test_data = producto_data['test'].copy()
        
        # Preparar features y target
        feature_cols = [col for col in train_data.columns if col not in [target_col, 'date', 'id_producto']]
        
        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values
        
        print(f"Producto {producto_id}: Datos - Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")
        
        # Verificar que hay suficientes datos para secuencias
        if len(X_train) <= sequence_length or len(X_test) <= sequence_length:
            print(f"Producto {producto_id}: Datos insuficientes para secuencias de longitud {sequence_length}")
            # Usar secuencia más corta
            sequence_length = min(10, len(X_train) // 2, len(X_test) // 2)
            if sequence_length < 5:
                raise ValueError("Datos insuficientes para crear secuencias mínimas")
        
        # Optimizar hiperparámetros (tiempo limitado)
        print(f"Producto {producto_id}: Optimizando hiperparámetros...")
        best_params, best_cv_score = optimize_lstm_model_optuna(
            X_train, y_train, 
            sequence_length=sequence_length,
            n_trials=10,     # Solo 10 trials por producto
            timeout=60.0     # Máximo 1 minuto para optimización
        )
        
        print(f"Producto {producto_id}: Mejores parámetros - {best_params}")
        print(f"Producto {producto_id}: Mejor CV Score (RMSE): {best_cv_score:.4f}")
        
        # Entrenar modelo final con todos los datos de entrenamiento
        # Normalizar datos
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        X_test_scaled = scaler_X.transform(X_test)
        
        # Crear secuencias LSTM
        X_train_seq, y_train_seq = create_lstm_sequences(X_train_scaled, y_train_scaled, sequence_length)
        X_test_seq, _ = create_lstm_sequences(X_test_scaled, y_test, sequence_length)
        
        # Crear datasets y dataloaders
        train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(train_dataset, batch_size=best_params.get('batch_size', 32), shuffle=False)
        
        test_dataset = TimeSeriesDataset(X_test_seq, np.zeros(len(X_test_seq)))  # Dummy targets para test
        test_loader = DataLoader(test_dataset, batch_size=best_params.get('batch_size', 32), shuffle=False)
        
        # Crear modelo final
        input_size = X_train_scaled.shape[1]
        final_model = LSTMModel(
            input_size=input_size,
            hidden_size=best_params.get('hidden_size', 50),
            num_layers=best_params.get('num_layers', 1),
            dropout_rate=best_params.get('dropout_rate', 0.2)
        ).to(device)
        
        # Entrenar modelo final (rápido)
        final_model = train_pytorch_lstm_model(
            final_model, train_loader, train_loader,  # Usar train como val para final
            num_epochs=15,  # Solo 15 epochs para modelo final
            learning_rate=best_params.get('learning_rate', 0.001),
            patience=5      # Paciencia reducida
        )
        
        # Predicciones
        final_model.eval()
        
        # Train predictions
        train_pred_scaled = []
        with torch.no_grad():
            for batch_X, _ in train_loader:
                batch_X = batch_X.to(device)
                outputs = final_model(batch_X)
                train_pred_scaled.extend(outputs.squeeze().cpu().numpy())
        
        train_pred_scaled = np.array(train_pred_scaled)
        train_pred = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
        train_true = y_train[sequence_length:]  # Ajustar por las secuencias perdidas
        
        # Test predictions
        test_pred_scaled = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                outputs = final_model(batch_X)
                test_pred_scaled.extend(outputs.squeeze().cpu().numpy())
        
        test_pred_scaled = np.array(test_pred_scaled)
        test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
        test_true = y_test[sequence_length:]  # Ajustar por las secuencias perdidas
        
        # Calcular métricas
        train_metrics = calculate_metrics(train_true, train_pred)
        test_metrics = calculate_metrics(test_true, test_pred)
        
        print(f"Producto {producto_id}: Train RMSE: {train_metrics['RMSE']:.4f}, Test RMSE: {test_metrics['RMSE']:.4f}")
        
        # Preparar datos para guardar (incluyendo fechas si están disponibles)
        test_results = pd.DataFrame({
            'demanda_real': test_true,
            'demanda_predicha': test_pred,
            'producto_id': producto_id
        })
        
        if 'date' in test_data.columns:
            test_dates = test_data['date'].iloc[sequence_length:].reset_index(drop=True)
            test_results['date'] = test_dates
        
        # Guardar predicciones
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'test_predicciones_producto_{producto_id}_modelo_lstm.csv')
        test_results.to_csv(output_path, index=False)
        
        # Preparar resultado
        result = {
            'producto_id': producto_id,
            'modelo': 'LSTM',
            'sequence_length': sequence_length,
            'best_params': str(best_params),
            'cv_score': best_cv_score,
            'train_rmse': train_metrics['RMSE'],
            'train_mae': train_metrics['MAE'],
            'train_mape': train_metrics['MAPE'],
            'train_r2': train_metrics['R2'],
            'test_rmse': test_metrics['RMSE'],
            'test_mae': test_metrics['MAE'],
            'test_mape': test_metrics['MAPE'],
            'test_r2': test_metrics['R2'],
            'n_train_samples': len(train_true),
            'n_test_samples': len(test_true),
            'n_features': len(feature_cols)
        }
        
        # ✅ AÑADIR: Guardar el modelo entrenado LSTM
        import pickle
        model_dir = "Modelos registrados"
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar el modelo completo LSTM (PyTorch + escaladores)
        model_filename = f"best_model_producto_{producto_id}_lstm.pkl"
        model_path = os.path.join(model_dir, model_filename)
        
        # Crear diccionario completo con modelo Y metadatos
        model_data = {
            'trained_model': final_model.state_dict(),  # Estado del modelo PyTorch
            'model_architecture': {
                'input_size': input_size,
                'hidden_size': best_params.get('hidden_size', 50),
                'num_layers': best_params.get('num_layers', 1),
                'dropout_rate': best_params.get('dropout_rate', 0.2)
            },
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'sequence_length': sequence_length,
            'feature_columns': feature_cols,
            'best_params': best_params,
            'cv_score': best_cv_score,
            'producto_id': producto_id,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model_type': 'lstm'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Producto {producto_id}: Modelo LSTM guardado en {model_path}")
        
        return result
        
    except Exception as e:
        print(f"Error procesando producto {producto_id}: {str(e)}")
        # Retornar resultado con error
        return {
            'producto_id': producto_id,
            'modelo': 'LSTM',
            'sequence_length': sequence_length,
            'best_params': f'ERROR: {str(e)}',
            'cv_score': float('inf'),
            'train_rmse': float('inf'),
            'train_mae': float('inf'),
            'train_mape': float('inf'),
            'train_r2': 0.0,
            'test_rmse': float('inf'),
            'test_mae': float('inf'),
            'test_mape': float('inf'),
            'test_r2': 0.0,
            'n_train_samples': 0,
            'n_test_samples': 0,
            'n_features': 0
        }

def run_lstm_models_all_products(
    productos_dict: Dict[int, Dict[str, pd.DataFrame]],
    sequence_length: int = 30,
    output_dir: str = 'output',
    target_col: str = 'demanda'
) -> pd.DataFrame:
    """
    Ejecuta modelos LSTM para todos los productos.
    
    Args:
        productos_dict: Diccionario con datos por producto
        sequence_length: Longitud de secuencias temporales
        output_dir: Directorio de salida
        target_col: Nombre de la columna objetivo
        
    Returns:
        DataFrame con resultados de todos los productos
    """
    print(f"\n🚀 INICIANDO ENTRENAMIENTO DE MODELOS LSTM")
    print(f"{'='*60}")
    print(f"Total productos a procesar: {len(productos_dict)}")
    print(f"Longitud de secuencias: {sequence_length}")
    print(f"Directorio de salida: {output_dir}")
    print(f"⏱️  Tiempo estimado: ~{len(productos_dict) * 1.5:.1f} minutos (1.5 min por producto)")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    processed = 0
    errors = 0
    
    for producto_id, producto_data in productos_dict.items():
        try:
            print(f"\n📊 Procesando Producto {producto_id}...")
            
            result = train_and_evaluate_lstm_model(
                producto_data=producto_data,
                producto_id=producto_id,
                sequence_length=sequence_length,
                output_dir=output_dir,
                target_col=target_col
            )
            
            results.append(result)
            processed += 1
            
            # Progreso cada 5 productos (LSTM puede ser más lento)
            if processed % 5 == 0:
                print(f"✅ Progreso: {processed}/{len(productos_dict)} productos completados")
                
        except Exception as e:
            print(f"❌ Error procesando producto {producto_id}: {str(e)}")
            errors += 1
            continue
    
    print(f"\n🎯 RESUMEN DE PROCESAMIENTO:")
    print(f"{'='*60}")
    print(f"✅ Productos procesados exitosamente: {processed}")
    print(f"❌ Productos con errores: {errors}")
    print(f"📊 Total productos: {len(productos_dict)}")
    
    # Convertir a DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # Guardar resumen
        summary_path = os.path.join(output_dir, 'resumen_modelo_lstm.csv')
        results_df.to_csv(summary_path, index=False)
        print(f"📁 Resumen guardado en: {summary_path}")
        
        # Estadísticas rápidas
        if processed > 0:
            print(f"\n📈 ESTADÍSTICAS RÁPIDAS:")
            print(f"{'='*60}")
            valid_results = results_df[results_df['test_rmse'] != float('inf')]
            if not valid_results.empty:
                print(f"Test RMSE promedio: {valid_results['test_rmse'].mean():.4f}")
                print(f"Test RMSE mediano: {valid_results['test_rmse'].median():.4f}")
                print(f"Test MAPE promedio: {valid_results['test_mape'].mean():.2f}%")
                print(f"Mejor producto (menor RMSE): {valid_results.loc[valid_results['test_rmse'].idxmin(), 'producto_id']}")
        
        return results_df
    else:
        print("❌ No se procesó ningún producto exitosamente")
        return pd.DataFrame()

def get_best_lstm_models_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene un resumen de los mejores modelos LSTM por producto.
    
    Args:
        results_df: DataFrame con resultados de todos los modelos
        
    Returns:
        DataFrame con el mejor modelo por producto
    """
    if results_df.empty:
        return pd.DataFrame()
    
    # Filtrar resultados válidos
    valid_results = results_df[results_df['test_rmse'] != float('inf')].copy()
    
    if valid_results.empty:
        print("⚠️ No se encontraron resultados válidos")
        return pd.DataFrame()
    
    # Como solo tenemos un modelo por producto (LSTM), simplemente retornamos todos ordenados
    best_models = valid_results.copy()
    best_models = best_models.sort_values('test_rmse').reset_index(drop=True)
    
    print(f"\n🏆 RESUMEN DE MEJORES MODELOS LSTM:")
    print(f"{'='*60}")
    print(f"Total productos con modelos válidos: {len(best_models)}")
    
    if len(best_models) > 0:
        print(f"Mejor producto: {best_models.iloc[0]['producto_id']} (RMSE: {best_models.iloc[0]['test_rmse']:.4f})")
        print(f"Peor producto: {best_models.iloc[-1]['producto_id']} (RMSE: {best_models.iloc[-1]['test_rmse']:.4f})")
        print(f"RMSE promedio: {best_models['test_rmse'].mean():.4f}")
        print(f"MAPE promedio: {best_models['test_mape'].mean():.2f}%")
    
    return best_models

def quick_lstm_analysis(results_df: pd.DataFrame) -> None:
    """
    Análisis rápido de resultados de modelos LSTM.
    
    Args:
        results_df: DataFrame con resultados
    """
    if results_df.empty:
        print("❌ No hay datos para analizar")
        return
    
    print(f"\n🔍 ANÁLISIS RÁPIDO DE RESULTADOS LSTM:")
    print(f"{'='*60}")
    
    # Filtrar resultados válidos
    valid_results = results_df[results_df['test_rmse'] != float('inf')]
    
    print(f"Productos procesados: {len(results_df)}")
    print(f"Productos válidos: {len(valid_results)}")
    print(f"Productos con error: {len(results_df) - len(valid_results)}")
    
    if not valid_results.empty:
        print(f"\nMétricas de Test:")
        print(f"  RMSE - Min: {valid_results['test_rmse'].min():.4f}, Max: {valid_results['test_rmse'].max():.4f}, Promedio: {valid_results['test_rmse'].mean():.4f}")
        print(f"  MAE  - Min: {valid_results['test_mae'].min():.4f}, Max: {valid_results['test_mae'].max():.4f}, Promedio: {valid_results['test_mae'].mean():.4f}")
        print(f"  MAPE - Min: {valid_results['test_mape'].min():.2f}%, Max: {valid_results['test_mape'].max():.2f}%, Promedio: {valid_results['test_mape'].mean():.2f}%")
        print(f"  R²   - Min: {valid_results['test_r2'].min():.4f}, Max: {valid_results['test_r2'].max():.4f}, Promedio: {valid_results['test_r2'].mean():.4f}")
        
        # Análisis de longitudes de secuencia más comunes
        if 'sequence_length' in valid_results.columns:
            print(f"\nLongitudes de secuencia más comunes:")
            seq_counts = valid_results['sequence_length'].value_counts().head(3)
            for seq_len, count in seq_counts.items():
                print(f"  {seq_len} pasos: {count} productos ({count/len(valid_results)*100:.1f}%)")

def run_complete_lstm_pipeline(
    productos_dict: Dict[int, Dict[str, pd.DataFrame]], 
    sequence_length: int = 15, 
    output_dir: str = 'output',
    target_col: str = 'demanda'
) -> pd.DataFrame:
    """
    Pipeline completo de modelos LSTM - Función integral que ejecuta todo el proceso.
    
    Incluye:
    - Entrenamiento de modelos LSTM para todos los productos
    - Análisis completo de resultados
    - Resumen de mejores modelos
    - Guardado de archivos de resultados
    
    Args:
        productos_dict: Diccionario con datos por producto
        sequence_length: Longitud de secuencias temporales
        output_dir: Directorio de salida
        target_col: Nombre de la columna objetivo
        
    Returns:
        DataFrame con resultados completos
    """
    print("="*80)
    print("IMPLEMENTACIÓN DE MODELOS LSTM (PyTorch - RÁPIDO)")
    print("="*80)
    print("📊 Entrenando modelos LSTM para todos los productos...")
    print(f"⏱️  Tiempo estimado: ~{len(productos_dict) * 1.5:.1f} minutos (1.5 min por producto)")
    print("🔧 Características optimizadas para velocidad:")
    print(f"   - Secuencias temporales: {sequence_length} pasos")
    print("   - Optimización rápida: 10 trials por producto")
    print("   - Entrenamiento: máximo 15 epochs")
    print("   - Early stopping agresivo")
    print("   - Implementación PyTorch estable")
    
    # Ejecutar modelos LSTM para todos los productos (RÁPIDO)
    resultados_lstm = run_lstm_models_all_products(
        productos_dict=productos_dict,
        sequence_length=sequence_length,
        output_dir=output_dir,
        target_col=target_col
    )
    
    # Mostrar resumen de resultados
    print("\n" + "="*80)
    print("📈 ANÁLISIS DE RESULTADOS LSTM")
    print("="*80)
    
    if not resultados_lstm.empty:
        # Análisis rápido
        quick_lstm_analysis(resultados_lstm)
        
        # Obtener mejores modelos
        mejores_lstm = get_best_lstm_models_summary(resultados_lstm)
        
        print(f"\n📊 Resultados guardados en:")
        print(f"   - Resumen: {output_dir}/resumen_modelo_lstm.csv")
        print(f"   - Predicciones individuales: {output_dir}/test_predicciones_producto_*_modelo_lstm.csv")
        
        # Mostrar top 5 productos
        if len(mejores_lstm) >= 5:
            print(f"\n🏆 TOP 5 PRODUCTOS CON MEJOR DESEMPEÑO LSTM:")
            for i in range(5):
                producto_id = mejores_lstm.iloc[i]['producto_id']
                rmse = mejores_lstm.iloc[i]['test_rmse']
                mape = mejores_lstm.iloc[i]['test_mape']
                print(f"   {i+1}. Producto {producto_id}: RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        
    else:
        print("❌ No se pudieron entrenar modelos LSTM")
    
    print("\n✅ Proceso de modelos LSTM completado!")
    
    return resultados_lstm
