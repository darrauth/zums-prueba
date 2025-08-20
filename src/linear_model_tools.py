"""
Herramientas para modelos lineales con optimización bayesiana usando Optuna y validación temporal.
Este módulo contiene funciones para entrenar modelos lineales (Ridge, Lasso)
sobre datos de series temporales, con búsqueda bayesiana de hiperparámetros y evaluación completa.
"""

import pandas as pd
import numpy as np
import os
import warnings
from typing import Dict, Tuple, List, Any
from datetime import datetime

# Machine Learning
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optimización Bayesiana
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de evaluación para regresión.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Dict con métricas calculadas
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf
    
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def create_linear_pipeline(model_type: str = 'ridge', **kwargs) -> Pipeline:
    """
    Crea un pipeline con StandardScaler y el modelo lineal especificado.
    
    Args:
        model_type: Tipo de modelo ('ridge', 'lasso')
        **kwargs: Parámetros para el modelo
        
    Returns:
        Pipeline configurado
    """
    # Filtrar parámetros según el tipo de modelo
    if model_type == 'ridge':
        # Ridge solo acepta alpha
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ['alpha']}
        model = Ridge(random_state=42, **filtered_kwargs)
    elif model_type == 'lasso':
        # Lasso acepta alpha
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ['alpha']}
        model = Lasso(random_state=42, **filtered_kwargs)
    else:
        raise ValueError(f"model_type debe ser uno de: ['ridge', 'lasso']")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    return pipeline

def optimize_linear_model_optuna(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    model_type: str = 'ridge',
    n_trials: int = 50,
    cv_folds: int = 5,
    timeout: float = 90.0,
    random_state: int = 42
) -> Tuple[Dict[str, Any], float]:
    """
    Optimiza hiperparámetros de modelos lineales usando optimización bayesiana con Optuna.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Variable objetivo de entrenamiento
        model_type: Tipo de modelo ('ridge', 'lasso')
        n_trials: Número de evaluaciones para la optimización
        cv_folds: Número de folds para TimeSeriesSplit
        timeout: Tiempo máximo en segundos (90 para 1.5 minutos)
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
            # Definir hiperparámetros según el tipo de modelo
            if model_type == 'ridge':
                params = {
                    'alpha': trial.suggest_float('alpha', 1e-6, 1e3, log=True)
                }
            elif model_type == 'lasso':
                params = {
                    'alpha': trial.suggest_float('alpha', 1e-6, 1e1, log=True)
                }
            else:
                raise ValueError(f"model_type no soportado: {model_type}")
            
            # Crear pipeline con parámetros actuales
            pipeline = create_linear_pipeline(model_type, **params)
            
            # Validación cruzada temporal
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Entrenar y predecir
                pipeline.fit(X_tr, y_tr)
                y_pred = pipeline.predict(X_val)
                
                # Calcular RMSE
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
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
        if model_type == 'ridge':
            best_params = {'alpha': 1.0}
        elif model_type == 'lasso':
            best_params = {'alpha': 1.0}
        else:
            best_params = {'alpha': 1.0}
        
        return best_params, np.inf

def train_and_evaluate_linear_model(
    producto_data: Dict[str, pd.DataFrame],
    producto_id: int,
    model_type: str = 'ridge',
    output_dir: str = 'output',
    target_col: str = 'demanda'
) -> Dict[str, Any]:
    """
    Entrena y evalúa un modelo lineal para un producto específico.
    
    Args:
        producto_data: Diccionario con datos 'train' y 'test' del producto
        producto_id: ID del producto
        model_type: Tipo de modelo ('ridge', 'lasso')
        output_dir: Directorio para guardar resultados
        target_col: Nombre de la columna objetivo
        
    Returns:
        Dict con métricas y información del modelo
    """
    
    # Verificar que existen los datos necesarios
    if 'train' not in producto_data or 'test' not in producto_data:
        raise ValueError("producto_data debe contener claves 'train' y 'test'")
    
    train_data = producto_data['train'].copy()
    test_data = producto_data['test'].copy()
    
    # Verificar que hay suficientes datos
    if len(train_data) < 10:
        print(f"Producto {producto_id}: Datos insuficientes para entrenamiento ({len(train_data)} muestras)")
        return None
    
    # Separar features y target
    feature_cols = [col for col in train_data.columns if col not in [target_col, 'date', 'id_producto']]
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    print(f"\nProcesando Producto {producto_id} con {model_type.upper()}")
    print(f"Datos de entrenamiento: {len(X_train)} muestras")
    print(f"Datos de test: {len(X_test)} muestras")
    print(f"Features: {len(feature_cols)}")
    
    # Optimización bayesiana de hiperparámetros con Optuna
    print("Optimizando hiperparámetros con Optuna...")
    best_params, best_cv_score = optimize_linear_model_optuna(
        X_train, y_train, 
        model_type=model_type,
        timeout=90.0
    )
    
    print(f"Mejores parámetros: {best_params}")
    print(f"Mejor CV Score (RMSE): {best_cv_score:.4f}")
    
    # Entrenar modelo final con mejores parámetros
    final_pipeline = create_linear_pipeline(model_type, **best_params)
    final_pipeline.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = final_pipeline.predict(X_train)
    y_test_pred = final_pipeline.predict(X_test)
    
    # Calcular métricas
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print(f"Métricas de Test - RMSE: {test_metrics['RMSE']:.4f}, MAE: {test_metrics['MAE']:.4f}, MAPE: {test_metrics['MAPE']:.2f}%")
    
    # Crear DataFrame con predicciones
    predictions_df = test_data[['date']].copy()
    predictions_df['y_real'] = y_test.values
    predictions_df['y_pred'] = y_test_pred
    predictions_df['producto_id'] = producto_id
    predictions_df['modelo'] = f'{model_type}_lineal'
    
    # Guardar predicciones
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, f'test_predicciones_producto_{producto_id}_modelo_{model_type}.csv')
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predicciones guardadas en: {predictions_file}")
    
    # GUARDAR EL MODELO ENTRENADO CON METADATOS
    import pickle
    model_dir = "Modelos registrados"
    os.makedirs(model_dir, exist_ok=True)
    
    # Preparar metadatos del modelo
    model_metadata = {
        'pipeline': final_pipeline,
        'feature_columns': feature_cols,
        'scaler_mean_': final_pipeline.named_steps['scaler'].mean_,
        'scaler_scale_': final_pipeline.named_steps['scaler'].scale_,
        'model_type': model_type,
        'best_params': best_params,
        'cv_score': best_cv_score,
        'producto_id': producto_id,
        'target_col': target_col,
        'n_features': len(feature_cols),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_date_range': {
            'start': str(train_data['date'].min()) if 'date' in train_data.columns else None,
            'end': str(train_data['date'].max()) if 'date' in train_data.columns else None
        },
        'test_metrics': test_metrics,
        'train_metrics': train_metrics
    }
    
    # Guardar el modelo completo con metadatos
    model_filename = f"best_model_producto_{producto_id}_{model_type}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_metadata, f)
    
    print(f"Modelo con metadatos guardado en: {model_path}")
    
    # Preparar información del modelo
    model_info = {
        'producto_id': producto_id,
        'modelo': f'{model_type}_lineal',
        'mejores_parametros': best_params,
        'cv_score': best_cv_score,
        'train_rmse': train_metrics['RMSE'],
        'train_mae': train_metrics['MAE'],
        'train_mape': train_metrics['MAPE'],
        'train_r2': train_metrics['R2'],
        'test_rmse': test_metrics['RMSE'],
        'test_mae': test_metrics['MAE'],
        'test_mape': test_metrics['MAPE'],
        'test_r2': test_metrics['R2'],
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_file': model_path
    }
    
    return model_info

def run_linear_models_all_products(
    productos_dict: Dict[int, Dict[str, pd.DataFrame]],
    model_types: List[str] = ['ridge', 'lasso'],
    output_dir: str = 'output',
    target_col: str = 'demanda'
) -> pd.DataFrame:
    """
    Ejecuta modelos lineales para todos los productos y guarda resumen.
    
    Args:
        productos_dict: Diccionario con datos de todos los productos
        model_types: Lista de tipos de modelos a entrenar
        output_dir: Directorio para guardar resultados
        target_col: Nombre de la columna objetivo
        
    Returns:
        DataFrame con resumen de todos los modelos
    """
    
    all_results = []
    total_products = len(productos_dict)
    
    print(f"\n{'='*80}")
    print(f"INICIANDO ENTRENAMIENTO DE MODELOS LINEALES")
    print(f"Productos a procesar: {total_products}")
    print(f"Tipos de modelo: {model_types}")
    print(f"{'='*80}")
    
    for i, (producto_id, producto_data) in enumerate(productos_dict.items()):
        print(f"\n[{i+1}/{total_products}] Producto {producto_id}")
        print("-" * 50)
        
        for model_type in model_types:
            try:
                result = train_and_evaluate_linear_model(
                    producto_data=producto_data,
                    producto_id=producto_id,
                    model_type=model_type,
                    output_dir=output_dir,
                    target_col=target_col
                )
                
                if result is not None:
                    all_results.append(result)
                
            except Exception as e:
                print(f"Error procesando producto {producto_id} con {model_type}: {e}")
                continue
    
    # Crear DataFrame resumen
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Guardar resumen
        summary_file = os.path.join(output_dir, 'resumen_modelo_lineal.csv')
        results_df.to_csv(summary_file, index=False)
        
        print(f"\n{'='*80}")
        print(f"PROCESAMIENTO COMPLETADO")
        print(f"Total modelos entrenados: {len(all_results)}")
        print(f"Resumen guardado en: {summary_file}")
        print(f"{'='*80}")
        
        # Mostrar estadísticas generales
        print(f"\nEstadísticas generales (Test RMSE):")
        print(f"Promedio: {results_df['test_rmse'].mean():.4f}")
        print(f"Mediana: {results_df['test_rmse'].median():.4f}")
        print(f"Mínimo: {results_df['test_rmse'].min():.4f}")
        print(f"Máximo: {results_df['test_rmse'].max():.4f}")
        
        return results_df
    else:
        print("No se pudieron entrenar modelos para ningún producto.")
        return pd.DataFrame()

def get_best_models_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene el mejor modelo por producto basado en RMSE de test.
    
    Args:
        results_df: DataFrame con resultados de todos los modelos
        
    Returns:
        DataFrame con el mejor modelo por producto
    """
    if results_df.empty:
        return pd.DataFrame()
    
    # Encontrar el mejor modelo por producto (menor RMSE de test)
    best_models = results_df.loc[results_df.groupby('producto_id')['test_rmse'].idxmin()]
    
    print(f"\nRESUMEN DE MEJORES MODELOS:")
    print(f"Total productos: {len(best_models)}")
    
    # Contar por tipo de modelo
    model_counts = best_models['modelo'].value_counts()
    print(f"\nDistribución de mejores modelos:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} productos ({count/len(best_models)*100:.1f}%)")
    
    return best_models.reset_index(drop=True)

# Función auxiliar para análisis rápido
def quick_analysis(output_dir: str = 'output') -> None:
    """
    Realiza un análisis rápido de los resultados guardados.
    
    Args:
        output_dir: Directorio donde están los resultados
    """
    
    summary_file = os.path.join(output_dir, 'resumen_modelo_lineal.csv')
    
    if not os.path.exists(summary_file):
        print(f"No se encuentra el archivo: {summary_file}")
        return
    
    results_df = pd.read_csv(summary_file)
    best_models = get_best_models_summary(results_df)
    
    print(f"\nTOP 10 MEJORES MODELOS (por RMSE de test):")
    top_10 = results_df.nsmallest(10, 'test_rmse')[['producto_id', 'modelo', 'test_rmse', 'test_mape', 'test_r2']]
    print(top_10.to_string(index=False))
    
    print(f"\nTOP 10 PEORES MODELOS (por RMSE de test):")
    worst_10 = results_df.nlargest(10, 'test_rmse')[['producto_id', 'modelo', 'test_rmse', 'test_mape', 'test_r2']]
    print(worst_10.to_string(index=False))

def load_model_with_metadata(producto_id: int, model_type: str = 'ridge', model_dir: str = 'Modelos registrados') -> Dict[str, Any]:
    """
    Carga un modelo guardado con todos sus metadatos.
    
    Args:
        producto_id: ID del producto
        model_type: Tipo de modelo ('ridge', 'lasso', etc.)
        model_dir: Directorio donde están los modelos
        
    Returns:
        Dict con el modelo y todos sus metadatos
    """
    import pickle
    
    model_filename = f"best_model_producto_{producto_id}_{model_type}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Verificar si es el nuevo formato con metadatos
    if isinstance(model_data, dict) and 'pipeline' in model_data:
        return model_data
    else:
        # Formato antiguo (solo pipeline)
        print(f"Advertencia: Modelo {model_filename} en formato antiguo (sin metadatos)")
        return {
            'pipeline': model_data,
            'feature_columns': None,
            'model_type': model_type,
            'producto_id': producto_id,
            'timestamp': 'Unknown (formato antiguo)'
        }

def create_date_features_for_prediction(dates: List[str], exog_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convierte fechas dadas en features adecuados para hacer predicciones con los modelos entrenados.
    
    Args:
        dates: Lista de fechas en formato string ('YYYY-MM-DD') o datetime
        exog_data: DataFrame con variables exógenas (opcional)
        
    Returns:
        DataFrame con features preparados para predicción
    """
    # Convertir fechas a datetime
    if isinstance(dates[0], str):
        dates_dt = pd.to_datetime(dates)
    else:
        dates_dt = pd.DatetimeIndex(dates)
    
    # Crear DataFrame base
    df_pred = pd.DataFrame({'date': dates_dt})
    
    # Crear features temporales básicos
    df_pred['year'] = df_pred['date'].dt.year
    df_pred['month'] = df_pred['date'].dt.month
    df_pred['quarter'] = df_pred['date'].dt.quarter
    df_pred['day_of_year'] = df_pred['date'].dt.dayofyear
    df_pred['week_of_year'] = df_pred['date'].dt.isocalendar().week
    df_pred['day_of_week'] = df_pred['date'].dt.dayofweek
    df_pred['is_weekend'] = (df_pred['date'].dt.dayofweek >= 5).astype(int)
    
    # Features cíclicos
    df_pred['month_sin'] = np.sin(2 * np.pi * df_pred['month'] / 12)
    df_pred['month_cos'] = np.cos(2 * np.pi * df_pred['month'] / 12)
    df_pred['day_of_week_sin'] = np.sin(2 * np.pi * df_pred['day_of_week'] / 7)
    df_pred['day_of_week_cos'] = np.cos(2 * np.pi * df_pred['day_of_week'] / 7)
    
    # Si hay variables exógenas, hacer merge
    if exog_data is not None:
        # Asegurar que exog_data tiene columna 'date'
        if 'date' in exog_data.columns:
            exog_data_copy = exog_data.copy()
            exog_data_copy['date'] = pd.to_datetime(exog_data_copy['date'])
            df_pred = df_pred.merge(exog_data_copy, on='date', how='left')
        else:
            print("Advertencia: exog_data no tiene columna 'date', se ignorará")
    
    return df_pred

def make_predictions_for_dates(
    producto_id: int, 
    dates: List[str], 
    model_type: str = 'ridge',
    exog_data: pd.DataFrame = None,
    model_dir: str = 'Modelos registrados'
) -> pd.DataFrame:
    """
    Hace predicciones para fechas específicas usando un modelo guardado.
    
    Args:
        producto_id: ID del producto
        dates: Lista de fechas para predecir
        model_type: Tipo de modelo a usar
        exog_data: Variables exógenas (opcional)
        model_dir: Directorio donde están los modelos
        
    Returns:
        DataFrame con fechas y predicciones
    """
    # Cargar modelo con metadatos
    model_data = load_model_with_metadata(producto_id, model_type, model_dir)
    pipeline = model_data['pipeline']
    feature_columns = model_data.get('feature_columns')
    
    # Crear features para las fechas
    df_features = create_date_features_for_prediction(dates, exog_data)
    
    # Si conocemos las columnas exactas del entrenamiento, usarlas
    if feature_columns:
        # Verificar que todas las columnas necesarias estén presentes
        missing_cols = set(feature_columns) - set(df_features.columns)
        if missing_cols:
            print(f"Advertencia: Faltan columnas {missing_cols}")
            # Rellenar con ceros las columnas faltantes
            for col in missing_cols:
                df_features[col] = 0
        
        # Usar solo las columnas del entrenamiento en el mismo orden
        X_pred = df_features[feature_columns]
    else:
        # Formato antiguo, usar todas las columnas excepto 'date'
        feature_cols = [col for col in df_features.columns if col != 'date']
        X_pred = df_features[feature_cols]
    
    # Hacer predicciones
    predictions = pipeline.predict(X_pred)
    
    # Crear DataFrame resultado
    result_df = pd.DataFrame({
        'date': df_features['date'],
        'prediction': predictions,
        'producto_id': producto_id,
        'model_type': model_type
    })
    
    return result_df

def run_complete_linear_model_analysis(
    df_all: pd.DataFrame,
    productos_dict: Dict[int, Dict[str, pd.DataFrame]],
    model_types: List[str] = ['ridge', 'lasso'],
    output_dir: str = 'output',
    target_col: str = 'demanda',
    test_size: float = 0.2,
    timeout_global: float = 120.0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Función integral que ejecuta análisis completo de modelos lineales:
    - Modelo global (todos los productos juntos)
    - Modelos individuales por producto
    - Comparación completa de resultados
    
    Args:
        df_all: DataFrame con datos de todos los productos
        productos_dict: Diccionario con datos separados por producto
        model_types: Lista de tipos de modelos a entrenar
        output_dir: Directorio para guardar resultados
        target_col: Nombre de la columna objetivo
        test_size: Proporción para conjunto de prueba
        timeout_global: Tiempo máximo para optimización del modelo global
        verbose: Si mostrar información detallada
        
    Returns:
        Tupla (results_df_productos, global_df) con resultados de modelos
    """
    
    if verbose:
        print("="*80)
        print("IMPLEMENTACIÓN DE MODELOS LINEALES")
        print("="*80)
    
    # ========================================
    # 1. MODELO GLOBAL
    # ========================================
    if verbose:
        print("\n1. ENTRENANDO MODELO GLOBAL")
        print("-" * 50)
        print("Preparando datos globales...")
    
    # Preparar datos globales
    global_feature_cols = [col for col in df_all.columns if col not in [target_col, 'date', 'id_producto']]
    X_global = df_all[global_feature_cols]
    y_global = df_all[target_col]
    
    if verbose:
        print(f"Datos globales:")
        print(f"  - Total muestras: {len(X_global)}")
        print(f"  - Features: {len(global_feature_cols)}")
        print(f"  - Productos únicos: {df_all['id_producto'].nunique()}")
    
    # División temporal para modelo global
    from .models_data_preprocessing import time_series_split
    
    X_train_global, X_test_global, y_train_global, y_test_global = time_series_split(
        df_all.drop(columns=['id_producto']), 
        target_col=target_col, 
        test_size=test_size
    )
    
    if verbose:
        print(f"División temporal global:")
        print(f"  - Train: {len(X_train_global)} muestras")
        print(f"  - Test: {len(X_test_global)} muestras")
        print("\nOptimizando modelo global...")
    
    # Entrenar modelos globales
    global_results = []
    
    for model_type in model_types:
        if verbose:
            print(f"\nEntrenando modelo GLOBAL {model_type.upper()}")
        
        # Optimizar hiperparámetros
        best_params, best_cv_score = optimize_linear_model_optuna(
            X_train_global.drop(columns=['date']), 
            y_train_global, 
            model_type=model_type,
            timeout=timeout_global
        )
        
        if verbose:
            print(f"Mejores parámetros: {best_params}")
            print(f"Mejor CV Score (RMSE): {best_cv_score:.4f}")
        
        # Entrenar modelo final
        final_pipeline = create_linear_pipeline(model_type, **best_params)
        final_pipeline.fit(X_train_global.drop(columns=['date']), y_train_global)
        
        # Predicciones
        y_train_pred = final_pipeline.predict(X_train_global.drop(columns=['date']))
        y_test_pred = final_pipeline.predict(X_test_global.drop(columns=['date']))
        
        # Métricas
        train_metrics = calculate_metrics(y_train_global, y_train_pred)
        test_metrics = calculate_metrics(y_test_global, y_test_pred)
        
        if verbose:
            print(f"Métricas - Train RMSE: {train_metrics['RMSE']:.4f}, Test RMSE: {test_metrics['RMSE']:.4f}")
        
        # Guardar resultados
        global_results.append({
            'modelo': f'global_{model_type}',
            'tipo': 'global',
            'best_params': best_params,
            'cv_score': best_cv_score,
            'train_rmse': train_metrics['RMSE'],
            'train_mae': train_metrics['MAE'],
            'train_mape': train_metrics['MAPE'],
            'train_r2': train_metrics['R2'],
            'test_rmse': test_metrics['RMSE'],
            'test_mae': test_metrics['MAE'],
            'test_mape': test_metrics['MAPE'],
            'test_r2': test_metrics['R2'],
            'n_features': len(global_feature_cols) - 1,  # -1 por 'date'
            'n_train_samples': len(X_train_global),
            'n_test_samples': len(X_test_global)
        })
    
    # ========================================
    # 2. MODELOS POR PRODUCTOS
    # ========================================
    if verbose:
        print(f"\n\n2. ENTRENANDO MODELOS POR PRODUCTOS")
        print("-" * 50)
        print("Iniciando entrenamiento por productos...")
    
    # Ejecutar modelos lineales para todos los productos
    results_df = run_linear_models_all_products(
        productos_dict=productos_dict,
        model_types=model_types,
        output_dir=output_dir,
        target_col=target_col
    )
    
    # ========================================
    # 3. COMPARACIÓN DE RESULTADOS
    # ========================================
    if verbose:
        print(f"\n\n3. COMPARACIÓN DE RESULTADOS")
        print("=" * 80)
    
    global_df = pd.DataFrame(global_results)
    
    if not results_df.empty and verbose:
        # Estadísticas de modelos por productos
        print(f"\nMODELOS POR PRODUCTOS:")
        print(f"Total modelos entrenados: {len(results_df)}")
        print(f"Productos procesados: {results_df['producto_id'].nunique()}")
        
        print(f"\nEstadísticas por productos (Test RMSE):")
        print(f"  Promedio: {results_df['test_rmse'].mean():.4f}")
        print(f"  Mediana: {results_df['test_rmse'].median():.4f}")
        print(f"  Mínimo: {results_df['test_rmse'].min():.4f}")
        print(f"  Máximo: {results_df['test_rmse'].max():.4f}")
        print(f"  Desv. Estándar: {results_df['test_rmse'].std():.4f}")
        
        # Mejores modelos por producto
        best_models_by_product = get_best_models_summary(results_df)
        
        print(f"\nDistribución de mejores modelos por producto:")
        model_dist = best_models_by_product['modelo'].value_counts()
        for model, count in model_dist.items():
            print(f"  {model}: {count} productos ({count/len(best_models_by_product)*100:.1f}%)")
    
    # Estadísticas de modelos globales
    if verbose:
        print(f"\nMODELOS GLOBALES:")
        for _, model_info in global_df.iterrows():
            print(f"  {model_info['modelo'].upper()}:")
            print(f"    - Test RMSE: {model_info['test_rmse']:.4f}")
            print(f"    - Test MAE: {model_info['test_mae']:.4f}")
            print(f"    - Test MAPE: {model_info['test_mape']:.2f}%")
            print(f"    - Test R²: {model_info['test_r2']:.4f}")
    
    # Comparación global vs por productos
    if not results_df.empty and verbose:
        best_models_by_product = get_best_models_summary(results_df)
        best_global = global_df.loc[global_df['test_rmse'].idxmin()]
        avg_best_products = best_models_by_product['test_rmse'].mean()
        
        print(f"\n" + "=" * 80)
        print(f"COMPARACIÓN FINAL:")
        print(f"=" * 80)
        print(f"Mejor modelo GLOBAL:")
        print(f"  - Modelo: {best_global['modelo']}")
        print(f"  - Test RMSE: {best_global['test_rmse']:.4f}")
        print(f"  - Test MAPE: {best_global['test_mape']:.2f}%")
        
        print(f"\nPromedio de mejores modelos POR PRODUCTOS:")
        print(f"  - Test RMSE promedio: {avg_best_products:.4f}")
        print(f"  - Test MAPE promedio: {best_models_by_product['test_mape'].mean():.2f}%")
        
        improvement = ((best_global['test_rmse'] - avg_best_products) / best_global['test_rmse']) * 100
        
        if improvement > 0:
            print(f"\n✅ Los modelos por productos son MEJORES")
            print(f"   Mejora promedio en RMSE: {improvement:.2f}%")
        else:
            print(f"\n⚠️ El modelo global es MEJOR")
            print(f"   El modelo global supera por: {abs(improvement):.2f}%")
        
        # Top 5 mejores productos
        print(f"\nTOP 5 MEJORES PRODUCTOS (menor RMSE):")
        top_5 = best_models_by_product.nsmallest(5, 'test_rmse')
        for _, row in top_5.iterrows():
            print(f"  Producto {row['producto_id']}: {row['modelo']} - RMSE: {row['test_rmse']:.4f}")
        
        # Top 5 peores productos
        print(f"\nTOP 5 PRODUCTOS MÁS DIFÍCILES (mayor RMSE):")
        bottom_5 = best_models_by_product.nlargest(5, 'test_rmse')
        for _, row in bottom_5.iterrows():
            print(f"  Producto {row['producto_id']}: {row['modelo']} - RMSE: {row['test_rmse']:.4f}")
    
    # Guardar resultados
    os.makedirs(output_dir, exist_ok=True)
    global_df.to_csv(os.path.join(output_dir, 'resumen_modelo_global.csv'), index=False)
    
    if verbose:
        print(f"\n📁 Resultados guardados en:")
        print(f"   - {output_dir}/resumen_modelo_global.csv")
        print(f"   - {output_dir}/resumen_modelo_lineal.csv")
        print(f"\n🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 80)
    
    return results_df, global_df
