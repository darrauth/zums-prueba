import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import time
import logging

# Configure logging for Optuna
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

def create_prophet_features(df):
    """
    Crear características para Prophet basándose en las variables exógenas disponibles
    
    Args:
        df: DataFrame con datos temporales
        
    Returns:
        DataFrame con características adicionales para Prophet
    """
    df_features = df.copy()
    
    # Prophet requiere columnas 'ds' (fecha) y 'y' (target)
    # Detectar automáticamente las columnas de fecha y demanda
    date_cols = ['fecha', 'date', 'ds', 'time', 'timestamp']
    demand_cols = ['demanda', 'demand', 'y', 'target', 'value']
    
    # Buscar columna de fecha
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    # Buscar columna de demanda/target
    demand_col = None
    for col in demand_cols:
        if col in df.columns:
            demand_col = col
            break
    
    # Si no encontramos, usar las primeras columnas disponibles
    if date_col is None and len(df.columns) > 0:
        # Buscar columna que contenga fechas
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'fecha' in col.lower() or 'date' in col.lower():
                date_col = col
                break
        
        # Si aún no encontramos, usar la primera columna
        if date_col is None:
            date_col = df.columns[0]
    
    if demand_col is None and len(df.columns) > 1:
        # Buscar columna numérica que no sea fecha
        for col in df.columns:
            if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                demand_col = col
                break
        
        # Si aún no encontramos, usar la segunda columna
        if demand_col is None:
            demand_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    # Crear columnas ds e y para Prophet
    if date_col and demand_col:
        df_features['ds'] = pd.to_datetime(df[date_col])
        df_features['y'] = pd.to_numeric(df[demand_col], errors='coerce')
        
        # Limpiar valores NaN
        df_features = df_features.dropna(subset=['ds', 'y'])
    else:
        raise ValueError("No se pudieron identificar columnas de fecha y demanda en el DataFrame")
    
    # Agregar variables exógenas disponibles
    regressor_cols = []
    potential_regressors = [
        'precio', 'promocion', 'competidor_precio', 'inventario',
        'temperatura', 'estacionalidad', 'tendencia', 'holiday_effect',
        'economic_index'
    ]
    
    for col in potential_regressors:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            regressor_cols.append(col)
    
    return df_features, regressor_cols

def optimize_prophet_model_optuna(train_data, n_trials=10, timeout=60):
    """
    Optimizar hiperparámetros de Prophet usando Optuna
    
    Args:
        train_data: Datos de entrenamiento
        n_trials: Número de pruebas de optimización (reducido para velocidad)
        timeout: Tiempo límite en segundos
        
    Returns:
        Mejores hiperparámetros encontrados
    """
    
    # Preparar datos para Prophet
    train_features, regressor_cols = create_prophet_features(train_data)
    
    def objective(trial):
        try:
            # Hiperparámetros para optimizar (reducidos para velocidad)
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.01, 0.5),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10),
                'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95),
                'n_changepoints': trial.suggest_int('n_changepoints', 15, 35)
            }
            
            # Crear modelo Prophet
            model = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                changepoint_range=params['changepoint_range'],
                n_changepoints=params['n_changepoints'],
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False
            )
            
            # Agregar regressors si están disponibles
            for regressor in regressor_cols[:3]:  # Limitar a 3 regressors para velocidad
                model.add_regressor(regressor)
            
            # Entrenar modelo (con datos mínimos para validación cruzada)
            train_subset = train_features.tail(min(200, len(train_features)))  # Usar subset para velocidad
            model.fit(train_subset)
            
            # Validación simple
            val_size = min(30, len(train_subset) // 4)
            if val_size < 5:
                return float('inf')
                
            train_val = train_subset[:-val_size]
            test_val = train_subset[-val_size:]
            
            # Entrenar en subset de validación
            model_val = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                changepoint_range=params['changepoint_range'],
                n_changepoints=params['n_changepoints'],
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False
            )
            
            for regressor in regressor_cols[:3]:
                model_val.add_regressor(regressor)
                
            model_val.fit(train_val)
            
            # Crear future dataframe para predicción
            future = model_val.make_future_dataframe(periods=val_size, freq='D')
            
            # Agregar regressors al future dataframe
            for regressor in regressor_cols[:3]:
                if regressor in test_val.columns:
                    future[regressor] = list(train_val[regressor]) + list(test_val[regressor])
                else:
                    future[regressor] = future[regressor].fillna(future[regressor].mean())
            
            # Predicción
            forecast = model_val.predict(future)
            pred_val = forecast['yhat'].tail(val_size).values
            true_val = test_val['y'].values
            
            # Calcular MAE
            mae = mean_absolute_error(true_val, pred_val)
            
            if np.isnan(mae) or np.isinf(mae):
                return float('inf')
                
            return mae
            
        except Exception as e:
            return float('inf')
    
    # Optimización con tiempo límite
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params if study.best_params else {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'seasonality_mode': 'additive',
        'changepoint_range': 0.8,
        'n_changepoints': 25
    }

def train_prophet_model(train_data, best_params, regressor_cols):
    """
    Entrenar modelo Prophet con los mejores hiperparámetros
    
    Args:
        train_data: Datos de entrenamiento preparados
        best_params: Mejores hiperparámetros de la optimización
        regressor_cols: Lista de variables exógenas
        
    Returns:
        Modelo Prophet entrenado
    """
    
    # Crear modelo con mejores parámetros
    model = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        holidays_prior_scale=best_params['holidays_prior_scale'],
        seasonality_mode=best_params['seasonality_mode'],
        changepoint_range=best_params['changepoint_range'],
        n_changepoints=best_params['n_changepoints'],
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    
    # Agregar regressors
    for regressor in regressor_cols[:3]:  # Limitar a 3 para velocidad
        model.add_regressor(regressor)
    
    # Entrenar modelo
    model.fit(train_data)
    
    return model

def train_and_evaluate_prophet_model(producto_id, train_data, test_data, output_dir):
    """
    Entrenar y evaluar modelo Prophet para un producto específico
    
    Args:
        producto_id: ID del producto
        train_data: Datos de entrenamiento
        test_data: Datos de prueba
        output_dir: Directorio de salida
        
    Returns:
        Diccionario con métricas del modelo
    """
    try:
        print(f"⏱️  Entrenando modelo Prophet para producto {producto_id} (~1.5 minutos)")
        start_time = time.time()
        
        # Preparar datos
        train_features, regressor_cols = create_prophet_features(train_data)
        test_features, _ = create_prophet_features(test_data)
        
        if len(train_features) < 10 or len(test_features) < 1:
            print(f"❌ Datos insuficientes para producto {producto_id}")
            return None
        
        # Optimizar hiperparámetros (tiempo reducido)
        print(f"🔍 Optimizando hiperparámetros...")
        best_params = optimize_prophet_model_optuna(
            train_data, 
            n_trials=10,  # Reducido para velocidad
            timeout=60    # 1 minuto máximo para optimización
        )
        
        # Entrenar modelo final
        print(f"🚀 Entrenando modelo final...")
        model = train_prophet_model(train_features, best_params, regressor_cols)
        
        # Crear future dataframe para predicción
        future = model.make_future_dataframe(periods=len(test_features), freq='D')
        
        # Agregar regressors al future dataframe
        for regressor in regressor_cols[:3]:
            if regressor in test_features.columns:
                # Combinar valores de train y test
                train_values = train_features[regressor].tolist()
                test_values = test_features[regressor].tolist()
                future[regressor] = train_values + test_values
            else:
                # Usar media si no está disponible
                future[regressor] = future[regressor].fillna(train_features[regressor].mean())
        
        # Realizar predicciones
        forecast = model.predict(future)
        
        # Extraer predicciones para período de test
        test_predictions = forecast['yhat'].tail(len(test_features)).values
        test_true = test_features['y'].values
        
        # Calcular métricas
        mae = mean_absolute_error(test_true, test_predictions)
        rmse = np.sqrt(mean_squared_error(test_true, test_predictions))
        mape = np.mean(np.abs((test_true - test_predictions) / test_true)) * 100
        
        # Guardar predicciones
        pred_df = pd.DataFrame({
            'fecha': test_features['ds'].values,
            'demanda_real': test_true,
            'demanda_predicha': test_predictions,
            'error_absoluto': np.abs(test_true - test_predictions)
        })
        
        pred_file = os.path.join(output_dir, f'test_predicciones_producto_{producto_id}_modelo_prophet.csv')
        pred_df.to_csv(pred_file, index=False)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Modelo Prophet completado en {elapsed_time:.1f}s - MAE: {mae:.2f}")
        
        # Guardar modelo (opcional - Prophet permite serialización)
        model_dir = "Modelos registrados"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        model_file = os.path.join(model_dir, f'prophet_model_producto_{producto_id}.json')
        with open(model_file, 'w') as f:
            f.write(model_to_json(model))
        
        return {
            'producto_id': producto_id,
            'modelo': 'Prophet',
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'tiempo_entrenamiento': elapsed_time,
            'parametros': str(best_params),
            'num_regressors': len(regressor_cols[:3])
        }
        
    except Exception as e:
        print(f"❌ Error en modelo Prophet para producto {producto_id}: {str(e)}")
        return {
            'producto_id': producto_id,
            'modelo': 'Prophet',
            'mae': np.inf,
            'rmse': np.inf,
            'mape': np.inf,
            'tiempo_entrenamiento': 0,
            'parametros': 'Error',
            'num_regressors': 0,
            'error': str(e)
        }

def run_complete_prophet_pipeline(productos_dict, output_dir='output', verbose=True):
    """
    Pipeline completo para entrenar modelos Prophet en todos los productos
    
    Args:
        productos_dict: Diccionario con datos de productos {'id': {'train': df, 'test': df}}
        output_dir: Directorio de salida para resultados
        verbose: Si mostrar información detallada del progreso
        
    Returns:
        DataFrame con resultados de todos los productos
    """
    print("🚀 INICIANDO ENTRENAMIENTO MODELO PROPHET - TODOS LOS PRODUCTOS")
    print("=" * 70)
    print(f"📊 Total de productos a procesar: {len(productos_dict)}")
    print(f"⏱️  Tiempo estimado total: ~{len(productos_dict) * 1.5:.0f} minutos ({len(productos_dict) * 1.5/60:.1f} horas)")
    print(f"🎯 Objetivo: <1.5 minutos por producto")
    print("=" * 70)

    start_time_total = time.time()
    resultados_prophet_completo = []
    productos_completos = list(productos_dict.keys())
    errores_productos = []

    for i, producto_id in enumerate(productos_completos, 1):
        if verbose:
            print(f"\n[{i}/{len(productos_completos)}] 🔄 Procesando Producto {producto_id}")
        
        try:
            # Obtener datos del producto
            train_data = productos_dict[producto_id]['train']
            test_data = productos_dict[producto_id]['test']
            
            # Verificar que los datos no estén vacíos
            if train_data.empty or test_data.empty:
                if verbose and i <= 5:
                    print(f"⚠️ Datos vacíos para producto {producto_id}, saltando...")
                errores_productos.append(producto_id)
                continue
            
            # Entrenar y evaluar modelo Prophet
            resultado = train_and_evaluate_prophet_model(
                producto_id=producto_id,
                train_data=train_data,
                test_data=test_data,
                output_dir=output_dir
            )
            
            if resultado and 'error' not in resultado:
                resultados_prophet_completo.append(resultado)
                
                # Mostrar progreso cada 10 productos
                if verbose and (i % 10 == 0 or i <= 5):
                    elapsed = time.time() - start_time_total
                    avg_time = elapsed / i
                    remaining_products = len(productos_completos) - i
                    eta = remaining_products * avg_time
                    
                    print(f"📈 Progreso: {i}/{len(productos_completos)} ({i/len(productos_completos)*100:.1f}%)")
                    print(f"⏱️  Tiempo promedio: {avg_time:.1f}s por producto")
                    print(f"🕐 ETA restante: {eta/60:.1f} minutos")
                    print(f"✅ Exitosos: {len(resultados_prophet_completo)} | ❌ Errores: {len(errores_productos)}")
            else:
                errores_productos.append(producto_id)
                if verbose and i <= 5:
                    error_msg = resultado.get('error', 'Error desconocido') if resultado else 'Error desconocido'
                    print(f"❌ Error en producto {producto_id}: {error_msg}")
        
        except Exception as e:
            errores_productos.append(producto_id)
            if verbose and i <= 5:
                print(f"❌ Excepción en producto {producto_id}: {str(e)}")
            continue

    # Crear DataFrame con resultados
    if resultados_prophet_completo:
        resultados_prophet_df = pd.DataFrame(resultados_prophet_completo)
    else:
        resultados_prophet_df = pd.DataFrame()

    # Calcular estadísticas finales
    tiempo_total = time.time() - start_time_total
    productos_exitosos = len(resultados_prophet_completo)

    # Mostrar resumen final
    print("\n" + "=" * 70)
    print("✅ MODELO PROPHET COMPLETADO - TODOS LOS PRODUCTOS")
    print("=" * 70)
    print(f"📊 Productos procesados exitosamente: {productos_exitosos}/{len(productos_completos)}")
    print(f"❌ Productos con errores: {len(errores_productos)}")
    print(f"⏱️  Tiempo total: {tiempo_total/60:.1f} minutos ({tiempo_total/3600:.1f} horas)")

    if productos_exitosos > 0:
        print(f"📈 Tiempo promedio por producto exitoso: {tiempo_total/productos_exitosos:.1f}s")
        print(f"🎯 Objetivo cumplido: {'✅ SÍ' if tiempo_total/productos_exitosos <= 90 else '❌ NO'}")
        
        # Guardar resultados
        os.makedirs(output_dir, exist_ok=True)
        resultados_prophet_df.to_csv(f'{output_dir}/resumen_modelo_prophet.csv', index=False)
        print(f"💾 Resultados guardados: {output_dir}/resumen_modelo_prophet.csv")
        
        # Mostrar estadísticas
        print(f"\n📊 ESTADÍSTICAS MODELO PROPHET:")
        print(f"MAE promedio: {resultados_prophet_df['mae'].mean():.2f} ± {resultados_prophet_df['mae'].std():.2f}")
        print(f"RMSE promedio: {resultados_prophet_df['rmse'].mean():.2f} ± {resultados_prophet_df['rmse'].std():.2f}")
        print(f"MAPE promedio: {resultados_prophet_df['mape'].mean():.2f}% ± {resultados_prophet_df['mape'].std():.2f}%")
        
        # Top 5 mejores productos
        if len(resultados_prophet_df) >= 5:
            print(f"\n🏆 TOP 5 MEJORES PRODUCTOS (menor MAE):")
            mejores = resultados_prophet_df.nsmallest(5, 'mae')
            for idx, (_, row) in enumerate(mejores.iterrows(), 1):
                print(f"   {idx}. Producto {row['producto_id']}: MAE={row['mae']:.2f}, MAPE={row['mape']:.2f}%")
    else:
        print("❌ No se pudieron entrenar modelos Prophet exitosamente")

    print("=" * 70)
    
    return resultados_prophet_df

def main():
    """
    Función principal para pruebas del modelo Prophet
    """
    # Configuración de prueba
    print("🔬 Iniciando prueba del modelo Prophet...")
    
    # Simular datos de ejemplo
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    # Crear datos sintéticos con tendencia y estacionalidad
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    demand = trend + seasonal + noise
    
    # Crear DataFrame
    df = pd.DataFrame({
        'fecha': dates,
        'demanda': demand,
        'precio': np.random.uniform(10, 20, len(dates)),
        'promocion': np.random.choice([0, 1], len(dates), p=[0.8, 0.2]),
        'temperatura': 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    })
    
    # Dividir en train/test
    train_size = int(0.8 * len(df))
    train_data = df[:train_size]
    test_data = df[train_size:]
    
    # Probar modelo
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result = train_and_evaluate_prophet_model(
        producto_id=999,
        train_data=train_data,
        test_data=test_data,
        output_dir=output_dir
    )
    
    if result:
        print(f"\n📊 Resultados del modelo Prophet:")
        print(f"MAE: {result['mae']:.2f}")
        print(f"RMSE: {result['rmse']:.2f}")
        print(f"MAPE: {result['mape']:.2f}%")
        print(f"Tiempo: {result['tiempo_entrenamiento']:.1f}s")
        print(f"Regressors utilizados: {result['num_regressors']}")

if __name__ == "__main__":
    main()
