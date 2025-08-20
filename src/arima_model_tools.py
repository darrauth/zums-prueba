"""
Herramientas para entrenamiento y evaluación de modelos ARIMA usando auto_arima.
Simplificado para usar auto_arima que ya optimiza automáticamente los parámetros.

Autor: AI Assistant
Fecha: Agosto 2025
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Machine Learning y estadísticas
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Modelos ARIMA
from pmdarima import auto_arima

# Suprimir warnings
warnings.filterwarnings('ignore')

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
    ss_res = np.sum((y_true - y_pred_safe) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def train_and_evaluate_arima_model(
    producto_data: Dict[str, pd.DataFrame],
    producto_id: int,
    target_col: str = 'demanda',
    seasonal: bool = True,
    m: int = 7,
    output_dir: str = 'output'
) -> Dict[str, Any]:
    """
    Entrena y evalúa un modelo ARIMA para un producto específico usando auto_arima.
    
    Args:
        producto_data: Diccionario con datos de train y test
        producto_id: ID del producto
        target_col: Nombre de la columna objetivo
        seasonal: Si usar componentes estacionales
        m: Frecuencia estacional (7 para datos diarios con patrón semanal)
        output_dir: Directorio de salida
        
    Returns:
        Diccionario con resultados del modelo
    """
    try:
        # Extraer datos
        train_data = producto_data['train'].copy()
        test_data = producto_data['test'].copy()
        
        # Preparar series temporales
        if 'date' in train_data.columns:
            ts_train = train_data.set_index('date')[target_col].sort_index()
            ts_test = test_data.set_index('date')[target_col].sort_index()
        else:
            ts_train = train_data[target_col]
            ts_test = test_data[target_col]
        
        # Usar auto_arima para encontrar el mejor modelo automáticamente
        print(f"Producto {producto_id}: Optimizando ARIMA con auto_arima...")
        
        model = auto_arima(
            ts_train,
            seasonal=seasonal,
            m=m if seasonal else 1,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=3, max_q=3, max_P=2, max_Q=2,
            max_d=2, max_D=1,
            random_state=42,
            n_jobs=-1  # Usar todos los cores disponibles
        )
        
        # Obtener información del modelo encontrado
        order = model.order
        seasonal_order = model.seasonal_order if seasonal else None
        aic_score = model.aic()
        
        print(f"Producto {producto_id}: Mejor modelo ARIMA{order} - AIC: {aic_score:.2f}")
        
        # Hacer predicciones
        # Predicciones en conjunto de entrenamiento (in-sample)
        train_pred = model.fittedvalues()
        
        # Predicciones en conjunto de test (out-of-sample)
        test_pred, conf_int = model.predict(n_periods=len(ts_test), return_conf_int=True)
        
        # Alinear índices para métricas de entrenamiento
        if len(train_pred) != len(ts_train):
            # Ajustar por diferencias en modelos con diferenciación
            min_len = min(len(train_pred), len(ts_train))
            train_pred = train_pred[-min_len:]
            ts_train_aligned = ts_train[-min_len:]
        else:
            ts_train_aligned = ts_train
        
        # Calcular métricas
        train_metrics = calculate_metrics(ts_train_aligned.values, train_pred)
        test_metrics = calculate_metrics(ts_test.values, test_pred)
        
        # Preparar datos para guardar
        test_results = pd.DataFrame({
            'date': ts_test.index if hasattr(ts_test, 'index') else range(len(ts_test)),
            'demanda_real': ts_test.values,
            'demanda_predicha': test_pred,
            'conf_int_lower': conf_int[:, 0],
            'conf_int_upper': conf_int[:, 1],
            'producto_id': producto_id
        })
        
        # Guardar predicciones
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'test_predicciones_producto_{producto_id}_modelo_arima.csv')
        test_results.to_csv(output_path, index=False)
        
        # Preparar resultado
        result = {
            'producto_id': producto_id,
            'modelo': 'AUTO_ARIMA',
            'order': str(order),
            'seasonal_order': str(seasonal_order) if seasonal_order else 'None',
            'aic_score': aic_score,
            'train_rmse': train_metrics['RMSE'],
            'train_mae': train_metrics['MAE'], 
            'train_mape': train_metrics['MAPE'],
            'train_r2': train_metrics['R2'],
            'test_rmse': test_metrics['RMSE'],
            'test_mae': test_metrics['MAE'],
            'test_mape': test_metrics['MAPE'],
            'test_r2': test_metrics['R2'],
            'n_train_samples': len(ts_train),
            'n_test_samples': len(ts_test),
            'seasonal': seasonal,
            'm': m if seasonal else None
        }
        
        # ✅ AÑADIR: Guardar el modelo entrenado ARIMA
        import pickle
        model_dir = "Modelos registrados"
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar el modelo completo ARIMA
        model_filename = f"best_model_producto_{producto_id}_arima.pkl"
        model_path = os.path.join(model_dir, model_filename)
        
        # Crear diccionario completo con modelo Y metadatos
        model_data = {
            'trained_model': model,
            'order': order,
            'seasonal_order': seasonal_order,
            'aic_score': aic_score,
            'seasonal': seasonal,
            'm': m,
            'producto_id': producto_id,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model_type': 'arima'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Producto {producto_id}: Modelo ARIMA guardado en {model_path}")
        
        return result
        
    except Exception as e:
        print(f"Error procesando producto {producto_id}: {str(e)}")
        # Retornar resultado con error
        return {
            'producto_id': producto_id,
            'modelo': 'AUTO_ARIMA',
            'order': 'ERROR',
            'seasonal_order': 'ERROR',
            'aic_score': float('inf'),
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
            'seasonal': seasonal,
            'm': m if seasonal else None,
            'error': str(e)
        }

def run_arima_models_all_products(
    productos_dict: Dict[int, Dict[str, pd.DataFrame]],
    target_col: str = 'demanda',
    seasonal: bool = True,
    m: int = 7,
    output_dir: str = 'output'
) -> pd.DataFrame:
    """
    Ejecuta modelos AUTO_ARIMA para todos los productos.
    
    Args:
        productos_dict: Diccionario con datos por producto
        target_col: Nombre de la columna objetivo
        seasonal: Si usar componentes estacionales
        m: Frecuencia estacional (7 para datos diarios con patrón semanal)
        output_dir: Directorio de salida
        
    Returns:
        DataFrame con resultados de todos los productos
    """
    print(f"\n🚀 INICIANDO ENTRENAMIENTO DE MODELOS AUTO_ARIMA")
    print(f"{'='*60}")
    print(f"Total productos a procesar: {len(productos_dict)}")
    print(f"Modelo: {'SARIMA (auto)' if seasonal else 'ARIMA (auto)'}")
    print(f"Frecuencia estacional: {m if seasonal else 'N/A'}")
    print(f"Directorio de salida: {output_dir}")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    processed = 0
    errors = 0
    
    for producto_id, producto_data in productos_dict.items():
        try:
            print(f"\n📊 Procesando Producto {producto_id}...")
            
            result = train_and_evaluate_arima_model(
                producto_data=producto_data,
                producto_id=producto_id,
                target_col=target_col,
                seasonal=seasonal,
                m=m,
                output_dir=output_dir
            )
            
            results.append(result)
            processed += 1
            
            # Progreso cada 10 productos
            if processed % 10 == 0:
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
        summary_path = os.path.join(output_dir, 'resumen_modelo_arima.csv')
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
                print(f"AIC promedio: {valid_results['aic_score'].mean():.2f}")
        
        return results_df
    else:
        print("❌ No se procesó ningún producto exitosamente")
        return pd.DataFrame()

def get_best_arima_models_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene un resumen de los mejores modelos ARIMA por producto.
    
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
    
    # Como solo tenemos un modelo por producto (AUTO_ARIMA), simplemente retornamos todos
    best_models = valid_results.copy()
    best_models = best_models.sort_values('test_rmse').reset_index(drop=True)
    
    print(f"\n🏆 RESUMEN DE MEJORES MODELOS AUTO_ARIMA:")
    print(f"{'='*60}")
    print(f"Total productos con modelos válidos: {len(best_models)}")
    
    if len(best_models) > 0:
        print(f"Mejor producto: {best_models.iloc[0]['producto_id']} (RMSE: {best_models.iloc[0]['test_rmse']:.4f})")
        print(f"Peor producto: {best_models.iloc[-1]['producto_id']} (RMSE: {best_models.iloc[-1]['test_rmse']:.4f})")
        print(f"RMSE promedio: {best_models['test_rmse'].mean():.4f}")
        print(f"MAPE promedio: {best_models['test_mape'].mean():.2f}%")
        print(f"AIC promedio: {best_models['aic_score'].mean():.2f}")
    
    return best_models

def quick_arima_analysis(results_df: pd.DataFrame) -> None:
    """
    Análisis rápido de resultados de modelos ARIMA.
    
    Args:
        results_df: DataFrame con resultados
    """
    if results_df.empty:
        print("❌ No hay datos para analizar")
        return
    
    print(f"\n🔍 ANÁLISIS RÁPIDO DE RESULTADOS AUTO_ARIMA:")
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
        print(f"  AIC  - Min: {valid_results['aic_score'].min():.2f}, Max: {valid_results['aic_score'].max():.2f}, Promedio: {valid_results['aic_score'].mean():.2f}")
        
        # Análisis de órdenes más comunes
        if 'order' in valid_results.columns:
            print(f"\nÓrdenes ARIMA más comunes:")
            order_counts = valid_results['order'].value_counts().head(5)
            for order, count in order_counts.items():
                print(f"  {order}: {count} productos ({count/len(valid_results)*100:.1f}%)")

def run_complete_arima_pipeline(
    productos_dict: Dict[int, Dict[str, pd.DataFrame]],
    target_col: str = 'demanda',
    seasonal: bool = True,
    m: int = 7,
    output_dir: str = 'output'
) -> pd.DataFrame:
    """
    Pipeline completo para ejecutar modelos ARIMA incluyendo instalación de dependencias,
    entrenamiento, análisis y reporte de resultados.
    
    Esta función comprende todo el flujo de trabajo de ARIMA desde la verificación
    de dependencias hasta el análisis final de resultados.
    
    Args:
        productos_dict: Diccionario con datos por producto {producto_id: {'train': df, 'test': df}}
        target_col: Nombre de la columna objetivo (default: 'demanda')
        seasonal: Si usar componentes estacionales SARIMA (default: True)
        m: Frecuencia estacional - 7 para datos diarios con patrón semanal (default: 7)
        output_dir: Directorio de salida para resultados (default: 'output')
        
    Returns:
        DataFrame con resultados completos de todos los productos
        
    Ejemplo:
        >>> arima_results = run_complete_arima_pipeline(
        ...     productos_dict=productos_dict,
        ...     seasonal=True,
        ...     m=7,
        ...     output_dir='output'
        ... )
    """
    print("="*80)
    print("IMPLEMENTACIÓN DE MODELOS ARIMA")
    print("="*80)
    
    # ========================================
    # VERIFICACIÓN E INSTALACIÓN DE DEPENDENCIAS
    # ========================================
    print("\n📦 VERIFICANDO DEPENDENCIAS...")
    
    # Verificar e instalar pmdarima si es necesario
    try:
        import pmdarima
        print("✅ pmdarima ya está instalado")
    except ImportError:
        print("📦 Instalando pmdarima...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pmdarima"])
            import pmdarima
            print("✅ pmdarima instalado exitosamente")
        except Exception as e:
            print(f"❌ Error instalando pmdarima: {e}")
            return pd.DataFrame()
    
    # ========================================
    # VALIDACIÓN DE DATOS
    # ========================================
    print(f"\n🔍 VALIDANDO DATOS DE ENTRADA...")
    
    if not productos_dict:
        print("❌ Error: productos_dict está vacío.")
        return pd.DataFrame()
    
    print(f"✅ Datos disponibles: {len(productos_dict)} productos")
    
    # Validar estructura de datos
    sample_product = next(iter(productos_dict.values()))
    if 'train' not in sample_product or 'test' not in sample_product:
        print("❌ Error: estructura de datos incorrecta. Se esperan claves 'train' y 'test'.")
        return pd.DataFrame()
    
    if target_col not in sample_product['train'].columns:
        print(f"❌ Error: columna '{target_col}' no encontrada en los datos.")
        return pd.DataFrame()
    
    # ========================================
    # ENTRENAMIENTO DE MODELOS ARIMA
    # ========================================
    print(f"\n🔮 ENTRENANDO MODELOS ARIMA POR PRODUCTOS")
    print("-" * 60)
    print("Iniciando entrenamiento ARIMA...")
    
    # Configuración del modelo
    model_config = {
        'seasonal': seasonal,
        'm': m if seasonal else None,
        'target_col': target_col,
        'output_dir': output_dir
    }
    
    print(f"Configuración:")
    print(f"  - Modelo: {'SARIMA (auto)' if seasonal else 'ARIMA (auto)'}")
    print(f"  - Frecuencia estacional: {m if seasonal else 'N/A'}")
    print(f"  - Variable objetivo: {target_col}")
    print(f"  - Directorio salida: {output_dir}")
    
    # Ejecutar modelos ARIMA para todos los productos
    arima_results_df = run_arima_models_all_products(
        productos_dict=productos_dict,
        target_col=target_col,
        seasonal=seasonal,
        m=m,
        output_dir=output_dir
    )
    
    # ========================================
    # ANÁLISIS DE RESULTADOS ARIMA
    # ========================================
    print(f"\n\n📊 ANÁLISIS DE RESULTADOS ARIMA")
    print("=" * 80)
    
    if not arima_results_df.empty:
        # Análisis rápido detallado
        quick_arima_analysis(arima_results_df)
        
        # Obtener mejores modelos
        best_arima_models = get_best_arima_models_summary(arima_results_df)
        
        # ========================================
        # REPORTE DETALLADO
        # ========================================
        if not best_arima_models.empty:
            print(f"\n🏆 REPORTE DETALLADO DE RESULTADOS:")
            print("-" * 60)
            
            # Top 5 mejores productos (menor RMSE)
            print(f"\nTOP 5 MEJORES PRODUCTOS ARIMA (menor RMSE):")
            top_5_arima = best_arima_models.head(5)
            for idx, row in top_5_arima.iterrows():
                print(f"  #{idx+1}. Producto {row['producto_id']}: {row['order']} - RMSE: {row['test_rmse']:.4f}, MAPE: {row['test_mape']:.2f}%, AIC: {row['aic_score']:.2f}")
            
            # Top 5 productos más difíciles (mayor RMSE)
            print(f"\nTOP 5 PRODUCTOS MÁS DIFÍCILES ARIMA (mayor RMSE):")
            bottom_5_arima = best_arima_models.tail(5)
            for idx, row in bottom_5_arima.iterrows():
                print(f"  #{len(best_arima_models)-4+idx-(len(best_arima_models)-5)}. Producto {row['producto_id']}: {row['order']} - RMSE: {row['test_rmse']:.4f}, MAPE: {row['test_mape']:.2f}%, AIC: {row['aic_score']:.2f}")
            
            # Estadísticas generales
            print(f"\n📈 ESTADÍSTICAS GENERALES:")
            print(f"  Total productos procesados: {len(arima_results_df)}")
            print(f"  Productos con modelos válidos: {len(best_arima_models)}")
            print(f"  Tasa de éxito: {len(best_arima_models)/len(arima_results_df)*100:.1f}%")
            
            valid_results = best_arima_models
            print(f"  RMSE promedio: {valid_results['test_rmse'].mean():.4f}")
            print(f"  RMSE mediano: {valid_results['test_rmse'].median():.4f}")
            print(f"  MAPE promedio: {valid_results['test_mape'].mean():.2f}%")
            print(f"  R² promedio: {valid_results['test_r2'].mean():.4f}")
            print(f"  AIC promedio: {valid_results['aic_score'].mean():.2f}")
        
        # ========================================
        # ARCHIVOS GENERADOS
        # ========================================
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print("-" * 60)
        print(f"📊 Resumen de resultados:")
        print(f"   - {output_dir}/resumen_modelo_arima.csv")
        
        print(f"🔮 Predicciones por producto:")
        print(f"   - {output_dir}/test_predicciones_producto_*_modelo_arima.csv")
        
        print(f"💾 Modelos entrenados:")
        print(f"   - Modelos registrados/best_model_producto_*_arima.pkl")
        
        # Verificar archivos existentes
        if os.path.exists(output_dir):
            prediction_files = [f for f in os.listdir(output_dir) if f.startswith('test_predicciones_producto_') and f.endswith('_modelo_arima.csv')]
            print(f"   Total archivos de predicciones: {len(prediction_files)}")
        
        if os.path.exists("Modelos registrados"):
            model_files = [f for f in os.listdir("Modelos registrados") if f.startswith('best_model_producto_') and f.endswith('_arima.pkl')]
            print(f"   Total modelos guardados: {len(model_files)}")
        
    else:
        print("❌ No se generaron resultados ARIMA válidos")
        print("\nPosibles causas:")
        print("  - Datos insuficientes en algunos productos")
        print("  - Problemas de convergencia en auto_arima")
        print("  - Errores en la estructura de datos")
    
    # ========================================
    # RESUMEN FINAL
    # ========================================
    print(f"\n🎉 PROCESAMIENTO ARIMA COMPLETADO")
    print("=" * 80)
    
    if not arima_results_df.empty:
        valid_count = len(arima_results_df[arima_results_df['test_rmse'] != float('inf')])
        print(f"✅ Pipeline ejecutado exitosamente")
        print(f"📊 {valid_count}/{len(arima_results_df)} productos procesados correctamente")
        
        if valid_count > 0:
            best_rmse = arima_results_df[arima_results_df['test_rmse'] != float('inf')]['test_rmse'].min()
            best_product = arima_results_df.loc[arima_results_df['test_rmse'].idxmin(), 'producto_id']
            print(f"🏆 Mejor resultado: Producto {best_product} (RMSE: {best_rmse:.4f})")
    else:
        print(f"❌ Pipeline falló - revisar logs anteriores")
    
    print("=" * 80)
    
    return arima_results_df
