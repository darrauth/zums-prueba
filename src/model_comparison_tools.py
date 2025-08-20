"""
Herramientas para comparación y selección de mejores modelos por producto.
Este módulo permite consolidar resultados de diferentes tipos de modelos sin re-entrenarlos,
usando los archivos CSV generados por cada tipo de modelo.

Autor: AI Assistant
Fecha: Agosto 2025
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

def load_model_results(output_dir: str = 'output') -> Dict[str, pd.DataFrame]:
    """
    Carga todos los archivos de resultados de modelos disponibles.
    
    Args:
        output_dir: Directorio donde están los archivos CSV
        
    Returns:
        Diccionario con DataFrames de cada tipo de modelo
    """
    results = {}
    
    # Mapeo de archivos a nombres de modelos
    model_files = {
        'linear': 'resumen_modelo_lineal.csv',
        'arima': 'resumen_modelo_arima.csv',
        'lstm': 'resumen_modelo_lstm.csv',
        'prophet': 'resumen_modelo_prophet.csv',
        'global': 'resumen_modelo_global.csv'
    }
    
    print(f"🔍 CARGANDO RESULTADOS DE MODELOS")
    print(f"{'='*50}")
    
    for model_type, filename in model_files.items():
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                results[model_type] = df
                print(f"✅ {model_type.upper()}: {len(df)} registros cargados desde {filename}")
            except Exception as e:
                print(f"❌ Error cargando {filename}: {e}")
        else:
            print(f"⚠️ {model_type.upper()}: Archivo {filename} no encontrado")
    
    return results

def standardize_model_results(model_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Estandariza y consolida resultados de diferentes tipos de modelos.
    
    Args:
        model_results: Diccionario con DataFrames de cada tipo de modelo
        
    Returns:
        DataFrame consolidado con formato estándar
    """
    consolidated_results = []
    
    print(f"\n🔧 ESTANDARIZANDO RESULTADOS")
    print(f"{'='*50}")
    
    for model_type, df in model_results.items():
        if df.empty:
            continue
            
        # Crear copia para no modificar el original
        df_std = df.copy()
        
        # Agregar identificador de tipo de modelo
        df_std['model_type'] = model_type.upper()
        
        # Estandarizar nombres de columnas comunes
        column_mapping = {
            'producto_id': 'producto_id',
            'model': 'modelo_especifico',
            'modelo': 'modelo_especifico', 
            'test_rmse': 'test_rmse',
            'test_mae': 'test_mae',
            'test_mape': 'test_mape',
            'test_r2': 'test_r2',
            'train_rmse': 'train_rmse',
            'train_mae': 'train_mae',
            'train_mape': 'train_mape',
            'train_r2': 'train_r2'
        }
        
        # Renombrar columnas que existan
        existing_columns = {k: v for k, v in column_mapping.items() if k in df_std.columns}
        df_std = df_std.rename(columns=existing_columns)
        
        # Asegurar que existan las columnas mínimas requeridas
        required_cols = ['producto_id', 'test_rmse', 'test_mae', 'test_mape', 'test_r2']
        
        if all(col in df_std.columns for col in required_cols):
            consolidated_results.append(df_std)
            print(f"✅ {model_type.upper()}: Estandarizado - {len(df_std)} registros")
        else:
            missing_cols = [col for col in required_cols if col not in df_std.columns]
            print(f"❌ {model_type.upper()}: Faltan columnas {missing_cols}")
    
    if consolidated_results:
        final_df = pd.concat(consolidated_results, ignore_index=True)
        print(f"\n✅ CONSOLIDACIÓN COMPLETA: {len(final_df)} registros totales")
        return final_df
    else:
        print(f"\n❌ NO SE PUDO CONSOLIDAR NINGÚN RESULTADO")
        return pd.DataFrame()

def calculate_composite_score(df: pd.DataFrame, 
                            weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Calcula un puntaje compuesto basado en múltiples métricas.
    
    Args:
        df: DataFrame con métricas de modelos
        weights: Pesos para cada métrica
        
    Returns:
        DataFrame con puntajes compuestos agregados
    """
    if weights is None:
        weights = {
            'rmse': 0.4,    # RMSE tiene mayor peso
            'mae': 0.3,     # MAE segundo peso
            'mape': 0.2,    # MAPE tercer peso  
            'r2': 0.1       # R2 menor peso
        }
    
    df_score = df.copy()
    
    # Normalizar métricas (convertir a "mayor es mejor")
    # Para RMSE, MAE, MAPE: menor es mejor → usar 1/(1+métrica)
    df_score['rmse_norm'] = 1 / (1 + df_score['test_rmse'])
    df_score['mae_norm'] = 1 / (1 + df_score['test_mae'])  
    df_score['mape_norm'] = 1 / (1 + df_score['test_mape'])
    
    # Para R2: mayor es mejor → usar directamente (pero asegurar que sea positivo)
    df_score['r2_norm'] = np.maximum(df_score['test_r2'], 0)
    
    # Calcular puntaje compuesto
    df_score['composite_score'] = (
        weights['rmse'] * df_score['rmse_norm'] +
        weights['mae'] * df_score['mae_norm'] +
        weights['mape'] * df_score['mape_norm'] +
        weights['r2'] * df_score['r2_norm']
    )
    
    return df_score

def get_best_model_per_product(df: pd.DataFrame, 
                             criterion: str = 'test_rmse') -> pd.DataFrame:
    """
    Selecciona el mejor modelo para cada producto basado en un criterio.
    
    Args:
        df: DataFrame consolidado con todos los modelos
        criterion: Criterio de selección ('test_rmse', 'test_mae', 'test_mape', 'composite_score')
        
    Returns:
        DataFrame con el mejor modelo por producto
    """
    if df.empty:
        return pd.DataFrame()
    
    # Determinar si el criterio es "menor es mejor" o "mayor es mejor"
    ascending = criterion not in ['test_r2', 'composite_score']
    
    print(f"\n🏆 SELECCIONANDO MEJORES MODELOS POR PRODUCTO")
    print(f"{'='*60}")
    print(f"Criterio de selección: {criterion} ({'menor es mejor' if ascending else 'mayor es mejor'})")
    
    # Seleccionar el mejor modelo por producto
    best_models = df.loc[df.groupby('producto_id')[criterion].idxmin() if ascending 
                        else df.groupby('producto_id')[criterion].idxmax()]
    
    best_models = best_models.reset_index(drop=True)
    
    print(f"✅ {len(best_models)} productos con mejor modelo seleccionado")
    
    return best_models

def generate_model_comparison_report(df: pd.DataFrame, 
                                   best_models: pd.DataFrame) -> Dict[str, any]:
    """
    Genera un reporte completo de comparación de modelos.
    
    Args:
        df: DataFrame consolidado con todos los modelos
        best_models: DataFrame con mejores modelos por producto
        
    Returns:
        Diccionario con estadísticas del reporte
    """
    report = {}
    
    print(f"\n📊 GENERANDO REPORTE DE COMPARACIÓN")
    print(f"{'='*60}")
    
    # 1. Distribución de tipos de modelos ganadores
    model_distribution = best_models['model_type'].value_counts()
    report['model_distribution'] = model_distribution
    
    print(f"\n🏅 DISTRIBUCIÓN DE MODELOS GANADORES:")
    print(f"{'-'*50}")
    for model_type, count in model_distribution.items():
        percentage = (count / len(best_models)) * 100
        print(f"  {model_type}: {count} productos ({percentage:.1f}%)")
    
    # 2. Estadísticas por tipo de modelo
    model_stats = df.groupby('model_type').agg({
        'test_rmse': ['mean', 'median', 'std', 'min', 'max'],
        'test_mae': ['mean', 'median'],
        'test_mape': ['mean', 'median'],
        'test_r2': ['mean', 'median'],
        'producto_id': 'count'
    }).round(4)
    
    report['model_statistics'] = model_stats
    
    print(f"\n📈 ESTADÍSTICAS POR TIPO DE MODELO:")
    print(f"{'-'*50}")
    print(model_stats)
    
    # 3. Top mejores y peores productos
    best_products = best_models.nsmallest(5, 'test_rmse')
    worst_products = best_models.nlargest(5, 'test_rmse')
    
    report['best_products'] = best_products
    report['worst_products'] = worst_products
    
    print(f"\n🥇 TOP 5 MEJORES PRODUCTOS (menor RMSE):")
    print(f"{'-'*50}")
    for _, row in best_products.iterrows():
        print(f"  Producto {row['producto_id']}: {row['model_type']} - RMSE: {row['test_rmse']:.4f}")
    
    print(f"\n🔻 TOP 5 PRODUCTOS MÁS DIFÍCILES (mayor RMSE):")
    print(f"{'-'*50}")
    for _, row in worst_products.iterrows():
        print(f"  Producto {row['producto_id']}: {row['model_type']} - RMSE: {row['test_rmse']:.4f}")
    
    # 4. Estadísticas generales
    general_stats = {
        'total_products': len(best_models),
        'total_models_evaluated': len(df),
        'avg_rmse': best_models['test_rmse'].mean(),
        'median_rmse': best_models['test_rmse'].median(),
        'avg_mape': best_models['test_mape'].mean(),
        'avg_r2': best_models['test_r2'].mean()
    }
    
    report['general_statistics'] = general_stats
    
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"{'-'*50}")
    print(f"  Total productos: {general_stats['total_products']}")
    print(f"  Total modelos evaluados: {general_stats['total_models_evaluated']}")
    print(f"  RMSE promedio (mejores modelos): {general_stats['avg_rmse']:.4f}")
    print(f"  MAPE promedio (mejores modelos): {general_stats['avg_mape']:.2f}%")
    print(f"  R² promedio (mejores modelos): {general_stats['avg_r2']:.4f}")
    
    return report

def save_consolidated_results(best_models: pd.DataFrame, 
                            all_models: pd.DataFrame,
                            output_dir: str = 'output') -> None:
    """
    Guarda los resultados consolidados en archivos CSV.
    
    Args:
        best_models: DataFrame con mejores modelos por producto
        all_models: DataFrame con todos los modelos
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar mejores modelos por producto
    best_path = os.path.join(output_dir, 'mejor_modelo_por_producto.csv')
    best_models.to_csv(best_path, index=False)
    
    # Guardar todos los modelos consolidados
    all_path = os.path.join(output_dir, 'todos_los_modelos_consolidados.csv')
    all_models.to_csv(all_path, index=False)
    
    print(f"\n💾 ARCHIVOS GUARDADOS:")
    print(f"{'-'*50}")
    print(f"  ✅ Mejores modelos: {best_path}")
    print(f"  ✅ Todos los modelos: {all_path}")

def create_comparison_visualizations(df: pd.DataFrame, 
                                   best_models: pd.DataFrame,
                                   save_plots: bool = True,
                                   output_dir: str = 'output') -> None:
    """
    Crea visualizaciones para comparar modelos.
    
    Args:
        df: DataFrame consolidado con todos los modelos
        best_models: DataFrame con mejores modelos por producto
        save_plots: Si guardar los gráficos
        output_dir: Directorio de salida
    """
    plt.style.use('default')
    
    # 1. Distribución de modelos ganadores
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico de barras de distribución
    model_counts = best_models['model_type'].value_counts()
    axes[0, 0].bar(model_counts.index, model_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title('Distribución de Modelos Ganadores por Producto', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Número de Productos')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Gráfico de pastel
    axes[0, 1].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Proporción de Modelos Ganadores', fontsize=14, fontweight='bold')
    
    # Distribución de RMSE por tipo de modelo
    for model_type in df['model_type'].unique():
        model_data = df[df['model_type'] == model_type]['test_rmse']
        axes[1, 0].hist(model_data, alpha=0.7, label=model_type, bins=20)
    
    axes[1, 0].set_title('Distribución de RMSE por Tipo de Modelo', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('RMSE')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].legend()
    
    # Boxplot de RMSE por tipo de modelo
    df.boxplot(column='test_rmse', by='model_type', ax=axes[1, 1])
    axes[1, 1].set_title('Comparación de RMSE por Tipo de Modelo', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Tipo de Modelo')
    axes[1, 1].set_ylabel('RMSE')
    
    plt.tight_layout()
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'model_comparison_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Gráficos guardados: {plot_path}")
    
    plt.show()

def run_complete_model_comparison(output_dir: str = 'output',
                                criterion: str = 'test_rmse',
                                save_results: bool = True,
                                create_plots: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Ejecuta el proceso completo de comparación de modelos.
    
    Args:
        output_dir: Directorio con los archivos de resultados
        criterion: Criterio para seleccionar mejor modelo
        save_results: Si guardar los resultados consolidados
        create_plots: Si crear visualizaciones
        
    Returns:
        Tupla con (mejores_modelos, todos_los_modelos, reporte)
    """
    print(f"🚀 INICIANDO COMPARACIÓN COMPLETA DE MODELOS")
    print(f"{'='*80}")
    
    # 1. Cargar resultados
    model_results = load_model_results(output_dir)
    
    if not model_results:
        print("❌ No se encontraron archivos de resultados de modelos")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # 2. Estandarizar y consolidar
    all_models = standardize_model_results(model_results)
    
    if all_models.empty:
        print("❌ No se pudieron consolidar los resultados")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # 3. Calcular puntajes compuestos si es necesario
    if criterion == 'composite_score':
        all_models = calculate_composite_score(all_models)
    
    # 4. Seleccionar mejores modelos
    best_models = get_best_model_per_product(all_models, criterion)
    
    # 5. Generar reporte
    report = generate_model_comparison_report(all_models, best_models)
    
    # 6. Guardar resultados
    if save_results:
        save_consolidated_results(best_models, all_models, output_dir)
    
    # 7. Crear visualizaciones
    if create_plots:
        create_comparison_visualizations(all_models, best_models, save_results, output_dir)
    
    print(f"\n🎉 COMPARACIÓN DE MODELOS COMPLETADA EXITOSAMENTE")
    print(f"{'='*80}")
    
    return best_models, all_models, report
