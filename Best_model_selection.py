"""
Best Model Selection Script
===========================

Este script compara el desempeño de todos los modelos entrenados para cada producto,
selecciona el mejor modelo basado en RMSE y MAPE, y genera reportes visuales.

Funcionalidades:
- Calcula RMSE y MAPE para cada modelo y producto
- Genera tabla resumen con desempeño promedio por tipo de modelo
- Crea gráfico de barras comparativo
- Encuentra el mejor modelo por producto
- Genera mapping de productos a mejores modelos
- Copia mejores modelos a carpeta de modelos registrados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import shutil
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

def calcular_metricas(y_true, y_pred):
    """
    Calcula RMSE y MAPE entre valores reales y predichos
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        tuple: (rmse, mape)
    """
    # Filtrar valores nulos
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return np.nan, np.nan
    
    # Calcular RMSE
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    # Calcular MAPE evitando división por cero
    y_true_nonzero = y_true_clean[y_true_clean != 0]
    y_pred_nonzero = y_pred_clean[y_true_clean != 0]
    
    if len(y_true_nonzero) > 0:
        mape = mean_absolute_percentage_error(y_true_nonzero, y_pred_nonzero) * 100
    else:
        mape = np.nan
    
    return rmse, mape

def cargar_predicciones_desde_output(output_dir='output'):
    """
    Carga todas las predicciones desde la carpeta output
    
    Args:
        output_dir: Directorio donde están las predicciones
    
    Returns:
        dict: Diccionario con resultados por producto y modelo
    """
    resultados = {}
    
    # Buscar todos los archivos de predicciones
    pattern = os.path.join(output_dir, 'test_predicciones_producto_*_modelo_*.csv')
    archivos = glob.glob(pattern)
    
    print(f"📁 Encontrados {len(archivos)} archivos de predicciones")
    
    for archivo in archivos:
        try:
            # Extraer información del nombre del archivo
            nombre_archivo = os.path.basename(archivo)
            # Formato: test_predicciones_producto_{id}_modelo_{tipo}.csv
            partes = nombre_archivo.replace('test_predicciones_producto_', '').replace('.csv', '').split('_modelo_')
            producto_id = int(partes[0])
            tipo_modelo = partes[1]
            
            # Cargar datos
            df = pd.read_csv(archivo)
            
            # Verificar columnas necesarias - manejar diferentes formatos
            columnas_reales = None
            columnas_predichas = None
            
            # Formato estándar (ARIMA, LSTM, Prophet, etc.)
            if 'demanda_real' in df.columns and 'demanda_predicha' in df.columns:
                columnas_reales = df['demanda_real']
                columnas_predichas = df['demanda_predicha']
            # Formato Lasso/Ridge
            elif 'y_real' in df.columns and 'y_pred' in df.columns:
                columnas_reales = df['y_real']
                columnas_predichas = df['y_pred']
            
            if columnas_reales is not None and columnas_predichas is not None:
                # Calcular métricas
                rmse, mape = calcular_metricas(columnas_reales, columnas_predichas)
                
                # Almacenar resultados
                if producto_id not in resultados:
                    resultados[producto_id] = {}
                
                resultados[producto_id][tipo_modelo] = {
                    'rmse': rmse,
                    'mape': mape,
                    'archivo': archivo,
                    'n_observaciones': len(df)
                }
                
                print(f"✅ Producto {producto_id} - {tipo_modelo}: RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            else:
                columnas_disponibles = list(df.columns)
                print(f"⚠️  Columnas faltantes en {archivo}. Columnas disponibles: {columnas_disponibles}")
                
        except Exception as e:
            print(f"❌ Error procesando {archivo}: {e}")
    
    return resultados

def crear_tabla_resumen_por_modelo(resultados):
    """
    Crea tabla resumen con desempeño promedio por tipo de modelo
    
    Args:
        resultados: Diccionario con resultados por producto y modelo
    
    Returns:
        DataFrame: Tabla resumen ordenada por desempeño
    """
    datos_resumen = []
    
    # Recopilar todas las métricas por tipo de modelo
    metricas_por_modelo = {}
    
    for producto_id, modelos in resultados.items():
        for tipo_modelo, metricas in modelos.items():
            if tipo_modelo not in metricas_por_modelo:
                metricas_por_modelo[tipo_modelo] = {'rmse': [], 'mape': []}
            
            if not (np.isnan(metricas['rmse']) or np.isnan(metricas['mape'])):
                metricas_por_modelo[tipo_modelo]['rmse'].append(metricas['rmse'])
                metricas_por_modelo[tipo_modelo]['mape'].append(metricas['mape'])
    
    # Calcular estadísticas por modelo
    for tipo_modelo, valores in metricas_por_modelo.items():
        if len(valores['rmse']) > 0:
            datos_resumen.append({
                'Tipo_Modelo': tipo_modelo,
                'RMSE_Promedio': np.mean(valores['rmse']),
                'RMSE_Std': np.std(valores['rmse']),
                'MAPE_Promedio': np.mean(valores['mape']),
                'MAPE_Std': np.std(valores['mape']),
                'Num_Productos': len(valores['rmse']),
                'RMSE_Min': np.min(valores['rmse']),
                'RMSE_Max': np.max(valores['rmse'])
            })
    
    # Crear DataFrame y ordenar por RMSE promedio (menor es mejor)
    df_resumen = pd.DataFrame(datos_resumen)
    df_resumen = df_resumen.sort_values('RMSE_Promedio', ascending=True)
    df_resumen['Ranking'] = range(1, len(df_resumen) + 1)
    
    return df_resumen

def crear_grafico_comparacion_modelos(df_resumen, output_dir='output'):
    """
    Crea gráfico de barras comparando el desempeño de los modelos
    
    Args:
        df_resumen: DataFrame con resumen por modelo
        output_dir: Directorio donde guardar el gráfico
    """
    # Configurar el estilo
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colores para los diferentes tipos de modelo
    colores = plt.cm.Set3(np.linspace(0, 1, len(df_resumen)))
    
    # Gráfico 1: RMSE Promedio
    bars1 = ax1.bar(range(len(df_resumen)), df_resumen['RMSE_Promedio'], 
                    color=colores, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Tipo de Modelo', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE Promedio', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de Modelos por RMSE\n(Menor es Mejor)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(df_resumen)))
    ax1.set_xticklabels(df_resumen['Tipo_Modelo'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Agregar barras de error para RMSE
    ax1.errorbar(range(len(df_resumen)), df_resumen['RMSE_Promedio'], 
                yerr=df_resumen['RMSE_Std'], fmt='none', color='black', 
                capsize=3, capthick=1)
    
    # Agregar valores en las barras
    for i, (bar, valor) in enumerate(zip(bars1, df_resumen['RMSE_Promedio'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + df_resumen.iloc[i]['RMSE_Std'],
                f'{valor:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Gráfico 2: MAPE Promedio
    bars2 = ax2.bar(range(len(df_resumen)), df_resumen['MAPE_Promedio'], 
                    color=colores, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Tipo de Modelo', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAPE Promedio (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Comparación de Modelos por MAPE\n(Menor es Mejor)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(df_resumen)))
    ax2.set_xticklabels(df_resumen['Tipo_Modelo'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar barras de error para MAPE
    ax2.errorbar(range(len(df_resumen)), df_resumen['MAPE_Promedio'], 
                yerr=df_resumen['MAPE_Std'], fmt='none', color='black', 
                capsize=3, capthick=1)
    
    # Agregar valores en las barras
    for i, (bar, valor) in enumerate(zip(bars2, df_resumen['MAPE_Promedio'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + df_resumen.iloc[i]['MAPE_Std'],
                f'{valor:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    # Guardar gráfico
    ruta_grafico = os.path.join(output_dir, 'comparacion_modelos_desempeno.png')
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: {ruta_grafico}")
    
    plt.show()

def encontrar_mejor_modelo_por_producto(resultados):
    """
    Encuentra el mejor modelo para cada producto basado en RMSE
    
    Args:
        resultados: Diccionario con resultados por producto y modelo
    
    Returns:
        dict: Diccionario con el mejor modelo por producto
    """
    mejores_modelos = {}
    
    for producto_id, modelos in resultados.items():
        mejor_rmse = float('inf')
        mejor_modelo = None
        mejor_info = None
        
        for tipo_modelo, metricas in modelos.items():
            if not np.isnan(metricas['rmse']) and metricas['rmse'] < mejor_rmse:
                mejor_rmse = metricas['rmse']
                mejor_modelo = tipo_modelo
                mejor_info = metricas
        
        if mejor_modelo is not None:
            mejores_modelos[producto_id] = {
                'modelo': mejor_modelo,
                'rmse': mejor_rmse,
                'mape': mejor_info['mape'],
                'archivo': mejor_info['archivo']
            }
            print(f"🏆 Producto {producto_id}: Mejor modelo = {mejor_modelo} (RMSE={mejor_rmse:.2f})")
    
    return mejores_modelos

def crear_mapping_dataframe(mejores_modelos, modelos_dir='Modelos registrados'):
    """
    Crea DataFrame con mapping de productos a mejores modelos
    
    Args:
        mejores_modelos: Diccionario con mejores modelos por producto
        modelos_dir: Directorio donde están los modelos guardados
    
    Returns:
        DataFrame: Mapping de productos a archivos de modelos
    """
    datos_mapping = []
    
    # Listar todos los archivos de modelos disponibles
    archivos_modelos = []
    for extension in ['*.pkl', '*.json']:
        archivos_modelos.extend(glob.glob(os.path.join(modelos_dir, extension)))
    
    for producto_id, info in mejores_modelos.items():
        tipo_modelo = info['modelo']
        
        # Buscar archivo correspondiente del modelo
        archivo_modelo = None
        
        # Patrones posibles para el nombre del archivo
        patrones = [
            f"best_model_producto_{producto_id}_{tipo_modelo}.pkl",
            f"{tipo_modelo}_model_producto_{producto_id}.json",
            f"prophet_model_producto_{producto_id}.json" if tipo_modelo == 'prophet' else None
        ]
        
        for patron in patrones:
            if patron is not None:
                ruta_completa = os.path.join(modelos_dir, patron)
                if os.path.exists(ruta_completa):
                    archivo_modelo = patron
                    break
        
        if archivo_modelo is None:
            # Buscar por patrón más flexible
            for archivo in archivos_modelos:
                nombre_archivo = os.path.basename(archivo)
                if f"producto_{producto_id}" in nombre_archivo and tipo_modelo in nombre_archivo:
                    archivo_modelo = nombre_archivo
                    break
        
        if archivo_modelo is not None:
            datos_mapping.append({
                'id_producto': producto_id,
                'mejor_modelo': archivo_modelo,
                'tipo_modelo': tipo_modelo,
                'rmse': info['rmse'],
                'mape': info['mape']
            })
        else:
            print(f"⚠️  No se encontró archivo de modelo para Producto {producto_id} - {tipo_modelo}")
    
    # Crear DataFrame
    df_mapping = pd.DataFrame(datos_mapping)
    df_mapping = df_mapping.sort_values('id_producto')
    
    return df_mapping

def guardar_reportes(df_resumen, df_mapping, mejores_modelos, output_dir='output'):
    """
    Guarda todos los reportes en archivos CSV
    
    Args:
        df_resumen: DataFrame con resumen por modelo
        df_mapping: DataFrame con mapping de productos
        mejores_modelos: Diccionario con mejores modelos
        output_dir: Directorio donde guardar los reportes
    """
    # Guardar resumen de modelos
    ruta_resumen = os.path.join(output_dir, 'resumen_comparacion_modelos.csv')
    df_resumen.to_csv(ruta_resumen, index=False)
    print(f"📄 Resumen de modelos guardado en: {ruta_resumen}")
    
    # Guardar mapping de mejores modelos
    ruta_mapping = os.path.join(output_dir, 'mapping_mejores_modelos.csv')
    df_mapping.to_csv(ruta_mapping, index=False)
    print(f"📄 Mapping de mejores modelos guardado en: {ruta_mapping}")
    
    # Crear reporte detallado por producto
    datos_detalle = []
    for producto_id, modelos in mejores_modelos.items():
        datos_detalle.append({
            'producto_id': producto_id,
            'mejor_modelo': modelos['modelo'],
            'mejor_rmse': modelos['rmse'],
            'mejor_mape': modelos['mape'],
            'archivo_prediccion': os.path.basename(modelos['archivo'])
        })
    
    df_detalle = pd.DataFrame(datos_detalle)
    ruta_detalle = os.path.join(output_dir, 'mejores_modelos_detalle.csv')
    df_detalle.to_csv(ruta_detalle, index=False)
    print(f"📄 Detalle de mejores modelos guardado en: {ruta_detalle}")

def mostrar_codigo_mapping(df_mapping):
    """
    Genera el código de mapping para usar en el notebook
    
    Args:
        df_mapping: DataFrame con mapping de productos
    """
    print("\n" + "="*60)
    print("📋 CÓDIGO PARA COPIAR EN EL NOTEBOOK:")
    print("="*60)
    
    # Generar código del mapping
    ids_productos = df_mapping['id_producto'].tolist()
    nombres_modelos = df_mapping['mejor_modelo'].tolist()
    
    codigo = f"""modelo_mapping = pd.DataFrame({{
    'id_producto': {ids_productos},
    'mejor_modelo': {nombres_modelos}
}})"""
    
    print(codigo)
    print("="*60)
    
    return codigo

def ejecutar_analisis_completo(output_dir='output', modelos_dir='Modelos registrados'):
    """
    Ejecuta el análisis completo de selección de mejores modelos
    
    Args:
        output_dir: Directorio con las predicciones
        modelos_dir: Directorio con los modelos guardados
    
    Returns:
        tuple: (df_resumen, df_mapping, mejores_modelos, codigo_mapping)
    """
    print("🚀 INICIANDO ANÁLISIS DE SELECCIÓN DE MEJORES MODELOS")
    print("="*60)
    
    # 1. Cargar todas las predicciones
    print("\n1️⃣ Cargando predicciones...")
    resultados = cargar_predicciones_desde_output(output_dir)
    
    if not resultados:
        print("❌ No se encontraron predicciones para analizar")
        return None, None, None, None
    
    print(f"✅ Cargados resultados para {len(resultados)} productos")
    
    # 2. Crear tabla resumen por modelo
    print("\n2️⃣ Creando tabla resumen por modelo...")
    df_resumen = crear_tabla_resumen_por_modelo(resultados)
    print("✅ Tabla resumen creada:")
    print(df_resumen[['Tipo_Modelo', 'RMSE_Promedio', 'MAPE_Promedio', 'Num_Productos', 'Ranking']])
    
    # 3. Crear gráfico comparativo
    print("\n3️⃣ Creando gráfico comparativo...")
    crear_grafico_comparacion_modelos(df_resumen, output_dir)
    
    # 4. Encontrar mejores modelos por producto
    print("\n4️⃣ Encontrando mejor modelo por producto...")
    mejores_modelos = encontrar_mejor_modelo_por_producto(resultados)
    
    # 5. Crear mapping DataFrame
    print("\n5️⃣ Creando mapping de productos a modelos...")
    df_mapping = crear_mapping_dataframe(mejores_modelos, modelos_dir)
    print("✅ Mapping creado:")
    print(df_mapping)
    
    # 6. Guardar reportes
    print("\n6️⃣ Guardando reportes...")
    guardar_reportes(df_resumen, df_mapping, mejores_modelos, output_dir)
    
    # 7. Mostrar código para notebook
    print("\n7️⃣ Generando código para notebook...")
    codigo_mapping = mostrar_codigo_mapping(df_mapping)
    
    print("\n🎉 ¡ANÁLISIS COMPLETADO CON ÉXITO!")
    print("="*60)
    
    return df_resumen, df_mapping, mejores_modelos, codigo_mapping

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Ejecutar análisis completo
    df_resumen, df_mapping, mejores_modelos, codigo_mapping = ejecutar_analisis_completo()
    
    # Mostrar estadísticas finales
    if df_resumen is not None:
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   • Total de tipos de modelo evaluados: {len(df_resumen)}")
        print(f"   • Mejor modelo general: {df_resumen.iloc[0]['Tipo_Modelo']} (RMSE={df_resumen.iloc[0]['RMSE_Promedio']:.2f})")
        print(f"   • Total de productos con modelo seleccionado: {len(df_mapping)}")
        
        # Distribución de mejores modelos
        distribucion = df_mapping['tipo_modelo'].value_counts()
        print(f"\n📈 Distribución de mejores modelos:")
        for modelo, cantidad in distribucion.items():
            print(f"   • {modelo}: {cantidad} productos")
