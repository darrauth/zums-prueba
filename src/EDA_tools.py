import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

def format_millions_axis(ax, axis='y'):
    """
    Formatea el eje especificado para mostrar valores en millones con 'M'
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        El eje a formatear
    axis : str
        'y' para eje Y, 'x' para eje X
    """
    if axis == 'y':
        # Obtener los valores actuales del eje Y
        ylims = ax.get_ylim()
        if ylims[1] >= 1000000:  # Solo formatear si los valores son >= 1 millón
            # Formatear las etiquetas del eje Y
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000000:.1f}M'))
            return True
    elif axis == 'x':
        xlims = ax.get_xlim()
        if xlims[1] >= 1000000:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000000:.1f}M'))
            return True
    return False

def missing_values(df, name):
    print(f"=== {name} ===")
    print(df.isna().sum())
    print('-' * 20)


def verificar_duplicados(df, nombre_tabla, columna_llave=None):
    """
    Función combinada para verificar duplicados.
    
    Parámetros:
    - df: DataFrame a analizar
    - nombre_tabla: Nombre descriptivo de la tabla
    - columna_llave: Columna específica para verificar duplicados. 
                     Si es None, verifica duplicados completos (todas las columnas)
    """
    
    if columna_llave is None:
        # Verificar duplicados completos (todas las columnas)
        print(f"=== Duplicados completos - {nombre_tabla} ===")
        print(f"Filas totales: {len(df)}")
        
        duplicados = df.duplicated().sum()
        print(f"Duplicados encontrados: {duplicados}")
        
        if duplicados > 0:
            porcentaje = (duplicados / len(df)) * 100
            print(f"Porcentaje de duplicados: {porcentaje:.2f}%")
            
            # Mostrar todas las filas duplicadas (incluyendo originales)
            filas_duplicadas = df[df.duplicated(keep=False)]
            print(f"\nTodas las filas duplicadas ({len(filas_duplicadas)} filas):")
            print(filas_duplicadas.sort_values(list(df.columns)))
        else:
            print("✅ No se encontraron duplicados completos")
    
    else:
        # Verificar duplicados por columna específica (llave)
        print(f"=== Duplicados por '{columna_llave}' - {nombre_tabla} ===")
        
        if columna_llave not in df.columns:
            print(f"❌ Error: La columna '{columna_llave}' no existe en {nombre_tabla}")
            print(f"Columnas disponibles: {list(df.columns)}")
            return
        
        print(f"Valores únicos en '{columna_llave}': {df[columna_llave].nunique()}")
        print(f"Filas totales: {len(df)}")
        
        duplicados = df.duplicated(subset=[columna_llave]).sum()
        print(f"Duplicados por '{columna_llave}': {duplicados}")
        
        if duplicados > 0:
            porcentaje = (duplicados / len(df)) * 100
            print(f"Porcentaje de duplicados: {porcentaje:.2f}%")
            
            # Mostrar todas las filas con valores duplicados en la columna
            filas_duplicadas = df[df.duplicated(subset=[columna_llave], keep=False)]
            print(f"\nTodas las filas con '{columna_llave}' duplicado ({len(filas_duplicadas)} filas):")
            print(filas_duplicadas.sort_values([columna_llave]))
            
            # Mostrar conteo de valores duplicados
            print(f"\nConteo de valores duplicados en '{columna_llave}':")
            valores_duplicados = df[df.duplicated(subset=[columna_llave], keep=False)][columna_llave].value_counts()
            print(valores_duplicados)
        else:
            print(f"✅ No se encontraron duplicados por '{columna_llave}'")
    
    print("-" * 50)


# Funciones para análisis univariado de variables categóricas

def plot_categorical_univariate(df_cat, categorical_vars, figsize=(20, 15), style='default', palette='husl'):
    """
    Crear gráficos de barras para análisis univariado de variables categóricas
    
    Parameters:
    -----------
    df_cat : pandas.DataFrame
        DataFrame con variables categóricas
    categorical_vars : list
        Lista de nombres de variables categóricas
    figsize : tuple
        Tamaño de la figura (ancho, alto)
    style : str
        Estilo de matplotlib
    palette : str
        Paleta de colores de seaborn
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    # Configurar el estilo de los gráficos
    plt.style.use(style)
    sns.set_palette(palette)
    
    # Calcular dimensiones de la cuadrícula
    n_vars = len(categorical_vars)
    n_rows = (n_vars + 2) // 3  # Redondear hacia arriba
    n_cols = min(3, n_vars)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_vars > 1 else [axes]
    
    for i, var in enumerate(categorical_vars):
        # Calcular frecuencias
        value_counts = df_cat[var].value_counts()
        prop_counts = df_cat[var].value_counts(normalize=True)
        
        # Crear gráfico de barras
        ax = axes[i]
        bars = ax.bar(range(len(value_counts)), value_counts.values, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(value_counts))))
        
        # Configurar el gráfico
        ax.set_title(f'Distribución de {var}', fontsize=12, fontweight='bold')
        ax.set_xlabel(var, fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        
        # Rotar etiquetas si son muchas categorías
        if len(value_counts) > 5:
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        else:
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index)
        
        # Agregar valores en las barras
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{int(height)}\n({prop_counts.iloc[j]:.1%})',
                   ha='center', va='bottom', fontsize=8)
    
    # Eliminar ejes vacíos
    for i in range(len(categorical_vars), len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    return fig, axes


def categorical_univariate_summary(df_cat, categorical_vars, show_percentages=True):
    """
    Generar resumen estadístico de variables categóricas
    
    Parameters:
    -----------
    df_cat : pandas.DataFrame
        DataFrame con variables categóricas
    categorical_vars : list
        Lista de nombres de variables categóricas
    show_percentages : bool
        Si mostrar porcentajes además de proporciones
    
    Returns:
    --------
    dict : Diccionario con resúmenes por variable
    """
    summary_results = {}
    
    print("\n" + "="*80)
    print("RESUMEN ESTADÍSTICO - ANÁLISIS UNIVARIADO")
    print("="*80)
    
    for var in categorical_vars:
        print(f"\n{var.upper()}:")
        print("-" * 50)
        
        value_counts = df_cat[var].value_counts()
        prop_counts = df_cat[var].value_counts(normalize=True)
        
        if show_percentages:
            summary_df = pd.DataFrame({
                'Frecuencia': value_counts,
                'Proporción': prop_counts,
                'Porcentaje': prop_counts * 100
            }).round(3)
        else:
            summary_df = pd.DataFrame({
                'Frecuencia': value_counts,
                'Proporción': prop_counts
            }).round(3)
        
        print(summary_df)
        summary_results[var] = summary_df
    
    return summary_results





def plot_time_series_with_competitor(data, date_col='date', value_col='demanda', 
                                    competitor_date='2021-07-02', title="Serie de Tiempo - Demanda", 
                                    ax=None, show_competitor_line=True):
    """
    Grafica una serie de tiempo con línea vertical marcando la entrada del competidor
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame con los datos
    date_col : str
        Nombre de la columna de fecha
    value_col : str
        Nombre de la columna de valores a graficar
    competitor_date : str
        Fecha de entrada del competidor en formato 'YYYY-MM-DD'
    title : str
        Título del gráfico
    ax : matplotlib.axes
        Axes object para el gráfico
    show_competitor_line : bool
        Si mostrar la línea vertical del competidor
    
    Returns:
    --------
    ax : matplotlib.axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Preparar datos
    data_sorted = data.sort_values(date_col)
    
    # Crear el gráfico principal
    ax.plot(data_sorted[date_col], data_sorted[value_col], 
           linewidth=2, color='steelblue', label='Demanda')
    
    # Agregar línea vertical para marcar entrada del competidor si se solicita
    if show_competitor_line:
        fecha_competidor = pd.to_datetime(competitor_date)
        ax.axvline(x=fecha_competidor, color='red', linestyle='--', 
                  alpha=0.8, linewidth=2, label='Entrada Competidor')
        ax.text(fecha_competidor, ax.get_ylim()[1]*0.9, 'Entrada\nCompetidor', 
                ha='left', va='top', fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Configuración del gráfico
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    
    # Formatear eje Y en millones si es necesario
    if format_millions_axis(ax, 'y'):
        ax.set_ylabel('Demanda (Millones)', fontsize=12)
    else:
        ax.set_ylabel('Demanda', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Formato de fechas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    return ax

def plot_percentage_variation(data, date_col='date', value_col='demanda', title="Variación Porcentual", ax=None):
    """
    Grafica la variación porcentual de una serie de tiempo
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calcular variación porcentual
    data_sorted = data.sort_values(date_col)
    pct_change = data_sorted[value_col].pct_change() * 100
    
    # Crear el gráfico
    ax.plot(data_sorted[date_col], pct_change, linewidth=2, color='steelblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Variación Porcentual (%)')
    ax.grid(True, alpha=0.3)
    
    # Formato de fechas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    return ax

def plot_seasonal_decomposition(data, date_col='date', value_col='demanda', period=12, title="Descomposición Estacional", ax=None):
    """
    Realiza y grafica la descomposición estacional de una serie de tiempo
    """
    # Preparar datos
    data_sorted = data.sort_values(date_col).reset_index(drop=True)
    ts = data_sorted.set_index(date_col)[value_col]
    
    # Asegurar frecuencia mensual
    ts = ts.resample('M').mean()
    
    # Descomposición estacional
    try:
        decomposition = seasonal_decompose(ts, model='additive', period=period)
        
        if ax is None:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        else:
            # Si se pasa un ax, crear subplots dentro de él
            gs = ax.figure.add_gridspec(4, 1, hspace=0.3)
            axes = [ax.figure.add_subplot(gs[i, 0]) for i in range(4)]
        
        # Gráficos
        decomposition.observed.plot(ax=axes[0], title=f'{title} - Serie Original', color='steelblue')
        decomposition.trend.plot(ax=axes[1], title='Tendencia', color='orange')
        decomposition.seasonal.plot(ax=axes[2], title='Estacionalidad', color='green')
        decomposition.resid.plot(ax=axes[3], title='Residuos', color='red')
        
        for ax_sub in axes:
            ax_sub.grid(True, alpha=0.3)
            ax_sub.tick_params(axis='x', rotation=45)
            
        return axes, decomposition
    
    except Exception as e:
        print(f"Error en descomposición estacional: {e}")
        return None, None

def plot_seasonal_pattern(data, date_col='date', value_col='demanda', title="Patrón Estacional por Años", ax=None, aggregation='sum'):
    """
    Grafica el patrón estacional con cada año como una línea separada
    
    Parameters:
    -----------
    aggregation : str
        Tipo de agregación ('sum' para suma, 'mean' para promedio)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Preparar datos
    data_copy = data.copy()
    data_copy[date_col] = pd.to_datetime(data_copy[date_col])
    data_copy['year'] = data_copy[date_col].dt.year
    data_copy['month'] = data_copy[date_col].dt.month
    
    # Agrupar por año y mes con la agregación especificada
    if aggregation == 'sum':
        seasonal_data = data_copy.groupby(['year', 'month'])[value_col].sum().reset_index()
        ylabel = 'Demanda Total'
    else:
        seasonal_data = data_copy.groupby(['year', 'month'])[value_col].mean().reset_index()
        ylabel = 'Demanda Promedio'
    
    # Crear gráfico por año
    years = sorted(seasonal_data['year'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
    
    for i, year in enumerate(years):
        year_data = seasonal_data[seasonal_data['year'] == year]
        ax.plot(year_data['month'], year_data[value_col], 
               marker='o', label=str(year), color=colors[i], linewidth=2)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Mes')
    
    # Formatear eje Y en millones si es necesario
    if format_millions_axis(ax, 'y'):
        if aggregation == 'sum':
            ylabel = 'Demanda Total (Millones)'
        else:
            ylabel = 'Demanda Promedio (Millones)'
    
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                       'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_autocorrelation(data, date_col='date', value_col='demanda', lags=20, title="Autocorrelación", ax=None):
    """
    Grafica la función de autocorrelación (ACF)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Preparar datos
    data_sorted = data.sort_values(date_col)
    ts = data_sorted[value_col].dropna()
    
    # Validar que tenemos suficientes datos
    if len(ts) < 10:
        ax.text(0.5, 0.5, f'Datos insuficientes\n({len(ts)} observaciones)\nMínimo: 10', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        return ax
    
    # Ajustar lags si es necesario
    max_lags = min(lags, len(ts) // 2 - 1)
    if max_lags <= 0:
        max_lags = 1
    
    try:
        # Calcular ACF
        plot_acf(ts, lags=max_lags, ax=ax, alpha=0.05)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return ax
    except Exception as e:
        ax.text(0.5, 0.5, f'Error en ACF:\n{str(e)[:50]}...', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')
        ax.set_title(title, fontsize=12, fontweight='bold')
        return ax

def plot_partial_autocorrelation(data, date_col='date', value_col='demanda', lags=20, title="Autocorrelación Parcial", ax=None):
    """
    Grafica la función de autocorrelación parcial (PACF)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Preparar datos
    data_sorted = data.sort_values(date_col)
    ts = data_sorted[value_col].dropna()
    
    # Validar que tenemos suficientes datos
    if len(ts) < 10:
        ax.text(0.5, 0.5, f'Datos insuficientes\n({len(ts)} observaciones)\nMínimo: 10', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        return ax
    
    # Ajustar lags si es necesario
    max_lags = min(lags, len(ts) // 2 - 1)
    if max_lags <= 0:
        max_lags = 1
    
    try:
        # Calcular PACF
        plot_pacf(ts, lags=max_lags, ax=ax, alpha=0.05)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return ax
    except Exception as e:
        ax.text(0.5, 0.5, f'Error en PACF:\n{str(e)[:50]}...', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')
        ax.set_title(title, fontsize=12, fontweight='bold')
        return ax


# Nueva función para análisis de estacionalidad por categorías

def plot_seasonal_analysis_by_column(data, group_col='categoria', date_col='date', value_col='demanda', 
                                        max_plots_per_subplot=20, figsize_per_plot=(5, 4),
                                        competitor_date='2021-07-02'):
    """
    Crea gráficos de estacionalidad para todas las categorías (o grupos), mostrando
    cada año como una línea diferente con indicación del período pre/post competidor.
    Los gráficos se ordenan por variación porcentual (de mayor variación negativa a menor).
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame con los datos
    group_col : str
        Columna para agrupar (ej: 'categoria', 'id_producto')
    date_col : str
        Nombre de la columna de fecha
    value_col : str
        Nombre de la columna de valores
    max_plots_per_subplot : int
        Máximo número de gráficos por bloque de subplots
    figsize_per_plot : tuple
        Tamaño base por cada subplot individual
    competitor_date : str
        Fecha de entrada del competidor en formato 'YYYY-MM-DD'
    
    Returns:
    --------
    list : Lista de figuras matplotlib generadas
    """
    
    # Preparar datos
    data_copy = data.copy()
    data_copy[date_col] = pd.to_datetime(data_copy[date_col])
    fecha_competidor = pd.to_datetime(competitor_date)
    competitor_year = fecha_competidor.year
    
    # Crear períodos pre y post competidor
    data_copy['periodo'] = data_copy[date_col].apply(
        lambda x: 'Pre-Competidor' if x < fecha_competidor else 'Post-Competidor'
    )
    
    # Calcular variación porcentual para cada grupo y ordenar
    group_variations = []
    unique_groups = data_copy[group_col].unique()
    
    for group in unique_groups:
        group_data = data_copy[data_copy[group_col] == group]
        
        # Calcular demanda promedio pre y post competidor
        pre_mean = group_data[group_data['periodo'] == 'Pre-Competidor'][value_col].mean()
        post_mean = group_data[group_data['periodo'] == 'Post-Competidor'][value_col].mean()
        
        # Calcular variación porcentual
        if not np.isnan(pre_mean) and not np.isnan(post_mean) and pre_mean != 0:
            variation_pct = ((post_mean - pre_mean) / pre_mean) * 100
        else:
            variation_pct = 0
            
        group_variations.append((group, variation_pct))
    
    # Ordenar por variación porcentual (de mayor variación negativa a menor variación negativa)
    group_variations.sort(key=lambda x: x[1])
    groups = [group for group, _ in group_variations]
    n_groups = len(groups)
    
    print(f"Generando análisis de estacionalidad para {n_groups} {group_col}s")
    print(f"Máximo de gráficos por bloque: {max_plots_per_subplot}")
    print(f"Ordenamiento: De mayor variación negativa ({group_variations[0][1]:.1f}%) a menor variación negativa ({group_variations[-1][1]:.1f}%)")
    
    # Dividir grupos ordenados en bloques
    group_blocks = [groups[i:i + max_plots_per_subplot] 
                   for i in range(0, len(groups), max_plots_per_subplot)]
    
    all_figures = []
    
    for block_idx, groups_block in enumerate(group_blocks):
        n_plots = len(groups_block)
        
        # Configurar grid de subplots dinámicamente
        if n_plots <= 4:
            rows, cols = 2, 2
        elif n_plots <= 6:
            rows, cols = 2, 3
        elif n_plots <= 9:
            rows, cols = 3, 3
        elif n_plots <= 12:
            rows, cols = 3, 4
        elif n_plots <= 16:
            rows, cols = 4, 4
        elif n_plots <= 20:
            rows, cols = 4, 5
        else:
            rows, cols = 5, 4
        
        # Calcular tamaño de figura
        figsize = (cols * figsize_per_plot[0], rows * figsize_per_plot[1])
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else axes
        
        # Título del bloque
        fig.suptitle(f'Análisis de Estacionalidad - {group_col.title()} - Bloque {block_idx + 1}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Crear gráficos para cada grupo
        for i, group in enumerate(groups_block):
            if i >= len(axes):
                break
                
            ax = axes[i]
            group_data = data_copy[data_copy[group_col] == group].copy()
            
            if len(group_data) == 0:
                ax.text(0.5, 0.5, f'No hay datos\npara {group}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{group}', fontsize=10, fontweight='bold')
                continue
            
            try:
                # Preparar datos para estacionalidad
                group_data['year'] = group_data[date_col].dt.year
                group_data['month'] = group_data[date_col].dt.month
                
                # Agrupar por año y mes
                seasonal_data = group_data.groupby(['year', 'month'])[value_col].sum().reset_index()
                
                if len(seasonal_data) == 0:
                    ax.text(0.5, 0.5, 'Sin datos suficientes', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    ax.set_title(f'{group}', fontsize=10, fontweight='bold')
                    continue
                
                # Crear gráfico por año
                years = sorted(seasonal_data['year'].unique())
                colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
                
                for j, year in enumerate(years):
                    year_data = seasonal_data[seasonal_data['year'] == year]
                    
                    # Determinar estilo de línea según período
                    if year < competitor_year:
                        linestyle = '-'
                        alpha = 0.7
                        marker = 'o'
                        markersize = 4
                        label = f'{year} (Pre)'
                    elif year == competitor_year:
                        linestyle = '-'
                        alpha = 1.0
                        marker = 's'
                        markersize = 5
                        label = f'{year} (Competidor)'
                    else:
                        linestyle = '--'
                        alpha = 0.8
                        marker = '^'
                        markersize = 4
                        label = f'{year} (Post)'
                    
                    ax.plot(year_data['month'], year_data[value_col], 
                           marker=marker, label=label, color=colors[j], 
                           linewidth=2, linestyle=linestyle, alpha=alpha, markersize=markersize)
                
                # Configurar el gráfico
                ax.set_title(f'{group}', fontsize=11, fontweight='bold')
                ax.set_xlabel('Mes', fontsize=9)
                ax.set_ylabel('Demanda', fontsize=9)
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(['E', 'F', 'M', 'A', 'M', 'J',
                                   'J', 'A', 'S', 'O', 'N', 'D'], fontsize=8)
                ax.tick_params(axis='y', labelsize=8)
                
                # Mostrar leyenda solo si hay espacio (menos de 6 años)
                if len(years) <= 6:
                    ax.legend(fontsize=7, loc='upper right')
                
                ax.grid(True, alpha=0.3)
                
                # Obtener la variación porcentual precalculada para este grupo
                group_variation_pct = next((var for grp, var in group_variations if grp == group), 0)
                
                # Mostrar variación porcentual en el gráfico
                impact_color = 'green' if group_variation_pct >= 0 else 'red'
                ax.text(0.02, 0.98, f'{group_variation_pct:+.1f}%', 
                       transform=ax.transAxes, fontsize=9, fontweight='bold',
                       verticalalignment='top', color=impact_color,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                       ha='center', va='center', transform=ax.transAxes, 
                       color='red', fontsize=9)
                ax.set_title(f'{group} (Error)', fontsize=10, fontweight='bold')
        
        # Ocultar ejes vacíos
        for i in range(len(groups_block), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        all_figures.append(fig)
        
        print(f"Bloque {block_idx + 1}/{len(group_blocks)} completado ({len(groups_block)} gráficos)")
    
    return all_figures


def plot_competitor_impact_analysis(data, date_col='date', value_col='demanda', 
                                   competitor_date='2021-07-02', group_col=None,
                                   title="Análisis de Impacto del Competidor", figsize=(16, 10)):
    """
    Crea un análisis completo del impacto del competidor con múltiples visualizaciones
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame con los datos
    date_col : str
        Nombre de la columna de fecha
    value_col : str
        Nombre de la columna de valores
    competitor_date : str
        Fecha de entrada del competidor
    group_col : str, optional
        Columna para agrupar los datos (ej: 'categoria', 'id_producto')
    title : str
        Título principal del análisis
    figsize : tuple
        Tamaño de la figura
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Preparar datos
    data_copy = data.copy()
    data_copy[date_col] = pd.to_datetime(data_copy[date_col])
    fecha_competidor = pd.to_datetime(competitor_date)
    
    # Crear períodos pre y post competidor
    data_copy['periodo'] = data_copy[date_col].apply(
        lambda x: 'Pre-Competidor' if x < fecha_competidor else 'Post-Competidor'
    )
    
    if group_col is None:
        # Análisis agregado - 3x3 grid reorganizado por tipos de análisis
        fig, axes = plt.subplots(3, 3, figsize=(figsize[0] * 1.2, figsize[1] * 1.3))
        
        # Ajustar espaciado
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # FILA 1: SERIES TEMPORALES
        # 1.1 Serie de tiempo completa (diaria)
        ax1 = axes[0, 0]
        plot_time_series_with_competitor(data_copy, date_col=date_col, value_col=value_col,
                                       competitor_date=competitor_date, 
                                       title="Serie Temporal Diaria", ax=ax1)
        
        # 1.2 Serie temporal mensualizada
        ax2 = axes[0, 1]
        data_monthly = data_copy.copy()
        data_monthly['year_month'] = data_monthly[date_col].dt.to_period('M')
        monthly_data = data_monthly.groupby('year_month')[value_col].sum()
        monthly_data.plot(kind='line', ax=ax2, linewidth=2, color='steelblue', marker='o', markersize=3)
        
        # Formatear eje Y en millones y actualizar título si es necesario
        if format_millions_axis(ax2, 'y'):
            ax2.set_title('Serie Temporal Mensual')
            ax2.set_ylabel('Demanda Mensual (Millones)')
        else:
            ax2.set_title('Serie Temporal Mensual')
            ax2.set_ylabel('Demanda Mensual')
            
        ax2.set_xlabel('Año-Mes')
        ax2.grid(True, alpha=0.3)
        # Añadir línea del competidor
        competitor_period = pd.Period(fecha_competidor, freq='M')
        if competitor_period in monthly_data.index:
            competitor_value = monthly_data[competitor_period]
            ax2.axvline(x=competitor_period.ordinal, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax2.text(competitor_period.ordinal, ax2.get_ylim()[1]*0.9, 'Entrada\nCompetidor', 
                    ha='left', va='top', fontsize=8, color='red', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax2.tick_params(axis='x', rotation=45)
        
        # 1.3 Serie temporal trimestral
        ax3 = axes[0, 2]
        data_quarterly = data_copy.copy()
        data_quarterly['year_quarter'] = data_quarterly[date_col].dt.to_period('Q')
        quarterly_data = data_quarterly.groupby('year_quarter')[value_col].sum()
        quarterly_data.plot(kind='line', ax=ax3, linewidth=2, color='darkgreen', marker='s', markersize=4)
        
        # Formatear eje Y en millones y actualizar título si es necesario
        if format_millions_axis(ax3, 'y'):
            ax3.set_title('Serie Temporal Trimestral')
            ax3.set_ylabel('Demanda Trimestral (Millones)')
        else:
            ax3.set_title('Serie Temporal Trimestral')
            ax3.set_ylabel('Demanda Trimestral')
            
        ax3.set_xlabel('Año-Trimestre')
        ax3.grid(True, alpha=0.3)
        # Añadir línea del competidor
        competitor_quarter = pd.Period(fecha_competidor, freq='Q')
        if competitor_quarter in quarterly_data.index:
            ax3.axvline(x=competitor_quarter.ordinal, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax3.text(competitor_quarter.ordinal, ax3.get_ylim()[1]*0.9, 'Entrada\nCompetidor', 
                    ha='left', va='top', fontsize=8, color='red', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax3.tick_params(axis='x', rotation=45)
        
        # FILA 2: ANÁLISIS DE DEMANDA
        # 2.1 Patrón de estacionalidad (una línea por año)
        ax4 = axes[1, 0]
        plot_seasonal_pattern(data_copy, date_col=date_col, value_col=value_col,
                            title="Patrón Estacional por Años", ax=ax4, aggregation='sum')
        
        # 2.2 Serie anual agregada (demanda total por año) - LÍNEAS
        ax5 = axes[1, 1]
        data_copy['año'] = data_copy[date_col].dt.year
        yearly_data = data_copy.groupby('año')[value_col].sum()
        
        # Crear gráfico de líneas en lugar de barras
        line = ax5.plot(yearly_data.index, yearly_data.values, 
                       linewidth=3, color='steelblue', marker='o', markersize=8, alpha=0.8)
        ax5.set_title('Demanda Total por Año')
        ax5.set_xlabel('Año')
        ax5.set_ylabel('Demanda Total')
        ax5.grid(True, alpha=0.3)
        
        # Ajustar el eje X para evitar superposición
        ax5.set_xticks(yearly_data.index)
        ax5.set_xticklabels([str(int(year)) for year in yearly_data.index], rotation=0)
        
        # Marcar año del competidor
        competitor_year = fecha_competidor.year
        if competitor_year in yearly_data.index:
            # Resaltar el punto del año del competidor
            competitor_value = yearly_data[competitor_year]
            ax5.plot(competitor_year, competitor_value, 'o', color='red', markersize=12, 
                    markerfacecolor='lightcoral', markeredgecolor='red', markeredgewidth=2)
            ax5.text(competitor_year, competitor_value + yearly_data.max()*0.05,
                    'Entrada\nCompetidor', ha='center', va='bottom', fontsize=8, 
                    color='red', fontweight='bold')
        
        # Agregar valores en los puntos
        for year, value in yearly_data.items():
            ax5.text(year, value + yearly_data.max()*0.01, f'{value:.0f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2.3 Evolución mensual promedio
        ax6 = axes[1, 2]
        data_copy['mes'] = data_copy[date_col].dt.month
        monthly_evolution = data_copy.groupby(['mes', 'periodo'])[value_col].mean().unstack()
        monthly_evolution.plot(kind='line', ax=ax6, marker='o', linewidth=2)
        ax6.set_title('Evolución Mensual Promedio')
        ax6.set_xlabel('Mes')
        ax6.set_ylabel('Demanda Promedio')
        ax6.legend(title='Período')
        ax6.set_xticks(range(1, 13))
        ax6.set_xticklabels(['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'])
        ax6.grid(True, alpha=0.3)
        
        # FILA 3: ANÁLISIS ESTADÍSTICOS
        # 3.1 Variación porcentual
        ax7 = axes[2, 0]
        plot_percentage_variation(data_copy, date_col=date_col, value_col=value_col,
                                title="Variación Porcentual", ax=ax7)
        # Añadir línea del competidor
        ax7.axvline(x=fecha_competidor, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax7.text(fecha_competidor, ax7.get_ylim()[1]*0.9, 'Entrada\nCompetidor', 
                ha='left', va='top', fontsize=8, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 3.2 Distribución por período (boxplot)
        ax8 = axes[2, 1]
        colors = ['skyblue', 'lightcoral']  # Definir colores para el boxplot
        sns.boxplot(data=data_copy, x='periodo', y=value_col, ax=ax8, palette=colors)
        ax8.set_title('Distribución de Demanda por Período')
        ax8.set_xlabel('Período')
        ax8.set_ylabel('Demanda')
        
        # 3.3 Análisis de volatilidad
        ax9 = axes[2, 2]
        # Calcular volatilidad como desviación estándar móvil
        data_volatility = data_copy.copy()
        data_volatility = data_volatility.sort_values(date_col)
        
        # Calcular volatilidad usando rolling window de 30 días
        data_volatility['volatilidad'] = data_volatility[value_col].rolling(window=30, min_periods=1).std()
        
        # Graficar volatilidad
        ax9.plot(data_volatility[date_col], data_volatility['volatilidad'], 
                linewidth=2, color='purple', alpha=0.7, label='Volatilidad (30d)')
        
        # Añadir línea del competidor
        ax9.axvline(x=fecha_competidor, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax9.text(fecha_competidor, ax9.get_ylim()[1]*0.9, 'Entrada\nCompetidor', 
                ha='left', va='top', fontsize=8, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Calcular volatilidad promedio por período para comparación
        pre_volatility = data_volatility[data_volatility['periodo'] == 'Pre-Competidor']['volatilidad'].mean()
        post_volatility = data_volatility[data_volatility['periodo'] == 'Post-Competidor']['volatilidad'].mean()
        
        # Añadir líneas horizontales de volatilidad promedio
        ax9.axhline(y=pre_volatility, color='skyblue', linestyle=':', alpha=0.7, label=f'Pre: {pre_volatility:.0f}')
        ax9.axhline(y=post_volatility, color='lightcoral', linestyle=':', alpha=0.7, label=f'Post: {post_volatility:.0f}')
        
        ax9.set_title('Volatilidad de la Demanda')
        ax9.set_xlabel('Fecha')
        ax9.set_ylabel('Volatilidad (Desv. Std)')
        ax9.legend(loc='upper right')
        ax9.grid(True, alpha=0.3)
        ax9.tick_params(axis='x', rotation=45)
        
    else:
        # Análisis por grupos
        groups = data_copy[group_col].unique()
        n_groups = len(groups)
        
        # Configurar subplots dinámicamente
        if n_groups <= 4:
            n_rows, n_cols = 2, 2
        elif n_groups <= 6:
            n_rows, n_cols = 2, 3
        elif n_groups <= 9:
            n_rows, n_cols = 3, 3
        else:
            n_rows, n_cols = 4, 3
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        # Ajustar espaciado (sin título principal)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Calcular impacto para cada grupo y ordenar
        group_impacts = []
        for group in groups:
            group_data = data_copy[data_copy[group_col] == group]
            pre_mean = group_data[group_data['periodo'] == 'Pre-Competidor'][value_col].mean()
            post_mean = group_data[group_data['periodo'] == 'Post-Competidor'][value_col].mean()
            
            if not np.isnan(pre_mean) and not np.isnan(post_mean):
                impact_pct = ((post_mean - pre_mean) / pre_mean) * 100
            else:
                impact_pct = 0
                
            group_impacts.append((group, impact_pct))
        
        # Ordenar por impacto (de mayor impacto negativo a menor)
        group_impacts.sort(key=lambda x: x[1])
        ordered_groups = [group for group, _ in group_impacts]
        
        for i, group in enumerate(ordered_groups):
            if i >= len(axes):
                break
                
            ax = axes[i]
            group_data = data_copy[data_copy[group_col] == group]
            
            # Gráfico de serie de tiempo para este grupo
            plot_time_series_with_competitor(group_data, date_col=date_col, value_col=value_col,
                                           competitor_date=competitor_date,
                                           title=f'{group_col}: {group}', ax=ax)
            
            # Calcular estadísticas de impacto
            pre_mean = group_data[group_data['periodo'] == 'Pre-Competidor'][value_col].mean()
            post_mean = group_data[group_data['periodo'] == 'Post-Competidor'][value_col].mean()
            
            if not np.isnan(pre_mean) and not np.isnan(post_mean):
                impact_pct = ((post_mean - pre_mean) / pre_mean) * 100
                impact_color = 'green' if impact_pct >= 0 else 'red'
                ax.text(0.02, 0.98, f'Impacto: {impact_pct:.1f}%', 
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top', color=impact_color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Ocultar ejes vacíos
        for i in range(len(ordered_groups), len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes

