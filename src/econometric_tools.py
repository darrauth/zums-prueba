# Prueba de Estacionariedad - Test de Dickey-Fuller Aumentado (ADF)
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from scipy.stats import jarque_bera, shapiro
import pandas as pd


# Calcular métricas escala-libres adicionales
def calculate_mase(y_true, y_pred, y_train):
    """Calcular Mean Absolute Scaled Error (MASE)"""
    # Error absoluto medio del modelo
    mae_model = np.mean(np.abs(y_true - y_pred))
    # Error absoluto medio del método naive estacional (diferencias de un período)
    naive_errors = np.abs(np.diff(y_train))
    mae_naive = np.mean(naive_errors)
    return mae_model / mae_naive

def calculate_smape(y_true, y_pred):
    """Calcular symmetric Mean Absolute Percentage Error (sMAPE)"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))





# Test ADF en la serie original
def adf_test(series, title=""):
    result = adfuller(series, autolag='AIC')
    
    print(f'\n{title}')
    print('-' * len(title))
    print(f'Estadístico ADF: {result[0]:.6f}')
    print(f'p-valor: {result[1]:.6f}')
    print(f'Valores críticos:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print("✓ RESULTADO: Serie ESTACIONARIA (rechazamos H0)")
    else:
        print("✗ RESULTADO: Serie NO ESTACIONARIA (no rechazamos H0)")
    
    return result[1] <= 0.05


def plot_residual_diagnostics(ts, fitted_values, residuals, figsize=(18, 12)):
    """
    Crear gráficos de diagnóstico de residuos (función reutilizable para diferentes familias de modelos).
    
    Parameters:
    -----------
    ts : pandas Series
        Serie de tiempo original
    fitted_values : array-like
        Valores ajustados del modelo
    residuals : array-like
        Residuos del modelo
    figsize : tuple, optional
        Tamaño de la figura (default: (18, 12))
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Gráfico 1: Serie original vs valores ajustados
    axes[0,0].plot(ts.index, ts.values, label='Serie Original', alpha=0.7)
    axes[0,0].plot(ts.index, fitted_values, label='Valores Ajustados', alpha=0.8)
    axes[0,0].set_title('Serie Original vs Valores Ajustados')
    axes[0,0].set_ylabel('Valor')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Gráfico 2: Residuos en el tiempo
    axes[0,1].plot(ts.index, residuals, color='red', alpha=0.7)
    axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.8)
    axes[0,1].set_title('Residuos vs Tiempo')
    axes[0,1].set_ylabel('Residuos')
    axes[0,1].set_xlabel('Tiempo')
    axes[0,1].grid(True, alpha=0.3)
    
    # Gráfico 3: Residuos vs valores ajustados
    axes[0,2].scatter(fitted_values, residuals, alpha=0.6, color='purple')
    axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.8)
    axes[0,2].set_title('Residuos vs Valores Ajustados')
    axes[0,2].set_xlabel('Valores Ajustados')
    axes[0,2].set_ylabel('Residuos')
    axes[0,2].grid(True, alpha=0.3)
    
    # Añadir línea de tendencia para detectar patrones
    try:
        z = np.polyfit(fitted_values, residuals, 1)
        p = np.poly1d(z)
        axes[0,2].plot(fitted_values, p(fitted_values), "r--", alpha=0.8, 
                      label=f'Tendencia (pendiente: {z[0]:.4f})')
        axes[0,2].legend()
    except:
        pass
    
    # Gráfico 4: Distribución de residuos
    axes[1,0].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black', density=True)
    axes[1,0].axvline(residuals.mean(), color='red', linestyle='--', 
                     label=f'Media: {residuals.mean():.3f}')
    # Superponer distribución normal teórica
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1,0].plot(x, stats.norm.pdf(x, mu, sigma), 'k-', alpha=0.8, 
                   label=f'Normal teórica (μ={mu:.3f}, σ={sigma:.3f})')
    axes[1,0].set_title('Distribución de Residuos')
    axes[1,0].set_xlabel('Residuos')
    axes[1,0].set_ylabel('Densidad')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Gráfico 5: Q-Q Plot de residuos
    stats.probplot(residuals, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot de Residuos')
    axes[1,1].grid(True, alpha=0.3)
    
    # Gráfico 6: Autocorrelación de residuos (ACF)
    from statsmodels.graphics.tsaplots import plot_acf
    try:
        plot_acf(residuals, lags=min(20, len(residuals)//4), ax=axes[1,2], alpha=0.05)
        axes[1,2].set_title('ACF de Residuos')
        axes[1,2].grid(True, alpha=0.3)
    except:
        # Fallback si plot_acf falla
        axes[1,2].text(0.5, 0.5, 'ACF no disponible', ha='center', va='center', 
                      transform=axes[1,2].transAxes)
        axes[1,2].set_title('ACF de Residuos')
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes


def normality_tests_residuals(residuals):
    """
    Realizar pruebas de normalidad sobre residuos (función reutilizable para diferentes familias de modelos).
    
    Parameters:
    -----------
    residuals : array-like
        Residuos del modelo
    
    Returns:
    --------
    dict : Dictionary con resultados de las pruebas de normalidad
    """
    
    print("\nPRUEBAS DE NORMALIDAD DE RESIDUOS")
    print("=" * 50)
    
    # Test de normalidad Jarque-Bera
    jb_stat, jb_pvalue = jarque_bera(residuals)
    print(f"Test Jarque-Bera (Normalidad):")
    print(f"  Estadístico: {jb_stat:.4f}")
    print(f"  p-valor: {jb_pvalue:.4f}")
    print(f"  Interpretación: {'Los residuos son normales' if jb_pvalue > 0.05 else 'Los residuos NO son normales'}")
    
    # Test de normalidad Shapiro-Wilk (mejor para muestras pequeñas)
    sw_stat, sw_pvalue = None, None
    if len(residuals) <= 5000:  # Shapiro-Wilk tiene límite de muestra
        sw_stat, sw_pvalue = shapiro(residuals)
        print(f"\nTest Shapiro-Wilk (Normalidad):")
        print(f"  Estadístico: {sw_stat:.4f}")
        print(f"  p-valor: {sw_pvalue:.4f}")
        print(f"  Interpretación: {'Los residuos son normales' if sw_pvalue > 0.05 else 'Los residuos NO son normales'}")
    
    # Retornar resultados
    results = {
        'jarque_bera': {'statistic': jb_stat, 'pvalue': jb_pvalue},
        'shapiro_wilk': {'statistic': sw_stat, 'pvalue': sw_pvalue}
    }
    
    return results


def autocorrelation_tests_residuals(residuals):
    """
    Realizar pruebas de autocorrelación sobre residuos (función reutilizable para diferentes familias de modelos).
    
    Parameters:
    -----------
    residuals : array-like
        Residuos del modelo
    
    Returns:
    --------
    dict : Dictionary con resultados de las pruebas de autocorrelación
    """
    
    print("\nPRUEBAS DE AUTOCORRELACIÓN DE RESIDUOS")
    print("=" * 50)
    
    # Test Ljung-Box para autocorrelación en residuos
    lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
    print(f"Test Ljung-Box (Autocorrelación en residuos):")
    print(lb_result)
    
    # Interpretación
    significant_lags = (lb_result['lb_pvalue'] < 0.05).sum()
    print(f"\nInterpretación: {'Hay autocorrelación significativa' if significant_lags > 0 else 'No hay autocorrelación significativa'} en los residuos")
    print(f"Lags con autocorrelación significativa: {significant_lags}/10")
    
    results = {
        'ljung_box': lb_result,
        'significant_lags': significant_lags
    }
    
    return results


def heteroscedasticity_tests_residuals(residuals, fitted_values):
    """
    Realizar pruebas de heterocedasticidad sobre residuos (función reutilizable para diferentes familias de modelos).
    
    Parameters:
    -----------
    residuals : array-like
        Residuos del modelo
    fitted_values : array-like
        Valores ajustados del modelo
    
    Returns:
    --------
    dict : Dictionary con resultados de las pruebas de heterocedasticidad
    """
    
    print("\nPRUEBAS DE HETEROCEDASTICIDAD DE RESIDUOS")
    print("=" * 50)
    
    # Preparar datos para las pruebas
    residuals_squared = residuals ** 2
    
    # Convertir a numpy array y reshape si es necesario
    if hasattr(fitted_values, 'values'):
        fitted_values_array = fitted_values.values
    else:
        fitted_values_array = np.array(fitted_values)
    
    # Agregar constante para las pruebas de heterocedasticidad
    import statsmodels.api as sm
    fitted_values_with_const = sm.add_constant(fitted_values_array)
    
    try:
        # Test de Breusch-Pagan
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, fitted_values_with_const)
        print(f"Test Breusch-Pagan (Heterocedasticidad):")
        print(f"  Estadístico: {bp_stat:.4f}")
        print(f"  p-valor: {bp_pvalue:.4f}")
        print(f"  Interpretación: {'Hay heterocedasticidad' if bp_pvalue < 0.05 else 'Homocedasticidad (varianza constante)'}")
        
        bp_results = {'statistic': bp_stat, 'pvalue': bp_pvalue}
    except Exception as e:
        print(f"Test Breusch-Pagan falló: {e}")
        bp_results = {'statistic': None, 'pvalue': None}
    
    try:
        # Test de White
        white_stat, white_pvalue, _, _ = het_white(residuals, fitted_values_with_const)
        print(f"\nTest White (Heterocedasticidad):")
        print(f"  Estadístico: {white_stat:.4f}")
        print(f"  p-valor: {white_pvalue:.4f}")
        print(f"  Interpretación: {'Hay heterocedasticidad' if white_pvalue < 0.05 else 'Homocedasticidad (varianza constante)'}")
        
        white_results = {'statistic': white_stat, 'pvalue': white_pvalue}
    except Exception as e:
        print(f"Test White falló: {e}")
        white_results = {'statistic': None, 'pvalue': None}
    
    results = {
        'breusch_pagan': bp_results,
        'white': white_results
    }
    
    return results


def analyze_residuals_pmdarima(model, ts, figsize=(18, 12)):
    """
    Análisis completo de residuos para modelos pmdarima (AutoARIMA).
    
    Parameters:
    -----------
    model : pmdarima fitted model object
        Modelo pmdarima ajustado con métodos fittedvalues() y resid()
    ts : pandas Series
        Serie de tiempo original
    figsize : tuple, optional
        Tamaño de la figura para los gráficos (default: (15, 12))
    
    Returns:
    --------
    dict : Dictionary con métricas de error y resultados de tests
    """
    
    # Obtener valores ajustados y residuos (específicos de pmdarima)
    fitted_values = model.fittedvalues()
    residuals = model.resid()
    
    # Calcular métricas de error
    mae = mean_absolute_error(ts, fitted_values)
    rmse = np.sqrt(mean_squared_error(ts, fitted_values))
    mape = np.mean(np.abs(residuals / ts)) * 100
    
    # Imprimir métricas
    print("MÉTRICAS DE ERROR DEL MODELO")
    print("=" * 40)
    print(f"MAE (Error Absoluto Medio): {mae:.2f}")
    print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}")
    print(f"MAPE (Error Porcentual Absoluto Medio): {mape:.2f}%")
    
    print("\nESTADÍSTICAS DE RESIDUOS")
    print("=" * 40)
    print(f"Media de residuos: {residuals.mean():.4f}")
    print(f"Desviación estándar de residuos: {residuals.std():.4f}")
    print(f"Mínimo residuo: {residuals.min():.2f}")
    print(f"Máximo residuo: {residuals.max():.2f}")
    
    # Usar función de gráficos reutilizable
    plot_residual_diagnostics(ts, fitted_values, residuals, figsize)
    
    # Usar funciones de pruebas estadísticas reutilizables
    normality_results = normality_tests_residuals(residuals)
    autocorr_results = autocorrelation_tests_residuals(residuals)
    hetero_results = heteroscedasticity_tests_residuals(residuals, fitted_values)
    
    # Interpretación general
    print(f"\nINTERPRETACIÓN GENERAL:")
    print(f"- MAE de {mae:.2f} indica error promedio absoluto")
    print(f"- RMSE de {rmse:.2f} penaliza más los errores grandes")
    print(f"- MAPE de {mape:.2f}% indica error porcentual promedio")
  
    # Retornar resultados en un diccionario
    results = {
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        },
        'residual_stats': {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max()
        },
        'normality_tests': normality_results,
        'autocorrelation_tests': autocorr_results,
        'heteroscedasticity_tests': hetero_results
    }
    
    return results
    



def detect_outliers_and_create_dummies(model, ts_index, threshold=3.5):
    """
    Detecta outliers en residuos SARIMAX y crea variables dummy correspondientes.
    
    Parámetros:
    -----------
    model : statsmodels fitted model
        Modelo SARIMAX ajustado con atributo .resid
    ts_index : pandas DatetimeIndex
        Índice de fechas de la serie temporal
    threshold : float, default 3.5
        Umbral para detección de outliers (residuos studentizados)
        
    Returns:
    --------
    dict : diccionario con resultados
        - 'outlier_dates': fechas de los outliers detectados
        - 'outlier_indices': índices de los outliers
        - 'dummy_matrix': DataFrame con dummies para cada outlier
        - 'residuals_std': residuos studentizados
        - 'outlier_info': información detallada de cada outlier
    """
    import pandas as pd
    import numpy as np
    
    print("DETECCIÓN DE OUTLIERS EN RESIDUOS")
    print("=" * 50)
    
    # Extraer residuos del modelo
    resid = model.resid
    sigma = resid.std()
    
    # Calcular residuos studentizados
    resid_std = resid / sigma
    
    print(f"Desviación estándar de residuos: {sigma:.4f}")
    print(f"Umbral para outliers: ±{threshold}")
    
    # Detectar outliers donde |resid_std| > threshold
    outlier_mask = np.abs(resid_std) > threshold
    outlier_indices = np.where(outlier_mask)[0]
    
    if len(outlier_indices) == 0:
        print("✓ No se detectaron outliers con el umbral especificado")
        return {
            'outlier_dates': [],
            'outlier_indices': [],
            'dummy_matrix': pd.DataFrame(),
            'residuals_std': resid_std,
            'outlier_info': []
        }
    
    # Obtener fechas de outliers
    outlier_dates = ts_index[outlier_indices]
    outlier_values = resid_std.iloc[outlier_indices]
    
    print(f"\n✗ Detectados {len(outlier_indices)} outliers:")
    print("-" * 40)
    
    # Crear matriz de dummies
    dummy_matrix = pd.DataFrame(index=ts_index)
    outlier_info = []
    
    for i, (idx, date, value) in enumerate(zip(outlier_indices, outlier_dates, outlier_values)):
        dummy_name = f"outlier_{date.strftime('%Y%m%d')}"
        
        # Crear dummy (1 en la fecha del outlier, 0 en el resto)
        dummy_matrix[dummy_name] = 0
        dummy_matrix.loc[date, dummy_name] = 1
        
        # Información del outlier
        outlier_info.append({
            'date': date,
            'index': idx,
            'residual_std': value,
            'dummy_name': dummy_name,
            'type': 'Positivo' if value > 0 else 'Negativo'
        })
        
        print(f"{i+1}. Fecha: {date.strftime('%Y-%m-%d')}, "
              f"Residual std: {value:.3f}, "
              f"Tipo: {'Positivo' if value > 0 else 'Negativo'}, "
              f"Dummy: {dummy_name}")
    
    print(f"\nRESUMEN:")
    print(f"- Total outliers detectados: {len(outlier_indices)}")
    print(f"- Outliers positivos: {sum(1 for info in outlier_info if info['type'] == 'Positivo')}")
    print(f"- Outliers negativos: {sum(1 for info in outlier_info if info['type'] == 'Negativo')}")
    print(f"- Variables dummy creadas: {len(dummy_matrix.columns)}")
    
    # Estadísticas adicionales
    print(f"\nESTADÍSTICAS DE OUTLIERS:")
    print(f"- Residual std máximo: {np.abs(outlier_values).max():.3f}")
    print(f"- Residual std promedio: {np.abs(outlier_values).mean():.3f}")
    print(f"- Porcentaje de outliers: {(len(outlier_indices)/len(resid))*100:.2f}%")
    
    return {
        'outlier_dates': outlier_dates,
        'outlier_indices': outlier_indices,
        'dummy_matrix': dummy_matrix,
        'residuals_std': resid_std,
        'outlier_info': outlier_info
    }




def analyze_residuals_statsmodels(model, ts, figsize=(18, 12), sanity_check=True):
    """
    Análisis completo de residuos para modelos statsmodels (ARIMA/SARIMAX),
    descartando automáticamente las primeras 'd' observaciones (no estacional)
    y calculando residuos como (y - yhat) usando one-step-ahead in-sample
    via get_prediction(dynamic=False) para evitar artefactos tempranos.

    Parameters
    ----------
    model : statsmodels fitted model object
        Resultado ajustado (tiene .get_prediction, .fittedvalues y .model.order).
    ts : pandas.Series
        Serie original en niveles con el mismo índice usado para ajustar.
    figsize : tuple
        Tamaño de los gráficos diagnósticos.
    sanity_check : bool
        Si True, imprime chequeos rápidos (primeras filas y el residuo mínimo).

    Returns
    -------
    dict
        Métricas, estadísticas de residuos y resultados de pruebas.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # 1) Extraer d (solo NO estacional)
    try:
        _, d, _ = model.model.order
    except Exception:
        d = 0
    burnin = int(d)

    # Asegurar orden temporal consistente
    ts = ts.sort_index()

    # 2) Pronóstico in-sample one-step-ahead desde el primer índice válido tras burn-in
    #    Esto evita desalineaciones y problemas de inicialización.
    try:
        start_label = ts.index[burnin]
        end_label = ts.index[-1]
        pred_res = model.get_prediction(start=start_label, end=end_label, dynamic=False)
        yhat = pred_res.predicted_mean.sort_index()
    except Exception:
        # Fallback: usar fittedvalues si algo falla
        yhat = model.fittedvalues.sort_index()
        # Recortar para mantener coherencia con burn-in
        if burnin > 0:
            yhat = yhat.reindex(ts.index[burnin:])

    # 3) Construir y alinear y yhat por índice; calcular residuo explícito
    y = ts.reindex(yhat.index)
    aligned = pd.concat({'y': y, 'yhat': yhat}, axis=1).dropna()
    aligned['resid'] = aligned['y'] - aligned['yhat']

    if aligned.empty:
        print("Advertencia: no hay datos válidos tras el recorte de d y la alineación.")
        return {
            'burnin_dropped': burnin,
            'metrics': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'smape': np.nan},
            'residual_stats': {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan},
            'normality_tests': None,
            'autocorrelation_tests': None,
            'heteroscedasticity_tests': None
        }

    # 4) (Opcional) sanity checks para detectar el caso que describes
    if sanity_check:
        print("\nChequeo rápido (primeras 8 obs. tras burn-in y alineación):")
        print(aligned.head(8))
        idx_min = aligned['resid'].idxmin()
        print("\nObservación con residuo mínimo (posible outlier temprano):")
        print(aligned.loc[[idx_min]])

    # 5) Métricas
    y_np = aligned['y'].to_numpy()
    yhat_np = aligned['yhat'].to_numpy()
    resid_np = aligned['resid'].to_numpy()

    mae = mean_absolute_error(y_np, yhat_np)
    rmse = np.sqrt(mean_squared_error(y_np, yhat_np))

    # MAPE con protección ante ceros
    denom = np.where(y_np == 0, np.nan, y_np)
    mape = np.nanmean(np.abs((y_np - yhat_np) / denom)) * 100

    # sMAPE (más estable con ceros)
    smape = 100 * np.nanmean(2 * np.abs(y_np - yhat_np) / (np.abs(y_np) + np.abs(yhat_np)))

    print("\nMÉTRICAS DE ERROR DEL MODELO (tras descartar las primeras d observaciones)")
    print("=" * 80)
    print(f"d descartado (no estacional): {burnin}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"sMAPE: {smape:.2f}%")

    print("\nESTADÍSTICAS DE RESIDUOS")
    print("=" * 80)
    print(f"Media de residuos            : {np.nanmean(resid_np):.6f}")
    print(f"Desviación estándar residuos : {np.nanstd(resid_np, ddof=1):.6f}")
    print(f"Mínimo residuo               : {np.nanmin(resid_np):.4f}")
    print(f"Máximo residuo               : {np.nanmax(resid_np):.4f}")

    # 6) Gráficos con TUS funciones (sin modificarlas)
    plot_residual_diagnostics(
        ts=aligned['y'],
        fitted_values=aligned['yhat'],
        residuals=aligned['resid'],
        figsize=figsize
    )

    # 7) Tests con TUS funciones (sin modificarlas)
    normality_results = normality_tests_residuals(aligned['resid'].values)
    autocorr_results = autocorrelation_tests_residuals(aligned['resid'].values)
    hetero_results = heteroscedasticity_tests_residuals(
        residuals=aligned['resid'].values,
        fitted_values=aligned['yhat'].values
    )

    print("\nINTERPRETACIÓN GENERAL:")
    print("- Usamos one-step-ahead in-sample desde el primer punto tras burn-in,")
    print("  por eso ya no deberías ver picos artificiales en los primeros periodos.")
    print("- MAE: error absoluto promedio. RMSE: penaliza más los grandes.")
    print("- Si hay muchos ceros en y, prioriza sMAPE sobre MAPE.")

    return {
        'burnin_dropped': burnin,
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'smape': float(smape)
        },
        'residual_stats': {
            'mean': float(np.nanmean(resid_np)),
            'std': float(np.nanstd(resid_np, ddof=1)),
            'min': float(np.nanmin(resid_np)),
            'max': float(np.nanmax(resid_np))
        },
        'normality_tests': normality_results,
        'autocorrelation_tests': autocorr_results,
        'heteroscedasticity_tests': hetero_results
    }