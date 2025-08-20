import pandas as pd
import os
from sklearn.model_selection import train_test_split

def decompose_date(df):
    df["date"] = pd.to_datetime(df["date"])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.weekday  # Monday=0, Sunday=6
    df['weekend'] = df['day_of_week'] >= 5  # Boolean, True for weekend
    df['quarter'] = df['date'].dt.quarter
    return df

def convert_data_types_for_model(df, is_global=False):
    """
    Convierte las columnas del DataFrame a los tipos de datos correctos para el modelo.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame a procesar (df_all o df_global)
    is_global : bool, default=False
        Si True, procesa df_global; si False, procesa df_all
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con tipos de datos corregidos
    """
    
    df_processed = df.copy()
    
    # Variables de fecha - convertir a datetime
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
    
    # Variables numéricas continuas - convertir a float64
    numeric_continuous_vars = ['demanda']
    for var in numeric_continuous_vars:
        if var in df_processed.columns:
            df_processed[var] = pd.to_numeric(df_processed[var], errors='coerce').astype('float64')
    
    # Variables categóricas nominales - convertir a category
    categorical_nominal_vars = ['categoria', 'subcategoria', 'tamaño']
    for var in categorical_nominal_vars:
        if var in df_processed.columns:
            df_processed[var] = df_processed[var].astype('category')
    
    # Variables binarias/dummy - convertir a int8 (0 o 1)
    binary_vars = ['premium', 'marca_exclusiva', 'estacional', 'Entrada_competidor']
    for var in binary_vars:
        if var in df_processed.columns:
            df_processed[var] = df_processed[var].astype('int')
    
    # Variables de identificación - convertir a category o string según corresponda
    id_vars = ['id_producto']
    for var in id_vars:
        if var in df_processed.columns:
            df_processed[var] = df_processed[var].astype('int')
    
    return df_processed


def apply_one_hot_encoding(df, drop_first=True, exclude_cols=None):
    """
    Aplica one-hot encoding a todas las variables categóricas del DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con variables categóricas
    drop_first : bool, default=True
        Si True, elimina la primera categoría de cada variable para evitar multicolinealidad
    exclude_cols : list, default=None
        Lista de columnas categóricas a excluir del one-hot encoding
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con variables categóricas convertidas a one-hot encoding
    dict
        Diccionario con información sobre las columnas creadas
    """
    
    df_encoded = df.copy()
    encoding_info = {}
    
    # Identificar columnas categóricas
    categorical_cols = df_encoded.select_dtypes(include=['category']).columns.tolist()
    
    # Excluir columnas si se especifica
    if exclude_cols:
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    print(f"Variables categóricas encontradas para encoding: {categorical_cols}")
    
    if not categorical_cols:
        print("No se encontraron variables categóricas para procesar.")
        return df_encoded, encoding_info
    
    # Aplicar one-hot encoding
    for col in categorical_cols:
        print(f"\nProcesando columna: {col}")
        
        # Obtener categorías únicas
        categories = df_encoded[col].cat.categories.tolist()
        print(f"  Categorías: {categories}")
        
        # Crear variables dummy
        dummies = pd.get_dummies(
            df_encoded[col], 
            prefix=col, 
            drop_first=drop_first,
            dtype='int8'  # Usar int8 para ahorrar memoria
        )
        
        # Guardar información sobre el encoding
        encoding_info[col] = {
            'original_categories': categories,
            'dummy_columns': dummies.columns.tolist(),
            'dropped_category': categories[0] if drop_first else None
        }
        
        print(f"  Columnas creadas: {dummies.columns.tolist()}")
        if drop_first:
            print(f"  Categoría de referencia (eliminada): {categories[0]}")
        
        # Concatenar las nuevas columnas al DataFrame
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Eliminar la columna original
        df_encoded = df_encoded.drop(columns=[col])
    
    print(f"\n{'='*60}")
    print(f"RESUMEN DEL ONE-HOT ENCODING")
    print(f"{'='*60}")
    print(f"Variables categóricas procesadas: {len(categorical_cols)}")
    print(f"Forma original: {df.shape}")
    print(f"Forma después del encoding: {df_encoded.shape}")
    print(f"Nuevas columnas agregadas: {df_encoded.shape[1] - df.shape[1]}")
    
    # Mostrar información detallada

    total_dummies = 0
    for col, info in encoding_info.items():
        total_dummies += len(info['dummy_columns'])
        print(f"  {col}: {len(info['dummy_columns'])} variables dummy")
    
    print(f"\nTotal de variables dummy creadas: {total_dummies}")
    
    return df_encoded, encoding_info


def time_series_split(df, target_col, test_size= 0.2):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test


def separar_productos(df_local, target_col='demanda', test_size=0.2, save_files=True, base_path=None):
    """
    Separa los datos por producto en conjuntos de entrenamiento y prueba usando división temporal.
    Opcionalmente guarda los archivos en carpetas organizadas.
    
    Parameters:
    -----------
    df_local : pandas.DataFrame
        DataFrame con datos de múltiples productos que contiene 'id_producto' y columna objetivo
    target_col : str, default='demanda'
        Nombre de la columna objetivo
    test_size : float, default=0.2
        Proporción de datos para el conjunto de prueba
    save_files : bool, default=True
        Si True, guarda los archivos en carpetas train/test
    base_path : str, default=None
        Ruta base donde crear las carpetas. Si None, usa 'data_by_product'
        
    Returns:
    --------
    dict
        Diccionario donde cada key es id_producto y el value es otro diccionario con 
        'train' y 'test', cada uno conteniendo un DataFrame completo con todas las columnas
    """
    
    # Verificar que las columnas necesarias existen
    if 'id_producto' not in df_local.columns:
        raise ValueError("El DataFrame debe contener la columna 'id_producto'")
    
    if target_col not in df_local.columns:
        raise ValueError(f"El DataFrame debe contener la columna objetivo '{target_col}'")
    
    # Configurar rutas para guardar archivos
    if save_files:
        if base_path is None:
            base_path = "data_by_product"
        
        train_path = os.path.join(base_path, "train")
        test_path = os.path.join(base_path, "test")
        
        # Crear carpetas si no existen
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        
        print(f"Archivos se guardarán en:")
        print(f"  Train: {train_path}")
        print(f"  Test: {test_path}")
    
    # Asegurar que el DataFrame esté ordenado por fecha
    if 'date' in df_local.columns:
        df_local = df_local.sort_values(['id_producto', 'date']).reset_index(drop=True)
    
    productos_data = {}
    productos_unicos = df_local['id_producto'].unique()
    
    print(f"\nSeparando datos para {len(productos_unicos)} productos...")
    print(f"Columna objetivo: {target_col}")
    print(f"Tamaño de prueba: {test_size}")
    print("-" * 50)
    
    productos_procesados = 0
    productos_con_error = []
    archivos_guardados = []
    
    for producto_id in productos_unicos:
        try:
            # Filtrar datos del producto específico
            df_producto = df_local[df_local['id_producto'] == producto_id].copy()
            
            # Verificar que hay suficientes datos
            if len(df_producto) < 2:
                print(f"Producto {producto_id}: Insuficientes datos ({len(df_producto)} observaciones). Omitiendo...")
                productos_con_error.append(producto_id)
                continue
            
            # Aplicar división temporal usando la función existente
            X_train, X_test, y_train, y_test = time_series_split(
                df_producto, 
                target_col=target_col, 
                test_size=test_size
            )
            
            # Combinar X y y para guardar datasets completos
            train_set = X_train.copy()
            train_set[target_col] = y_train
            
            test_set = X_test.copy()
            test_set[target_col] = y_test
            
            # Guardar en el diccionario - cambiar estructura para que sea más simple
            productos_data[producto_id] = {
                'train': train_set,  # DataFrame completo
                'test': test_set     # DataFrame completo
            }
            
            # Guardar archivos si está habilitado
            if save_files:
                try:
                    # Nombres de archivos
                    train_filename = f"product_id_{producto_id}_train_set.csv"
                    test_filename = f"product_id_{producto_id}_test_set.csv"
                    
                    train_filepath = os.path.join(train_path, train_filename)
                    test_filepath = os.path.join(test_path, test_filename)
                    
                    # Guardar archivos
                    train_set.to_csv(train_filepath, index=False)
                    test_set.to_csv(test_filepath, index=False)
                    
                    archivos_guardados.extend([train_filepath, test_filepath])
                    
                except Exception as e:
                    print(f"Error guardando archivos para producto {producto_id}: {str(e)}")
            
            productos_procesados += 1
            
            # Mostrar progreso cada 20 productos
            if productos_procesados % 20 == 0 or producto_id in list(productos_unicos)[:5]:
                status = f"Train={len(X_train)}, Test={len(X_test)}"
                if save_files:
                    status += " (archivos guardados)"
                print(f"Producto {producto_id}: {status}")
                
        except Exception as e:
            print(f"Error procesando producto {producto_id}: {str(e)}")
            productos_con_error.append(producto_id)
            continue
    
    print("-" * 50)
    print(f"Resumen del procesamiento:")
    print(f"  Total productos: {len(productos_unicos)}")
    print(f"  Productos procesados exitosamente: {productos_procesados}")
    print(f"  Productos con errores: {len(productos_con_error)}")
    
    if save_files:
        print(f"  Archivos guardados: {len(archivos_guardados)}")
        print(f"  Ubicación: {base_path}")
        
        # Mostrar algunos ejemplos de archivos creados
        if archivos_guardados:
            print(f"\nEjemplos de archivos creados:")
            for archivo in archivos_guardados[:6]:  # Mostrar primeros 6
                print(f"  {archivo}")
            if len(archivos_guardados) > 6:
                print(f"  ... y {len(archivos_guardados) - 6} más")
    
    if productos_con_error:
        print(f"\n  IDs con errores: {productos_con_error}")
    
    return productos_data


def load_and_preprocess_data(df_global_path="Data/df_global.csv", 
                           df_all_path="Data/data_clean.csv",
                           drop_columns=None,
                           apply_encoding=True,
                           verbose=True):
    """
    Función integral para cargar y preprocesar los datos de manera completa.
    
    Parameters:
    -----------
    df_global_path : str, default="Data/df_global.csv"
        Ruta al archivo df_global.csv
    df_all_path : str, default="Data/data_clean.csv"
        Ruta al archivo data_clean.csv
    drop_columns : list, default=None
        Lista adicional de columnas a eliminar. Por defecto elimina ["Unnamed: 0", "subcategoria"]
    apply_encoding : bool, default=True
        Si True, aplica one-hot encoding a las variables categóricas
    verbose : bool, default=True
        Si True, muestra información del procesamiento
        
    Returns:
    --------
    tuple
        (df_all_processed, df_global_processed, encoding_info_all, encoding_info_global)
    """
    
    if verbose:
        print("="*60)
        print("CARGA Y PREPROCESAMIENTO INTEGRAL DE DATOS")
        print("="*60)
    
    # Configurar pandas para mostrar todas las columnas
    pd.set_option('display.max_columns', None)
    
    # 1. Cargar los datos
    if verbose:
        print("1. Cargando archivos...")
        
    df1 = pd.read_csv(df_global_path)
    df2 = pd.read_csv(df_all_path)
    
    df_global = df1.copy()
    df_all = df2.copy()
    
    if verbose:
        print(f"   - df_global cargado: {df_global.shape}")
        print(f"   - df_all cargado: {df_all.shape}")
    
    # 2. Eliminar columnas no deseadas
    if verbose:
        print("\n2. Eliminando columnas innecesarias...")
        
    # Columnas por defecto a eliminar
    default_drop_columns = ["Unnamed: 0", "subcategoria"]
    
    # Combinar con columnas adicionales si se proporcionan
    if drop_columns is None:
        columns_to_drop = default_drop_columns
    else:
        columns_to_drop = default_drop_columns + drop_columns
    
    # Eliminar columnas de df_all
    initial_cols_all = df_all.columns.tolist()
    df_all.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    dropped_all = set(initial_cols_all) - set(df_all.columns)
    
    # Eliminar columnas de df_global (sin "Unnamed: 0")
    global_drops = [col for col in columns_to_drop if col != "Unnamed: 0"]
    initial_cols_global = df_global.columns.tolist()
    df_global.drop(columns=global_drops, inplace=True, errors='ignore')
    dropped_global = set(initial_cols_global) - set(df_global.columns)
    
    if verbose and (dropped_all or dropped_global):
        print(f"   - Columnas eliminadas de df_all: {list(dropped_all)}")
        print(f"   - Columnas eliminadas de df_global: {list(dropped_global)}")
    
    # 3. Descomponer fechas
    if verbose:
        print("\n3. Añadiendo variables de descomposición de fecha...")
        
    df_all = decompose_date(df_all)
    df_global = decompose_date(df_global)
    
    if verbose:
        print("   - Variables de fecha añadidas: month, year, day_of_week, weekend, quarter")
    
    # 4. Convertir tipos de datos
    if verbose:
        print("\n4. Convirtiendo tipos de datos...")
        
    df_all = convert_data_types_for_model(df_all, is_global=False)
    df_global = convert_data_types_for_model(df_global, is_global=True)
    
    if verbose:
        print("   - Tipos de datos convertidos correctamente")
        print(f"   - df_all shape después de conversión: {df_all.shape}")
        print(f"   - df_global shape después de conversión: {df_global.shape}")
    
    # 5. Aplicar one-hot encoding
    encoding_info_all = {}
    encoding_info_global = {}
    
    if apply_encoding:
        if verbose:
            print("\n5. Aplicando One-Hot Encoding...")
            print("="*60)
            print("APLICACIÓN DE ONE-HOT ENCODING")
            print("="*60)
        
        # Para df_all - excluir id_producto ya que es identificador, no variable predictora
        df_all, encoding_info_all = apply_one_hot_encoding(
            df_all, 
            drop_first=True, 
            exclude_cols=['id_producto']  # Excluir identificadores
        )
        
        if verbose:
            print("\n" + "="*60)
        
        # Para df_global (si tiene variables categóricas)
        df_global, encoding_info_global = apply_one_hot_encoding(
            df_global, 
            drop_first=True
        )
    else:
        if verbose:
            print("\n5. One-Hot Encoding omitido (apply_encoding=False)")
    
    # 6. Resumen final
    if verbose:
        print("\n" + "="*60)
        print("RESUMEN FINAL DEL PREPROCESAMIENTO")
        print("="*60)
        print(f"df_all:")
        print(f"  - Shape final: {df_all.shape}")
        print(f"  - Columnas: {len(df_all.columns)}")
        print(f"  - Memoria: {df_all.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\ndf_global:")
        print(f"  - Shape final: {df_global.shape}")
        print(f"  - Columnas: {len(df_global.columns)}")
        print(f"  - Memoria: {df_global.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if apply_encoding:
            total_encoded_vars_all = len(encoding_info_all)
            total_encoded_vars_global = len(encoding_info_global)
            print(f"\nVariables categóricas procesadas:")
            print(f"  - df_all: {total_encoded_vars_all} variables")
            print(f"  - df_global: {total_encoded_vars_global} variables")
    
    return df_all, df_global, encoding_info_all, encoding_info_global


def prepare_test_data_with_features(test_data_path="Data/demanda_test.csv", 
                                  reference_data_path="Data/data_clean.csv",
                                  drop_columns=None,
                                  apply_encoding=True,
                                  encoding_info=None,
                                  verbose=True):
    """
    Prepara el conjunto de datos de prueba agregando las características de productos
    desde el dataset de referencia y aplicando el mismo preprocesamiento.
    
    Parameters:
    -----------
    test_data_path : str, default="Data/demanda_test.csv"
        Ruta al archivo de datos de prueba (solo tiene date y id_producto)
    reference_data_path : str, default="Data/data_clean.csv"
        Ruta al archivo de referencia con todas las características de productos
    drop_columns : list, default=None
        Lista adicional de columnas a eliminar
    apply_encoding : bool, default=True
        Si True, aplica one-hot encoding a las variables categóricas
    encoding_info : dict, default=None
        Información de encoding previo para mantener consistencia
    verbose : bool, default=True
        Si True, muestra información del procesamiento
        
    Returns:
    --------
    tuple
        (df_test_processed, encoding_info_test)
    """
    
    if verbose:
        print("="*60)
        print("PREPARACIÓN DE DATOS DE PRUEBA CON CARACTERÍSTICAS")
        print("="*60)
    
    # 1. Cargar los datos
    if verbose:
        print("1. Cargando archivos...")
        
    df_test = pd.read_csv(test_data_path)
    df_reference = pd.read_csv(reference_data_path)
    
    if verbose:
        print(f"   - Datos de prueba cargados: {df_test.shape}")
        print(f"   - Datos de referencia cargados: {df_reference.shape}")
        print(f"   - Columnas en test: {list(df_test.columns)}")
    
    # 2. Obtener características únicas por producto desde los datos de referencia
    if verbose:
        print("\n2. Extrayendo características por producto...")
    
    # Eliminar columnas no deseadas del dataset de referencia
    default_drop_columns = ["Unnamed: 0", "subcategoria", "demanda", "date"]
    if drop_columns is None:
        columns_to_drop = default_drop_columns
    else:
        columns_to_drop = default_drop_columns + drop_columns
    
    # Preparar datos de referencia sin las columnas que no necesitamos
    df_reference_clean = df_reference.drop(columns=columns_to_drop, errors='ignore')
    
    # Obtener características únicas por producto (tomar el primer registro de cada producto)
    # Esto asume que las características del producto son constantes en el tiempo
    product_features = df_reference_clean.groupby('id_producto').first().reset_index()
    
    if verbose:
        print(f"   - Características extraídas para {len(product_features)} productos únicos")
        print(f"   - Columnas de características: {list(product_features.columns)}")
    
    # 3. Hacer merge con los datos de prueba
    if verbose:
        print("\n3. Fusionando datos de prueba con características de productos...")
    
    df_test_with_features = df_test.merge(product_features, on='id_producto', how='left')
    
    # Verificar si hay productos en test que no están en referencia
    missing_products = df_test_with_features[df_test_with_features.isnull().any(axis=1)]['id_producto'].unique()
    
    if len(missing_products) > 0 and verbose:
        print(f"   ⚠️  ADVERTENCIA: {len(missing_products)} productos en test no tienen características en referencia:")
        print(f"      {missing_products}")
    
    if verbose:
        print(f"   - Datos fusionados: {df_test_with_features.shape}")
        print(f"   - Columnas después del merge: {len(df_test_with_features.columns)}")
    
    # 4. Descomponer fechas
    if verbose:
        print("\n4. Añadiendo variables de descomposición de fecha...")
        
    df_test_processed = decompose_date(df_test_with_features)
    
    if verbose:
        print("   - Variables de fecha añadidas: month, year, day_of_week, weekend, quarter")
    
    # 5. Convertir tipos de datos
    if verbose:
        print("\n5. Convirtiendo tipos de datos...")
        
    df_test_processed = convert_data_types_for_model(df_test_processed, is_global=False)
    
    if verbose:
        print("   - Tipos de datos convertidos correctamente")
        print(f"   - Shape después de conversión: {df_test_processed.shape}")
    
    # 6. Aplicar one-hot encoding
    encoding_info_test = {}
    
    if apply_encoding:
        if verbose:
            print("\n6. Aplicando One-Hot Encoding...")
            print("="*40)
        
        if encoding_info is not None:
            # Si se proporciona información de encoding previo, intentar mantener consistencia
            if verbose:
                print("   - Usando información de encoding previo para mantener consistencia")
            
            # Aplicar encoding manteniendo las mismas categorías
            df_test_processed, encoding_info_test = apply_consistent_encoding(
                df_test_processed, 
                encoding_info,
                exclude_cols=['id_producto'],
                verbose=verbose
            )
        else:
            # Aplicar encoding normal
            df_test_processed, encoding_info_test = apply_one_hot_encoding(
                df_test_processed, 
                drop_first=True, 
                exclude_cols=['id_producto']
            )
    else:
        if verbose:
            print("\n6. One-Hot Encoding omitido (apply_encoding=False)")
    
    # 7. Resumen final
    if verbose:
        print("\n" + "="*60)
        print("RESUMEN FINAL - DATOS DE PRUEBA PREPARADOS")
        print("="*60)
        print(f"Shape final: {df_test_processed.shape}")
        print(f"Columnas: {len(df_test_processed.columns)}")
        print(f"Memoria: {df_test_processed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if len(missing_products) > 0:
            print(f"\n⚠️  Productos sin características: {len(missing_products)}")
            
        if apply_encoding and encoding_info_test:
            total_encoded_vars = len(encoding_info_test)
            print(f"\nVariables categóricas procesadas: {total_encoded_vars}")
    
    return df_test_processed, encoding_info_test


def apply_consistent_encoding(df, reference_encoding_info, exclude_cols=None, verbose=True):
    """
    Aplica one-hot encoding manteniendo consistencia con un encoding previo.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame a procesar
    reference_encoding_info : dict
        Información de encoding de referencia (de apply_one_hot_encoding previo)
    exclude_cols : list, default=None
        Columnas a excluir del encoding
    verbose : bool, default=True
        Si True, muestra información del proceso
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con encoding aplicado
    dict
        Información del encoding aplicado
    """
    
    df_encoded = df.copy()
    encoding_info = {}
    
    # Identificar columnas categóricas
    categorical_cols = df_encoded.select_dtypes(include=['category']).columns.tolist()
    
    # Excluir columnas si se especifica
    if exclude_cols:
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    if verbose:
        print(f"   Variables categóricas encontradas: {categorical_cols}")
    
    if not categorical_cols:
        if verbose:
            print("   No se encontraron variables categóricas para procesar.")
        return df_encoded, encoding_info
    
    # Procesar cada columna categórica
    for col in categorical_cols:
        if col in reference_encoding_info:
            # Usar información de referencia
            ref_info = reference_encoding_info[col]
            reference_categories = ref_info['original_categories']
            expected_dummy_cols = ref_info['dummy_columns']
            
            if verbose:
                print(f"   Procesando {col} con categorías de referencia: {reference_categories}")
            
            # Crear dummies usando las mismas categorías de referencia
            # Primero, asegurar que todas las categorías de referencia estén presentes
            current_categories = df_encoded[col].cat.categories.tolist()
            
            # Añadir categorías faltantes si es necesario
            missing_cats = set(reference_categories) - set(current_categories)
            if missing_cats:
                if verbose:
                    print(f"     Añadiendo categorías faltantes: {missing_cats}")
                df_encoded[col] = df_encoded[col].cat.add_categories(list(missing_cats))
            
            # Reordenar categorías para que coincidan con la referencia
            df_encoded[col] = df_encoded[col].cat.reorder_categories(reference_categories)
            
            # Crear dummies
            dummies = pd.get_dummies(
                df_encoded[col], 
                prefix=col, 
                drop_first=True,
                dtype='int8'
            )
            
            # Asegurar que todas las columnas esperadas estén presentes
            for expected_col in expected_dummy_cols:
                if expected_col not in dummies.columns:
                    dummies[expected_col] = 0
            
            # Mantener solo las columnas esperadas y en el mismo orden
            dummies = dummies[expected_dummy_cols]
            
            # Guardar información
            encoding_info[col] = {
                'original_categories': reference_categories,
                'dummy_columns': dummies.columns.tolist(),
                'dropped_category': reference_categories[0]
            }
            
            # Concatenar al DataFrame
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
            
            if verbose:
                print(f"     Columnas creadas: {dummies.columns.tolist()}")
                
        else:
            # Si no hay información de referencia, aplicar encoding normal
            if verbose:
                print(f"   {col}: No hay información de referencia, aplicando encoding normal")
            
            categories = df_encoded[col].cat.categories.tolist()
            dummies = pd.get_dummies(
                df_encoded[col], 
                prefix=col, 
                drop_first=True,
                dtype='int8'
            )
            
            encoding_info[col] = {
                'original_categories': categories,
                'dummy_columns': dummies.columns.tolist(),
                'dropped_category': categories[0]
            }
            
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
    
    if verbose:
        print(f"   Encoding consistente completado. Shape: {df_encoded.shape}")
    
    return df_encoded, encoding_info




import pandas as pd
from datetime import datetime

def crear_dataset_test_completo(df_all, ruta_demanda_test="Data/demanda_test.csv"):
    """
    Función para crear un dataset completo de test usando df_all y demanda_test.csv
    manteniendo el mismo orden de columnas que df_all
    
    Parameters:
    - df_all: DataFrame que ya contiene todas las variables de productos
    - ruta_demanda_test: Ruta al archivo demanda_test.csv
    
    Returns:
    - DataFrame con todas las variables procesadas para test en el mismo orden que df_all
    """
    
    # Cargar datos de test
    df_test = pd.read_csv(ruta_demanda_test)
    
    # Convertir date a datetime
    df_test['date'] = pd.to_datetime(df_test['date'])
    
    # Variables del producto que necesitamos traer de df_all
    variables_producto = ['premium', 'marca_exclusiva', 'estacional', 
                         'categoria_bebidas', 'categoria_carnes_y_aves', 'categoria_cereales_y_productos_secos',
                         'categoria_congelados', 'categoria_frutas_y_verduras', 'categoria_jabones',
                         'categoria_panaderia_y_panificados', 'categoria_productos_enlatados_y_alimentos_envasados',
                         'categoria_productos_lacteos', 'categoria_shampoos',
                         'tamaño_Mediano', 'tamaño_Pequeño']
    
    # Seleccionar columnas necesarias de df_all (solo una fila por producto)
    df_all_subset = df_all[['id_producto'] + variables_producto].drop_duplicates(subset=['id_producto'])
    
    # Merge para traer las variables del producto
    df_resultado = df_test.merge(df_all_subset, on='id_producto', how='left')
    
    # Crear variables temporales basadas en la fecha
    df_resultado['month'] = df_resultado['date'].dt.month
    df_resultado['year'] = df_resultado['date'].dt.year
    df_resultado['day_of_week'] = df_resultado['date'].dt.dayofweek
    df_resultado['weekend'] = df_resultado['day_of_week'].isin([5, 6])  # Sábado y domingo
    df_resultado['quarter'] = df_resultado['date'].dt.quarter
    
    # Crear variable Entrada_competidor (1 después del 2 de julio de 2021, 0 antes)
    fecha_entrada_competidor = datetime(2021, 7, 2)
    df_resultado['Entrada_competidor'] = (df_resultado['date'] > fecha_entrada_competidor).astype(int)
    
    # Obtener el orden de columnas de df_all (excluyendo 'demanda' que no está en test)
    columnas_df_all = [col for col in df_all.columns if col != 'demanda']
    
    # Reordenar las columnas del resultado para que coincidan con df_all
    # Solo seleccionar las columnas que existen en ambos DataFrames
    columnas_finales = [col for col in columnas_df_all if col in df_resultado.columns]
    
    # Seleccionar y reordenar columnas
    df_final = df_resultado[columnas_finales].copy()
    
    return df_final

def crear_productos_dict_optimizado(df_all, target_col='demanda', test_size=0.2, 
                                   save_files=False, verbose=True):
    """
    Función optimizada para crear y validar el diccionario de productos con estructura correcta.
    
    Parameters:
    -----------
    df_all : pandas.DataFrame
        DataFrame completo con todos los productos
    target_col : str, default='demanda'
        Nombre de la columna objetivo
    test_size : float, default=0.2
        Proporción para el conjunto de prueba
    save_files : bool, default=False
        Si True, guarda archivos CSV por producto
    verbose : bool, default=True
        Si True, muestra información detallada
        
    Returns:
    --------
    dict
        Diccionario con estructura productos_dict[id_producto] = {'train': df, 'test': df}
    """
    
    if verbose:
        print("="*60)
        print("CREACIÓN Y VALIDACIÓN DE PRODUCTOS_DICT")
        print("="*60)
        print("Regenerando productos_dict con estructura optimizada...")
    
    # Eliminar productos_dict anterior si existe en el scope global
    import gc
    gc.collect()  # Limpiar memoria
    
    if verbose:
        print("1. Creando diccionario de productos...")
    
    # Crear nuevo productos_dict con la estructura correcta
    productos_dict = separar_productos(
        df_local=df_all,
        target_col=target_col,
        test_size=test_size,
        save_files=save_files,
        base_path="data_by_product" if save_files else None
    )
    
    if verbose and productos_dict:
        print("\n2. Validando estructura del diccionario...")
        
        # Información general
        total_productos = len(productos_dict)
        print(f"   Total productos procesados: {total_productos}")
        
        # Seleccionar producto ejemplo para validación
        sample_key = list(productos_dict.keys())[0]
        sample_data = productos_dict[sample_key]
        
        print(f"\n3. Validación con producto ejemplo (ID: {sample_key}):")
        print(f"   - Tipo de dato: {type(sample_data)}")
        print(f"   - Claves disponibles: {list(sample_data.keys())}")
        
        # Validar estructura train/test
        if 'train' in sample_data and 'test' in sample_data:
            train_df = sample_data['train']
            test_df = sample_data['test']
            
            print(f"\n4. Detalles del conjunto TRAIN:")
            print(f"   - Tipo: {type(train_df)}")
            print(f"   - Forma: {train_df.shape}")
            print(f"   - Columnas: {len(train_df.columns)}")
            print(f"   - Primeras columnas: {list(train_df.columns[:5])}...")
            
            print(f"\n5. Detalles del conjunto TEST:")
            print(f"   - Tipo: {type(test_df)}")
            print(f"   - Forma: {test_df.shape}")
            print(f"   - Columnas: {len(test_df.columns)}")
            
            # Verificar consistencia de columnas
            train_cols = set(train_df.columns)
            test_cols = set(test_df.columns)
            columnas_coinciden = train_cols == test_cols
            
            print(f"\n6. Validación de consistencia:")
            print(f"   - Columnas train == test: {columnas_coinciden}")
            
            if not columnas_coinciden:
                print(f"   - Solo en train: {train_cols - test_cols}")
                print(f"   - Solo en test: {test_cols - train_cols}")
            
            # Verificar que contiene la columna objetivo
            tiene_target = target_col in train_df.columns
            print(f"   - Contiene columna '{target_col}': {tiene_target}")
            
            # Verificar fechas si existen
            if 'date' in train_df.columns:
                train_date_range = f"{train_df['date'].min()} a {train_df['date'].max()}"
                test_date_range = f"{test_df['date'].min()} a {test_df['date'].max()}"
                print(f"   - Rango fechas train: {train_date_range}")
                print(f"   - Rango fechas test: {test_date_range}")
        
        # Estadísticas generales de todos los productos
        print(f"\n7. Estadísticas generales de todos los productos:")
        
        train_sizes = []
        test_sizes = []
        
        for prod_id, data in productos_dict.items():
            if 'train' in data and 'test' in data:
                train_sizes.append(len(data['train']))
                test_sizes.append(len(data['test']))
        
        if train_sizes and test_sizes:
            print(f"   - Tamaños de train: min={min(train_sizes)}, max={max(train_sizes)}, promedio={sum(train_sizes)/len(train_sizes):.1f}")
            print(f"   - Tamaños de test: min={min(test_sizes)}, max={max(test_sizes)}, promedio={sum(test_sizes)/len(test_sizes):.1f}")
        
        print(f"\n✅ productos_dict creado y validado exitosamente")
        print(f"   📊 {total_productos} productos listos para modelado")
        
    elif not productos_dict:
        print("⚠️ ERROR: No se pudieron procesar productos")
        return {}
    
    return productos_dict