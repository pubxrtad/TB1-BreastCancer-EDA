import streamlit as st
import pandas as pd
#import joblib
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

#Configuración de la página Streamlit
st.set_page_config(
    page_title='Predicción de Cáncer de Mama: Aplicación de CRISP-DM',
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Cargar el modelo pre-entrenado
@st.cache_resource
def load_model_from_disk():
    model = joblib.load('optimized_rf_model.joblib')
    return model

model = load_model_from_disk()

#Cargar el DataFrame original para visualizaciones y preparación de datos
@st.cache_data
def load_data_and_prepare():
    df_raw = pd.read_csv(r"/content/breast_cancer_wisconsin.csv")

    # Preprocesamiento de datos
    df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ', '_')
    df_raw = df_raw.rename(columns={'target': 'diagnosis'})
    # Mapping: 0 from original 'target' was malignant, 1 was benign.
    df_raw['diagnosis_binary'] = df_raw['diagnosis'].map({0: 1, 1: 0})
    X_for_model = df_raw.drop(columns=['diagnosis', 'diagnosis_binary'])
    y = df_raw['diagnosis_binary']

    return df_raw, X_for_model, y

df_for_plots, X_for_model_processed, y_target = load_data_and_prepare()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_for_model_processed, y_target, test_size=0.2, random_state=42, stratify=y_target)

# Recalcular métricas for the optimized RF model using the loaded 'model' (optimized_rf_model)
preds_optimized = model.predict(X_test)
probs_optimized = model.predict_proba(X_test)[:, 1]

accuracy_optimized = accuracy_score(y_test, preds_optimized)
precision_optimized = precision_score(y_test, preds_optimized)
recall_optimized = recall_score(y_test, preds_optimized)
f1_optimized = f1_score(y_test, preds_optimized)
auc_roc_optimized = roc_auc_score(y_test, probs_optimized)

# Recalculate predictions_df
predictions_df = pd.DataFrame({
    'Diagnostico actual': y_test,
    'Diagnostico predecido': preds_optimized,
    'Probabilidad de malignidad': probs_optimized
})
predictions_df['Diagnostico actual etiqueta'] = predictions_df['Diagnostico actual'].map({0: 'Benigno', 1: 'Maligno'})
predictions_df['Diagnostico predecido etiqueta'] = predictions_df['Diagnostico predecido'].map({0: 'Benigno', 1: 'Maligno'})

# Recalculate feature importances
feature_importances = model.feature_importances_
feature_names_for_input = X_for_model_processed.columns.tolist() # Obtener nombres de las 30 características
importances_df = pd.DataFrame({'variable': feature_names_for_input, 'importancia': feature_importances})
importances_df = importances_df.sort_values(by='importancia', ascending=False)
top_n_features = 5
top_features = importances_df.head(top_n_features)

# BARRA LATERAL
st.sidebar.title("➲")
selection = st.sidebar.radio(
    "Ir a",
    [
        "Introducción",
        "Comparación de Modelos",
        "Métricas del RF Optimizado",
        "Análisis de Variables Clave",
        "Evaluación del RF Optimizado",
        "Predicción Interactiva",
        "Conclusiones Finales"
    ]
)

# CONTENIDO PRINCIPAL
st.title('✦ ᴘʀᴇᴅɪᴄᴄɪóɴ ᴅᴇ ᴄáɴᴄᴇʀ ᴅᴇ ᴍᴀᴍᴀ: ᴀᴘʟɪᴄᴀᴄɪóɴ ᴅᴇ ᴄʀɪsᴘ-ᴅᴍ ✦')

if selection == "Introducción":
    st.subheader("➤ Problemática")
    st.write("""
    Según la OMS, el cáncer de mama es una de las principales causas de mortalidad en mujeres a nivel mundial. En 2022, se diagnosticaron cerca de 2,3 millones de casos nuevos en mujeres y se registraron aproximadamente 670 000 muertes. La detección temprana puede aumentar las posibilidades de supervivencia, cuando se detecta en etapas iniciales, la tasa de curación y éxito del tratamiento es considerablemente más alta.
    """)
    st.subheader("➤ Objetivos del Proyecto:")
    st.markdown("""
    Desarrollar un modelo de clasificación basado que permita predecir la presencia de cáncer de mama a partir del dataset Breast Cancer Wisconsin, utilizando la metodología CRISP-DM para asegurar un proceso estructurado desde el análisis de datos hasta la evaluación del rendimiento del modelo.
    """)
    st.subheader("➤ Objetivos Específicos:")
    st.markdown("""
    - Realizar el entendimiento y exploración inicial del dataset, identificando la distribución de las variables,
      correlaciones relevantes y características representativas entre tumores benignos y malignos.
    - Preprocesar y preparar el conjunto de datos, aplicando limpieza, normalización y codificación necesaria para
      garantizar la calidad del entrenamiento del modelo.
    - Entrenar y comparar distintos algoritmos de clasificación, con el fin de determinar cuál presenta el mejor
      desempeño predictivo.
    - Evaluar los modelos mediante métricas cuantitativas, priorizando el Recall para minimizar falsos negativos
      en el diagnóstico.
    - Implementar visualizaciones interpretativas que faciliten la comprensión de resultados para análisis clínico
      y presentación.
    - Construir una visualización HTML simple, donde el usuario pueda ingresar valores y obtener una predicción del
      modelo junto con gráficos explicativos.
    """)
    st.image("/content/1_pxFCmhRFTighUn88baLcSA.png", caption="Cáncer de Mama", use_container_width=False)

elif selection == "Comparación de Modelos":
    st.header("Comparación de Modelos de Clasificación")
    st.write("Se evaluaron 3 modelos de clasificación para determinar cuál ofrece el mejor rendimiento en la predicción del cáncer de mama.")

    df_results_data = {
        "Modelo": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.938596, 0.973684, 0.964912],
        "Precision": [0.972973, 1.000000, 1.000000],
        "Recall": [0.857143, 0.928571, 0.904762],
        "F1-score": [0.911392, 0.962963, 0.950000],
        "AUC-ROC": [0.993717, 0.994378, 0.993056]
    }
    df_results = pd.DataFrame(df_results_data)
    st.dataframe(df_results)
    st.markdown("""
    - Los tres modelos muestran métricas superiores al 93% en casi todos los indicadores, lo que es un rendimiento excelente.
    - **Random Forest** fue seleccionado como el mejor modelo debido a su combinación de alta precisión y recall, minimizando los falsos negativos.
    """)

    # Matriz de calor de métricas
    st.subheader("➤ Visualización Comparativa de Métricas")
    df_results_plot = df_results.set_index("Modelo")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df_results_plot, annot=True, cmap="magma", fmt=".3f", ax=ax)
    ax.set_title("Comparación de métricas entre modelos")
    st.pyplot(fig)
    plt.close(fig)

elif selection == "Métricas del RF Optimizado":
    st.header("Métricas del Modelo Random Forest Optimizado")
    st.write("Rendimiento detallado del modelo Random Forest con hiperparámetros óptimos en el conjunto de prueba.")

    metrics_df = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"],
        "Valor": [accuracy_optimized, precision_optimized, recall_optimized, f1_optimized, auc_roc_optimized]
    })
    st.dataframe(metrics_df.set_index("Métrica"))

    st.markdown("""
    - **Accuracy**: Proporción de predicciones correctas.
    - **Precision**: De las predicciones positivas, cuántas fueron realmente positivas (minimiza falsos positivos).
    - **Recall (Sensibilidad)**: De todos los casos positivos reales, cuántos fueron correctamente identificados (minimiza falsos negativos).
    - **F1-score**: Media armónica de precisión y recall.
    - **AUC-ROC**: Capacidad del modelo para distinguir entre clases.
    """)

    st.subheader("➤ Matriz de Confusión")
    cm = confusion_matrix(y_test, preds_optimized)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
                xticklabels=['Benigno (0)', 'Maligno (1)'],
                yticklabels=['Benigno (0)', 'Maligno (1)'], ax=ax)
    ax.set_title('Matriz de Confusión del RF Optimizado')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    - La matriz de confusión muestra el número de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.
    - Un bajo número de falsos negativos es lo prioritario en el diagnóstico de cáncer para evitar diagnósticos erróneos.
    """)

elif selection == "Análisis de Variables Clave":
    st.header("Top 5 Variables Más Importantes")
    st.write("Las características que más contribuyen a la predicción del modelo Random Forest.")

    st.dataframe(top_features.set_index('variable'))

    # Gráfico de barras de importancia de características
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='importancia', y='variable', data=top_features, palette='viridis', hue='variable', legend=False, ax=ax)
    ax.set_title(f'Top {top_n_features} Variables más Importantes en el Random Forest Optimizado')
    ax.set_xlabel('Importancia')
    ax.set_ylabel('Variable')
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("➤ Visualizaciones de Variables Clave")
    st.write("Exploración visual de cómo estas variables se relacionan con el diagnóstico.")

    # Scatter Plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter Plot para worst_perimeter vs. worst_area
    sns.scatterplot(x='worst_perimeter', y='worst_area', hue='diagnosis_binary', data=df_for_plots, ax=axes[0], palette='viridis', s=50, alpha=0.7)
    axes[0].set_title('Scatter Plot: Worst Perimeter vs. Worst Area por Diagnóstico')
    axes[0].set_xlabel('Worst Perimeter')
    axes[0].set_ylabel('Worst Area')
    axes[0].legend(title='Diagnóstico (0:Benigno, 1:Maligno)')

    # Scatter Plot para mean_radius vs. mean_concave_points
    sns.scatterplot(x='mean_radius', y='mean_concave_points', hue='diagnosis_binary', data=df_for_plots, ax=axes[1], palette='viridis', s=50, alpha=0.7)
    axes[1].set_title('Scatter Plot: Mean Radius vs. Mean Concave Points por Diagnóstico')
    axes[1].set_xlabel('Mean Radius')
    axes[1].set_ylabel('Mean Concave Points')
    axes[1].legend(title='Diagnóstico (0:Benigno, 1:Maligno)')

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    - El primer gráfico muestra una fuerte correlación positiva entre el peor perímetro y el peor área. Esto indica que los tumores malignos tienden a tener células significativamente más grandes.
    - El segundo gráfico ilustra cómo la combinación de tamaño e irregularidad es un fuerte indicador de malignidad.
    """)

    # Boxplots Comparativos
    st.subheader("➤ Boxplots Comparativos de Puntos Cóncavos")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Boxplot para worst_concave_points
    sns.boxplot(x='diagnosis_binary', y='worst_concave_points', data=df_for_plots, ax=axes[0], palette='viridis', hue='diagnosis_binary', legend=False) # Agregado hue y legend
    axes[0].set_title('Distribución de worst_concave_points por Diagnóstico')
    axes[0].set_xlabel('Diagnóstico (0: Benigno, 1: Maligno)')
    axes[0].set_ylabel('Worst Concave Points')

    # Boxplot para mean_concave_points
    sns.boxplot(x='diagnosis_binary', y='mean_concave_points', data=df_for_plots, ax=axes[1], palette='viridis', hue='diagnosis_binary', legend=False) # Agregado hue y legend
    axes[1].set_title('Distribución de mean_concave_points por Diagnóstico')
    axes[1].set_xlabel('Diagnóstico (0: Benigno, 1: Maligno)')
    axes[1].set_ylabel('Mean Concave Points')

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    - Los boxplots muestran una diferencia marcada en la distribución de los puntos cóncavos, indicando mayor irregularidad y complejidad en los bordes celulares de los tumores malignos.
    """)

    # Histogramas Comparativos
    st.subheader("➤ Histogramas de Distribución de Variables Clave")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Histograma para worst_perimeter
    sns.histplot(df_for_plots, x='worst_perimeter', hue='diagnosis_binary', kde=True, ax=axes[0], palette='viridis')
    axes[0].set_title('Distribución de worst_perimeter por Diagnóstico')
    axes[0].set_xlabel('Worst Perimeter')
    axes[0].set_ylabel('Frecuencia')

    # Histograma para worst_area
    sns.histplot(df_for_plots, x='worst_area', hue='diagnosis_binary', kde=True, ax=axes[1], palette='viridis')
    axes[1].set_title('Distribución de worst_area por Diagnóstico')
    axes[1].set_xlabel('Worst Area')
    axes[1].set_ylabel('Frecuencia')

    # Histograma para worst_radius
    sns.histplot(df_for_plots, x='worst_radius', hue='diagnosis_binary', kde=True, ax=axes[2], palette='viridis')
    axes[2].set_title('Distribución de worst_radius por Diagnóstico')
    axes[2].set_xlabel('Worst Radius')
    axes[2].set_ylabel('Frecuencia')

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    - Los histogramas muestran una clara separación en la distribución de estas tres variables entre tumores benignos y malignos, confirmando su importancia.
    """)

elif selection == "Evaluación del RF Optimizado":
    st.header("Evaluación Detallada del Modelo Random Forest Optimizado")

    st.subheader("➤ Curva ROC")
    fig, ax = plt.subplots(figsize=(15, 4))
    roc_display = RocCurveDisplay.from_estimator(
        model, X_test, y_test, ax=ax, name='Optimized Random Forest'
    )
    ax.set_title('Curva ROC para el modelo Random Forest Optimizado')
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    - La curva ROC muestra una excelente capacidad del modelo para distinguir entre clases, con un área bajo la curva (AUC) de aproximadamente **0.9944**.
    - Un AUC cercano a 1.0 indica que el modelo tiene una alta probabilidad de clasificar correctamente los casos.
    """)

    st.subheader("➤ Predicciones Detalladas")
    st.write("Muestra las predicciones del modelo en una parte del conjunto de prueba.")
    st.dataframe(predictions_df.head(10))

elif selection == "Predicción Interactiva":
    st.header("Realizar una Predicción de Cáncer de Mama")
    st.write("Ajusta los valores de las características del tumor para obtener una predicción.")

    # Crear la interfaz de usuario para la entrada de datos
    user_input_data = {}
    st.sidebar.subheader("Valores de las Características del Tumor")

    # Obtener valores min, max y mean de X_for_model_processed para sliders
    min_vals = X_for_model_processed.min()
    max_vals = X_for_model_processed.max()
    mean_vals = X_for_model_processed.mean()

    for feature_name in feature_names_for_input:
        # Usar slider para float y number_input para int
        default_value = float(mean_vals[feature_name])
        min_value = float(min_vals[feature_name])
        max_value = float(max_vals[feature_name])

        # Asegurar que el valor por defecto esté dentro del rango min/max
        if not (min_value <= default_value <= max_value):
            default_value = min_value # Fallback if mean is outside range for some reason

        user_input_data[feature_name] = st.sidebar.slider(
            f"{feature_name.replace('_', ' ').title()}",
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=(max_value - min_value) / 100.0
        )

    input_df = pd.DataFrame([user_input_data])

    # Realizar la predicción
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    st.subheader("Resultados de la Predicción:")
    if prediction[0] == 1:
        st.error(f"El tumor es **Maligno** con una probabilidad del **{prediction_proba[0]*100:.2f}%**.")
        st.markdown("<p style='color:red;'>Se recomienda una evaluación médica urgente.</p>", unsafe_allow_html=True)
    else:
        st.success(f"El tumor es **Benigno** con una probabilidad del **{(1 - prediction_proba[0])*100:.2f}%**.")
        st.markdown("<p style='color:green;'>El riesgo de malignidad es bajo, pero se recomienda seguimiento médico.</p>", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Valores de Entrada:")
    st.dataframe(input_df.T.rename(columns={0: 'Valor Ingresado'}))

elif selection == "Conclusiones Finales":
    st.header("Conclusiones y Recomendaciones")
    st.write("""
    El desarrollo de este proyecto siguiendo la metodología CRISP-DM ha permitido identificar
    que el modelo **Random Forest** ofrece un rendimiento excepcional para la predicción
    del cáncer de mama, con alta precisión y un muy bajo número de falsos negativos.
    """)

    st.subheader("Hallazgos Clave:")
    st.markdown("""
    - Las características morfológicas como el perímetro, área, radio y los puntos cóncavos de las células son los indicadores más influyentes en el diagnóstico.
    - El modelo Random Forest supera a la Regresión Logística y XGBoost en este contexto.
    - La alta métrica de Recall (0.9286) y AUC-ROC (0.9944) demuestran la robustez del modelo.
    """)

    st.subheader("Recomendaciones:")
    st.markdown("""
    - Probar el modelo con conjuntos de datos de diferentes fuentes para asegurar su generalización.
    - Aunque Random Forest es menos interpretable que otros modelos, técnicas como SHAP o LIME podrían ofrecer mayor transparencia.
    - Explorar la integración de este modelo en sistemas de apoyo a la decisión clínica para asistir a los profesionales de la salud.
    - Monitorear el rendimiento del modelo en producción y reentrenarlo periódicamente con nuevos datos.
    """)

    st.info("Gracias por revisar este dashboard. ¡La detección temprana salva vidas!")
