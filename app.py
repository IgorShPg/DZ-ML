import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        scaler, model, feature_names = pickle.load(f)
    return model, scaler, feature_names


def prepare_features(df, feature_names):
    """Приводим данные к формату обучения модели."""
    df_proc = df.copy()
    # Преобразуем категориальные признаки в строки (как при обучении)
    for col in feature_names:
        if col in df_proc.columns:
            if df_proc[col].dtype in ('object', 'bool'):
                df_proc[col] = df_proc[col].astype(str)
    return df_proc[feature_names]


# Загружаем модель
try:
    MODEL, SCALER, FEATURE_NAMES = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()


# --- Основной интерфейс ---
st.title("🎯 Предсказание стоимости автомобилей")

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)

fig = sns.pairplot(df.drop('Unnamed: 0', axis=1))
st.pyplot(fig)

numeric_cols = df.drop('Unnamed: 0', axis=1).select_dtypes(include=['number']).columns.tolist()

if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    plt.title("Матрица корреляций", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    st.pyplot(fig)

try:
    features = df[FEATURE_NAMES]
    y_pred = MODEL.predict(features)
    
    df['prediction'] = y_pred
except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()



st.subheader("Результаты")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Всего клиентов", len(df))
with col2:
    churn_rate = df['prediction'].mean() * 100
    st.metric("Предсказанная стоимость", f"{churn_rate:.1f}%")




st.subheader("Важность признаков модели")

if hasattr(MODEL, 'coef_'):
    if len(MODEL.coef_.shape) > 1:
        coefficients = MODEL.coef_[0]  
    else:
        coefficients = MODEL.coef_
    
    feature_importance = pd.DataFrame({
        'Признак': FEATURE_NAMES,
        'Коэффициент': coefficients,
        'Абсолютное_значение': abs(coefficients)
    }).sort_values('Абсолютное_значение', ascending=False)
    

if feature_importance is not None:
    
    st.markdown("Полная таблица коэффициентов")
    st.dataframe(
        feature_importance.style.background_gradient(
            subset=['Коэффициент'] if 'Коэффициент' in feature_importance.columns else ['Важность'],
            cmap='RdYlGn',
            vmin=-1 if 'Коэффициент' in feature_importance.columns else None,
            vmax=1 if 'Коэффициент' in feature_importance.columns else None
        ).format({
            'Коэффициент': '{:.4f}',
            'Абсолютное_значение': '{:.4f}',
            'Важность': '{:.4f}'
        }),
        use_container_width=True,
        height=700 
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'Коэффициент' in feature_importance.columns:
            st.metric("Макс. коэффициент", f"{feature_importance['Коэффициент'].max():.4f}")
    with col2:
        if 'Коэффициент' in feature_importance.columns:
            st.metric("Мин. коэффициент", f"{feature_importance['Коэффициент'].min():.4f}")
    with col3:
        st.metric("Всего признаков", len(feature_importance))




