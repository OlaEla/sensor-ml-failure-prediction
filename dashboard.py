import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Загрузка артефактов
@st.cache_data
def load_artifacts():
    with open('failure_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('feature_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    with open('model_features.pkl', 'rb') as file:
        model_features = pickle.load(file)
    
    return model, scaler, model_features

model, scaler, model_features = load_artifacts()

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv("processed_full_dataset.csv")

df = load_data()

# Преобразуем данные в стандартизованный формат
df_scaled = scaler.transform(df[model_features])

# Прогнозируем вероятности отказа с помощью модели
failure_probabilities = model.predict_proba(df_scaled)[:, 1]  # Получаем вероятность класса "1" (отказ)
df['Failure_Risk_Prob'] = failure_probabilities

# === Название дашборда ===
st.title("Система предиктивного обслуживания")
st.subheader("Анализ отказов оборудования")

# # === Боковая панель ===
# with st.sidebar:
#     st.header("Настройки отображения")
#     threshold = st.slider("Порог вероятности отказа", 0.0, 1.0, 0.35)
#     machine_type = st.selectbox("Тип оборудования", options=["Все"] + sorted(data["Machine_Type"].unique().tolist()))

# Базовая информация
st.markdown("### Общая информация о данных")
st.write("Количество записей:", df.shape[0])
st.write("Типы станков:", df['Machine_Type'].unique())

# Разбиение экрана на 2 колонки
col1, col2 = st.columns(2)

# График 1: Распределение по типам станков
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Machine_Type', ax=ax1)
    ax1.set_title("Распределение по типам станков")
    st.pyplot(fig1)

# График 2: Распределение по риску отказа
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Failure_Risk', ax=ax2)
    ax2.set_title("Распределение по риску отказа")
    st.pyplot(fig2)

# График 3: Распределение температуры по типу станка
with col1:
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Machine_Type', y='Temperature', ax=ax3)
    ax3.set_title("Температура по типу станка")
    st.pyplot(fig3)

# График 4: Вибрация по типу станка
with col2:
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Machine_Type', y='Vibration', ax=ax4)
    ax4.set_title("Вибрация по типу станка")
    st.pyplot(fig4)

# --- Сравнение по типам станков и анализ по целевой переменной ---
st.markdown("### Сравнение типов станков и анализ целевой переменной")

# --- Выбор типа станка ---
with st.sidebar:
    st.markdown("### Выберите тип станка для анализа")
    selected_type = st.selectbox("Тип станка", df['Machine_Type'].unique())

    filtered_df = df[df['Machine_Type'] == selected_type]

    # --- Выбор уровня риска ---
    risk_threshold = st.slider("Порог вероятности отказа (0=низкий риск, 1=высокий риск)", 0.0, 1.0, 0.5)

    # Информативные индикаторы

    st.markdown(f"**Фильтрованных записей:** {len(filtered_df)}")

    avg_risk = filtered_df['Failure_Risk_Prob'].mean()
    st.markdown(f"**Средняя вероятность отказа:** {avg_risk:.2f}")

    failure_rate = filtered_df['Failure_Risk'].mean() * 100
    st.markdown(f"**Доля отказов:** {failure_rate:.1f}%")

# Фильтрация по риску
at_risk = filtered_df[filtered_df['Failure_Risk_Prob'] >= risk_threshold]

# # Отображение таблицы с отфильтрованными данными
# st.markdown(f"### Станки типа {selected_type} с риском отказа ≥ {risk_threshold}")
# st.dataframe(at_risk)

# --- Отображение результатов ---
# with st.sidebar:
st.markdown(f"### Станки типа {selected_type} с риском отказа ≥ {risk_threshold}")
if at_risk.empty:
    st.warning("Нет записей, удовлетворяющих текущему фильтру по вероятности. Попробуйте снизить порог.")
else:
    st.dataframe(at_risk)

# st.markdown("### Распределение вероятности отказа")
with col1:
    fig_prob, ax_prob = plt.subplots()
    sns.histplot(df['Failure_Risk_Prob'], bins=30, kde=True, ax=ax_prob)
    ax_prob.set_title("Гистограмма вероятности отказа")
    st.pyplot(fig_prob)

# 1. Boxplot вероятности отказа по всем типам
with col2:
    # st.subheader("Распределение вероятности отказа по всем типам станков")

    fig_all_1, ax_all_1 = plt.subplots()
    sns.boxplot(data=df, x='Machine_Type', y='Failure_Risk_Prob', palette="pastel", ax=ax_all_1)
    ax_all_1.set_xlabel("Тип станка")
    ax_all_1.set_ylabel("Вероятность отказа")
    ax_all_1.set_title("Распределение вероятности отказа по всем типам станков")
    st.pyplot(fig_all_1)

# 2. Средняя вероятность отказа по типам станков
# with col2:
    # st.subheader("Средняя вероятность отказа по типам станков")

# avg_probs_all = df.groupby("Machine_Type")['Failure_Risk_Prob'].mean().reset_index()
# # st.bar_chart(avg_probs_all.set_index("Machine_Type"))
# fig, ax = plt.subplots(figsize=(6, 4))
# sns.barplot(data=avg_probs_all, x='Machine_Type', y='Failure_Risk_Prob', ax=ax)
# ax.set_title("Средняя вероятность отказа по типам станков")
# st.pyplot(fig)

# 3. Распределение реальных отказов (целевой переменной)
with col1:
    # st.subheader("Распределение целевой переменной (реальных отказов)")
    failure_counts_all = df['Failure_Risk'].value_counts().sort_index()
    # st.bar_chart(failure_counts_all.rename({0: 'Без отказа', 1: 'Отказ'}))
    failure_counts_all.index = ['Без отказа', 'Отказ']

    fig3, ax3 = plt.subplots()
    sns.barplot(x=failure_counts_all.index, y=failure_counts_all.values, palette='Blues', ax=ax3)
    ax3.set_title("Распределение количества отказов")
    ax3.set_xlabel("Состояние")
    ax3.set_ylabel("Количество записей")
    st.pyplot(fig3)

# 4. Доля отказов по типу станка
with col2:
    # st.subheader("Доля отказов по типу станка (%)")
    failure_rate_by_type = df.groupby("Machine_Type")['Failure_Risk'].mean().reset_index()
    failure_rate_by_type['Failure_Risk'] *= 100  # в проценты
    # fig_all_2, ax_all_2 = plt.subplots()
    # sns.barplot(data=failure_rate_by_type, x='Machine_Type', y='Failure_Risk', palette='Reds', ax=ax_all_2)
    # ax_all_2.set_ylabel("Процент отказов (%)")
    # st.pyplot(fig_all_2)
    fig4, ax4 = plt.subplots()
    sns.barplot(data=failure_rate_by_type, x='Machine_Type', y='Failure_Risk', palette='Reds', ax=ax4)
    ax4.set_title("Процент отказов по типу станка")
    ax4.set_xlabel("Тип станка")
    ax4.set_ylabel("Процент отказов (%)")
    st.pyplot(fig4)

# --- Расширенная визуализация вероятности отказа по selected_type---

# Заголовок — до колонок
st.markdown("### Расширенная визуализация вероятности отказа по выбранному типу станков")

# Располагаем графики в двух колонках
col1, col2 = st.columns(2)
# Применяем фильтр по типу станка
filtered_viz_df = df[df['Machine_Type'] == selected_type]
# st.markdown("## Расширенная визуализация вероятности отказа по выбранному типу станков")
# 1. Гистограмма + KDE
with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_viz_df['Failure_Risk_Prob'], bins=20, kde=True, ax=ax1, color='skyblue')
    ax1.set_title(f"Гистограмма + плотность вероятности отказа ({selected_type})")
    ax1.set_xlabel("Вероятность отказа")
    st.pyplot(fig1)

# 2. Boxplot (в данном случае один тип, но оставляем структуру)
with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=filtered_viz_df, x='Machine_Type', y='Failure_Risk_Prob', ax=ax2, palette="Set2")
    ax2.set_title(f"Распределение вероятности отказа для {selected_type}")
    st.pyplot(fig2)

# --- Расширенная визуализация вероятности отказа ---
st.markdown("### Расширенная визуализация вероятности отказа")

# 1. Гистограмма + плотность KDE
with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Failure_Risk_Prob'], bins=20, kde=True, ax=ax1, color='skyblue')
    ax1.set_title("Гистограмма + плотность вероятности отказа")
    ax1.set_xlabel("Вероятность отказа")
    st.pyplot(fig1)

# 2. Boxplot по типу станков
with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='Machine_Type', y='Failure_Risk_Prob', ax=ax2, palette="Set2")
    ax2.set_title("Распределение вероятности отказа по типам станков")
    st.pyplot(fig2)

# with st.sidebar:
# 3. Среднее значение вероятности по типам
st.markdown("#### Средняя вероятность отказа по типам станков")
avg_probs = df.groupby("Machine_Type")['Failure_Risk_Prob'].mean().reset_index()
st.bar_chart(avg_probs.set_index("Machine_Type"))

# 4. Метрика: сколько станков превышают текущий порог
high_risk_count = df[df['Failure_Risk_Prob'] >= risk_threshold].shape[0]
st.metric(label=f"Станков с риском ≥ {risk_threshold}", value=high_risk_count)

# --- Визуализация распределения параметров ---
st.markdown("### Распределение параметров для выбранного типа станка")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=filtered_df, x='Failure_Risk', y='Temperature', ax=ax1)
    ax1.set_title('Температура vs Отказ')
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=filtered_df, x='Vibration', y='Power_Usage', hue='Failure_Risk', ax=ax2)
    ax2.set_title('Вибрация и Потребление энергии')
    st.pyplot(fig2)

# --- Средняя информация по каждому типу станка ---
st.markdown("### Средние значения параметров для каждого типа станка")
type_avg = df.groupby('Machine_Type')[['Temperature', 'Vibration', 'Power_Usage', 'Humidity']].mean()
st.dataframe(type_avg)

# # --- Расширенная аналитика ---
# st.markdown("## Средняя вероятность отказа по типу станка")

# # Проверим, есть ли нужный столбец
# if 'Failure_Risk_Prob' in df.columns:
#     # Агрегация по типу станка
#     avg_risk_by_type = df.groupby('Machine_Type')['Failure_Risk_Prob'].mean().reset_index()

#     # Визуализация
#     fig_avg, ax_avg = plt.subplots()
#     sns.barplot(data=avg_risk_by_type, x='Machine_Type', y='Failure_Risk_Prob', ax=ax_avg)
#     ax_avg.set_title("Средняя вероятность отказа по типу станка")
#     ax_avg.set_ylabel("Средняя вероятность отказа")
#     ax_avg.set_xlabel("Тип станка")
#     st.pyplot(fig_avg)
# else:
#     st.info("Вероятность отказа не загружена. Добавьте предсказания модели в датасет.")

# --- Корреляция между признаками ---
st.markdown("### Корреляция между признаками (heatmap)")

corr_features = df.select_dtypes(include=['float64', 'int64'])  # исключим категориальные
corr_matrix = corr_features.corr()

fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
ax_corr.set_title("Корреляция между числовыми признаками")
st.pyplot(fig_corr)

# --- PCA-график ---
with col1:    
    st.markdown("### Визуализация главных компонент (PCA)")


    fig_pca, ax_pca = plt.subplots()
    sns.scatterplot(data=df, x='PCA_1', y='PCA_2', hue='Machine_Type', palette='Set2', alpha=0.7, ax=ax_pca)
    ax_pca.set_title("Проекция данных на 2 главные компоненты (PCA)")
    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    st.pyplot(fig_pca)


# --- Визуализация кластеров (KMeans) ---
from sklearn.cluster import KMeans
with col2:
    st.markdown("### Кластеризация (KMeans на основе PCA)")

# Применим кластеризацию на PCA-компонентах

    X_pca = df[['PCA_1', 'PCA_2']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    df['Cluster'] = clusters

    fig_kmeans, ax_kmeans = plt.subplots()
    sns.scatterplot(data=df, x='PCA_1', y='PCA_2', hue='Cluster', palette='tab10', ax=ax_kmeans)
    ax_kmeans.set_title("Кластеры в PCA-пространстве")
    st.pyplot(fig_kmeans)


# --- Метрики качества модели ---
st.markdown("### Метрики качества модели")

# Применим модель к фильтрованным данным для получения метрик
X_filtered = filtered_df[model_features]  # Используем те же признаки для фильтрованных данных
y_true = filtered_df['Failure_Risk']  # Реальные значения

# Стандартизируем данные
X_scaled = scaler.transform(X_filtered)

# Прогнозы модели
y_pred = model.predict(X_scaled)
y_pred_prob = model.predict_proba(X_scaled)[:, 1]  # Для ROC

# Метрики
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# ROC-кривая
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Отображение метрик
st.write(f"Точность: {accuracy:.2f}")
st.write(f"Точность (Precision): {precision:.2f}")
st.write(f"Полнота (Recall): {recall:.2f}")
st.write(f"F1-мера: {f1:.2f}")

# Построение ROC-кривой
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Кривая')
ax_roc.legend(loc='lower right')
st.pyplot(fig_roc)

# # --- Возможность скачивания отфильтрованных данных ---

# st.markdown("### 📥 Скачивание отфильтрованных данных")

# # Сохранение отфильтрованных данных в CSV
# csv_data = at_risk.to_csv(index=False)

# # Кнопка для скачивания
# st.download_button(
#     label="Скачать отфильтрованные данные",
#     data=csv_data,
#     file_name="filtered_data.csv",
#     mime="text/csv"
# )

# Пример данных для скачивания
data = {'Column1': [1, 2, 3], 'Column2': [4, 5, 6]}
df = pd.DataFrame(data)
csv_data = df.to_csv(index=False)

# Стилизация кнопки через CSS
st.markdown("""
    <style>
    .stDownloadButton>button {
        background-color: #4CAF50; /* Зеленый фон */
        color: white; /* Белый текст */
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    .stDownloadButton>button:hover {
        background-color: #45a049; /* Темно-зеленый при наведении */
    }
    </style>
""", unsafe_allow_html=True)

# Кнопка для скачивания с изменением цвета при наведении
st.download_button(
    label="Скачать отфильтрованные данные",
    data=csv_data,
    file_name="filtered_data.csv",
    mime="text/csv"
)

# --- Подвал ---
st.markdown("---")
st.caption("Разработано в рамках дипломного проекта • 2025")

