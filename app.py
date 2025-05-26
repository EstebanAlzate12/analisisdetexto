# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import plotly.express as px
from gensim.summarization import summarize

# Descargar stopwords la primera vez
nltk.download('stopwords')

st.title("Análisis de Opiniones de Clientes")

# 1. Subida de archivo CSV
st.header("1. Sube tu archivo CSV")
st.write("El archivo debe tener una columna llamada 'opinion' con al menos 20 comentarios.")
uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'opinion' not in df.columns:
        st.error("El archivo debe tener una columna llamada 'opinion'.")
    elif len(df) < 20:
        st.error("El archivo debe tener al menos 20 opiniones.")
    else:
        st.success("Archivo cargado correctamente.")
        st.dataframe(df.head())

        # 2. Limpieza de texto y eliminación de stopwords
        st.header("2. Procesamiento de texto")
        stop_words = set(stopwords.words('spanish'))
        def clean_text(text):
            words = [word.lower() for word in str(text).split() if word.isalpha()]
            words = [word for word in words if word not in stop_words]
            return ' '.join(words)
        df['cleaned'] = df['opinion'].apply(clean_text)

        # 3. Visualizaciones
        st.header("3. Visualizaciones de palabras")
        all_text = ' '.join(df['cleaned'])
        # Nube de palabras
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.subheader("Nube de palabras")
        st.pyplot(fig)

        # Gráfico de barras: 10 palabras más frecuentes
        words = all_text.split()
        common_words = Counter(words).most_common(10)
        words_plot, counts_plot = zip(*common_words)
        fig2, ax2 = plt.subplots()
        ax2.bar(words_plot, counts_plot, color='skyblue')
        plt.xticks(rotation=45)
        st.subheader("Top 10 palabras más frecuentes")
        st.pyplot(fig2)

        # 4. Clasificación de sentimientos
        st.header("4. Análisis de sentimientos")
        def get_sentiment(text):
            analysis = TextBlob(text)
            if analysis.sentiment.polarity > 0.1:
                return 'Positivo'
            elif analysis.sentiment.polarity < -0.1:
                return 'Negativo'
            else:
                return 'Neutro'
        df['sentimiento'] = df['opinion'].apply(get_sentiment)
        st.dataframe(df[['opinion', 'sentimiento']])

        # Gráfico de porcentaje por clase
        st.subheader("Distribución de sentimientos")
        sentiment_counts = df['sentimiento'].value_counts()
        fig3 = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Porcentaje de opiniones por sentimiento"
        )
        st.plotly_chart(fig3)

        # 5. Funcionalidad extra: Nuevo comentario
        st.header("5. Analiza un comentario nuevo")
        nuevo_comentario = st.text_area("Escribe un nuevo comentario aquí")
        if st.button("Analizar comentario"):
            if nuevo_comentario.strip() == "":
                st.warning("Por favor escribe un comentario.")
            else:
                sentimiento_nuevo = get_sentiment(nuevo_comentario)
                try:
                    resumen_nuevo = summarize(nuevo_comentario, word_count=20)
                    if not resumen_nuevo:
                        resumen_nuevo = "Comentario muy corto para resumir."
                except:
                    resumen_nuevo = "Comentario muy corto para resumir."
                st.write(f"**Sentimiento:** {sentimiento_nuevo}")
                st.write(f"**Resumen:** {resumen_nuevo}")

        # 6. Funcionalidad extra: Resumen de los 20 comentarios
        st.header("6. Resumen de las opiniones cargadas")
        if st.button("Mostrar resumen de todas las opiniones"):
            try:
                resumen_total = summarize(' '.join(df['opinion']), word_count=50)
                if not resumen_total:
                    resumen_total = "Opiniones muy cortas para resumir."
            except:
                resumen_total = "Opiniones muy cortas para resumir."
            st.write(resumen_total)
