"""
Aplicacion Streamlit para clasificar textos segun los Objetivos de Desarrollo Sostenible (ODS).
Utiliza el modelo entrenado en el notebook microproyecto2.ipynb.
"""
import streamlit as st
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('stopwords', quiet=True)

# --- Configuracion de la pagina ---
st.set_page_config(
    page_title="Clasificador ODS",
    page_icon="🌍",
    layout="centered"
)

# --- Cargar modelo ---
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_ods.joblib')

try:
    artefactos = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontro el archivo 'modelo_ods.joblib'. Ejecuta primero el notebook para generar el modelo.")
    st.stop()

# --- Funciones de procesamiento ---
def preprocess_text(text):
    """Aplica el mismo preprocesamiento del pipeline del notebook."""
    stemmer = SnowballStemmer('spanish')
    stop_words = set(stopwords.words('spanish'))
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def predecir(texto):
    """Procesa el texto y genera la prediccion del ODS."""
    texto_procesado = preprocess_text(texto)
    representacion = artefactos['representacion']

    if 'TF-IDF' in representacion:
        vectorizer = artefactos['vectorizer']
        X = vectorizer.transform([texto_procesado])
        if 'reductor_intermedio' in artefactos:
            X = artefactos['reductor_intermedio'].transform(X)
        X = artefactos['reductor'].transform(X)
    else:
        # Word2Vec
        w2v_model = artefactos['w2v_model']
        tokens = texto_procesado.split()
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if len(vectors) == 0:
            doc_vector = np.zeros(w2v_model.wv.vector_size)
        else:
            doc_vector = np.mean(vectors, axis=0)
        X = artefactos['reductor'].transform([doc_vector])

    prediccion = artefactos['clasificador'].predict(X)[0]
    return prediccion

# --- Interfaz ---
st.title("Clasificador de Textos - Objetivos de Desarrollo Sostenible (ODS)")
st.markdown("""
Esta aplicacion clasifica textos en espanol segun los **Objetivos de Desarrollo Sostenible (ODS)**
de las Naciones Unidas, utilizando tecnicas de procesamiento de lenguaje natural y machine learning.

**Instrucciones:** Ingresa un texto en espanol relacionado con tematicas de desarrollo sostenible
y el modelo predecira a cual ODS corresponde.
""")

st.divider()

texto_input = st.text_area(
    "Ingresa el texto a clasificar:",
    height=200,
    placeholder="Escribe o pega aqui un texto en espanol relacionado con desarrollo sostenible..."
)

if st.button("Clasificar", type="primary"):
    if texto_input.strip():
        with st.spinner("Procesando texto..."):
            ods_predicho = predecir(texto_input)
            nombre_ods = artefactos['ods_nombres'].get(ods_predicho, "Desconocido")

        st.success(f"**ODS {ods_predicho}: {nombre_ods}**")
        st.markdown(f"""
        **Detalles del modelo:**
        - Representacion: `{artefactos['representacion']}`
        - Clasificador: `{type(artefactos['clasificador']).__name__}`
        """)
    else:
        st.warning("Por favor, ingresa un texto para clasificar.")

st.divider()
st.caption("Micro Proyecto 2 — Machine Learning No Supervisado — Universidad de los Andes")
