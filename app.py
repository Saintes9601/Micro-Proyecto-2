"""
Aplicacion Streamlit para clasificar textos segun los Objetivos de Desarrollo Sostenible (ODS).
Utiliza el pipeline entrenado en el notebook microproyecto2.ipynb.
"""
import streamlit as st
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
import nltk

nltk.download('stopwords', quiet=True)

# Constantes de preprocesamiento (misma definicion que en el notebook)
SPANISH_STOP_WORDS = frozenset(stopwords.words('spanish'))
SPANISH_STEMMER_CACHE = {}

def _stem_word(word):
    if word not in SPANISH_STEMMER_CACHE:
        SPANISH_STEMMER_CACHE[word] = SnowballStemmer('spanish').stem(word)
    return SPANISH_STEMMER_CACHE[word]

# Clases necesarias para que joblib pueda deserializar el pipeline guardado en el modelo.
# Deben tener la misma definicion que en el notebook.
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess(text) for text in X]

    @staticmethod
    def _preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
        tokens = text.split()
        tokens = [_stem_word(word) for word in tokens
                  if word not in SPANISH_STOP_WORDS and len(word) > 2]
        return ' '.join(tokens)


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Wrapper sklearn-compatible de Word2Vec para usar en Pipeline."""
    def __init__(self, vector_size=200, window=5, min_count=3, epochs=20, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y=None):
        tokenized = [text.split() for text in X]
        self.model_ = Word2Vec(
            sentences=tokenized, vector_size=self.vector_size,
            window=self.window, min_count=self.min_count,
            workers=4, seed=self.seed, epochs=self.epochs
        )
        return self

    def transform(self, X):
        tokenized = [text.split() for text in X]
        vectors = []
        for tokens in tokenized:
            word_vecs = [self.model_.wv[w] for w in tokens if w in self.model_.wv]
            if len(word_vecs) == 0:
                vectors.append(np.zeros(self.vector_size))
            else:
                vectors.append(np.mean(word_vecs, axis=0))
        return np.array(vectors)


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

pipeline = artefactos['pipeline']

# --- Interfaz ---
st.title("Clasificador de Textos - Objetivos de Desarrollo Sostenible (ODS)")
st.markdown("""
Esta aplicacion clasifica textos segun los **Objetivos de Desarrollo Sostenible (ODS)**
de las Naciones Unidas, utilizando tecnicas de procesamiento de lenguaje natural y machine learning.

**Instrucciones:** Ingresa un texto relacionado con tematicas de desarrollo sostenible
y el modelo predecira a cual ODS corresponde.
""")

st.divider()

texto_input = st.text_area(
    "Ingresa el texto a clasificar:",
    height=200,
    placeholder="Escribe o pega aqui un texto relacionado con desarrollo sostenible..."
)

if st.button("Clasificar", type="primary"):
    if texto_input.strip():
        with st.spinner("Procesando texto..."):
            ods_predicho = pipeline.predict([texto_input])[0]
            nombre_ods = artefactos['ods_nombres'].get(ods_predicho, "Desconocido")

        st.success(f"**ODS {ods_predicho}: {nombre_ods}**")
        st.markdown(f"""
        **Detalles del modelo:**
        - Pipeline: `{artefactos.get('representación', 'N/A')}`
        - Hiperparametros: `{artefactos.get('best_params', 'N/A')}`
        """)
    else:
        st.warning("Por favor, ingresa un texto para clasificar.")

st.divider()
st.caption("Micro Proyecto 2 — Machine Learning No Supervisado — Universidad de los Andes")
