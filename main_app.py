import streamlit as st
from groq import Groq
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import colorsys

# Configuración de página
st.set_page_config(
    page_title="Plataforma LLM & Embeddings",
    page_icon="🚀",
    layout="wide"
)

# Función para generar colores pastel basados en el ID del token
def get_pastel_color(token_id):
    hue = (token_id * 137.508) % 360 / 360.0  # Ángulo áureo para distribuir colores
    r, g, b = colorsys.hls_to_rgb(hue, 0.8, 0.6)
    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

# --- OBTENCIÓN DE API KEY (Secrets de Streamlit o Sidebar) ---
api_key = None

# Intenta obtener la API Key desde los Secretos de Streamlit Cloud
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

# --- SIDEBAR: Configuración y Parámetros ---
with st.sidebar:
    st.title("⚙️ Configuración")
    
    # Si no hay API Key en secrets, la solicita en la barra lateral
    if not api_key:
        api_key = st.text_input("Ingresa tu API Key de Groq", type="password")
        if not api_key:
            st.info("💡 Puedes ingresar tu API Key aquí o configurarla en Secrets de Streamlit.")
    else:
        st.success("🔑 API Key detectada desde Secrets.")
    
    st.markdown("---")
    st.markdown("### Parámetros de Generación")
    groq_model = st.selectbox(
        "Modelo de Groq", 
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
    )
    temperature = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 4000, 1024, 100)
    top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.1)

# --- CUERPO PRINCIPAL (TABS) ---
st.title("🚀 Playground de Modelos de Lenguaje & Embeddings")

tab_gen, tab_token, tab_embed = st.tabs([
    "📝 Generación de Texto", 
    "🔠 Tokenización y Colores", 
    "📐 Similitud de Embeddings"
])

# ----------------- TAB 1: GENERACIÓN (GROQ) -----------------
with tab_gen:
    st.header("Generación de texto usando Groq")
    system_prompt = st.text_area("System Prompt (Opcional)", "Eres un asistente útil y amable.")
    user_prompt = st.text_area("Mensaje del Usuario", "¿Cuáles son las ventajas de la inteligencia artificial en la ciencia?")
    
    if st.button("Generar Texto", type="primary"):
        if not api_key:
            st.error("Por favor, ingresa tu API Key de Groq en la barra lateral o en los Secrets de Streamlit.")
        else:
            try:
                client = Groq(api_key=api_key)
                with st.spinner(f"Generando con {groq_model}..."):
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=groq_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                    st.success("Respuesta generada:")
                    st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error en la API de Groq: {e}")

# ----------------- TAB 2: TOKENIZACIÓN VISUAL -----------------
with tab_token:
    st.header("Explorador de Tokens e IDs")
    st.write("Visualiza cómo los modelos descomponen el texto en Tokens con sus respectivos IDs.")
    
    text_to_tokenize = st.text_area("Texto a tokenizar", "El procesamiento de lenguaje natural es increíble.")
    
    if text_to_tokenize:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text_to_tokenize)
        
        st.write(f"**Total de tokens:** {len(tokens)}")
        
        html_content = "<div style='line-height: 2.5;'>"
        for t_id in tokens:
            token_str = enc.decode([t_id]).replace('<', '&lt;').replace('>', '&gt;')
            color = get_pastel_color(t_id)
            
            html_content += f"""
            <span style='background-color: {color}; padding: 4px 8px; border-radius: 4px; margin: 2px; 
                         font-family: monospace; color: black; border: 1px solid #ccc;'
                  title='Token ID: {t_id}'>
                {token_str} <sub style='color:#555; font-size:0.6em;'>{t_id}</sub>
            </span>
            """
        html_content += "</div>"
        
        st.markdown(html_content, unsafe_allow_html=True)

# ----------------- TAB 3: SIMILITUD COSENO (EMBEDDINGS) -----------------
with tab_embed:
    st.header("Comparación Semántica de Frases (Embeddings)")
    
    embed_model_name = st.selectbox(
        "Selecciona el Modelo de Embeddings",
        [
            "paraphrase-multilingual-MiniLM-L12-v2",
            "all-MiniLM-L6-v2"
        ]
    )
    
    @st.cache_resource
    def load_model(model_name):
        return SentenceTransformer(model_name)
    
    with st.spinner("Cargando modelo de embeddings..."):
        embedder = load_model(embed_model_name)
    
    col1, col2 = st.columns(2)
    with col1:
        frase1 = st.text_area("Frase 1", "El gato duerme en el sofá")
    with col2:
        frase2 = st.text_area("Frase 2", "Un felino está descansando sobre el sillón")
        
    if st.button("Calcular Similitud Coseno"):
        if frase1 and frase2:
            emb1 = embedder.encode([frase1])
            emb2 = embedder.encode([frase2])
            
            similitud = cosine_similarity(emb1, emb2)[0][0]
            porcentaje = similitud * 100
            
            st.markdown("### Resultado:")
            st.progress(float(similitud) if similitud > 0 else 0.0)
            
            if porcentaje > 80:
                st.success(f"**{porcentaje:.2f}% de similitud.** Las frases tienen un significado prácticamente idéntico.")
            elif porcentaje > 50:
                st.warning(f"**{porcentaje:.2f}% de similitud.** Las frases están estrechamente relacionadas.")
            else:
                st.error(f"**{porcentaje:.2f}% de similitud.** Las frases abordan temas diferentes.")
        else:
            st.error("Por favor, ingresa ambas frases.")