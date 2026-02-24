"""
Trustworthy AI Explainer - Streamlit Dashboard
Module 15 Team Project - Cimaprompter

Multi-page interactive LLM tutoring system for programming logic with
explainability, feedback mechanisms, and performance monitoring.

Team: Cimaprompter (UABC)
Members: Miguel Ángel González Mandujano, Monica Valenzuela Delgado,
         Karina Caro Corrales, Juan Francisco Flores Resendiz
"""

import streamlit as st
import os
import re
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
from datetime import datetime

# LangChain + Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Explainability
from explainability_module import CombinedExplainer, create_lime_chart_data, create_shap_chart_data

# RAG System (Module 16)
from rag_system import RAGSystem, initialize_rag_system

# ============================================================================
# API KEY — supports Streamlit Cloud (st.secrets) and local (.env / export)
# ============================================================================

def _get_api_key() -> str:
    """Return GOOGLE_API_KEY from Streamlit secrets (cloud) or env var (local)."""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_API_KEY", "")

# ============================================================================
# PAGE CONFIGURATION (Must be first Streamlit command)
# ============================================================================

st.set_page_config(
    page_title="Cimaprompter - Trustworthy AI Explainer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pedagogical system prompt (Spanish - for programming tutoring)
DEFAULT_SYSTEM_PROMPT = """Eres un tutor de lógica de programación para estudiantes de preparatoria y universidad.

Tu objetivo es GUIAR al estudiante, no darle la solución directamente.

Cuando un estudiante te pregunte sobre código o lógica de programación:
1. NO le des el código completo de inmediato
2. Explica la LÓGICA detrás de las estructuras (bucles while, condicionales if-else, tipos de datos)
3. Haz preguntas guía para que el estudiante piense
4. Proporciona ejemplos conceptuales primero
5. Si insiste, da pseudocódigo antes que código real

Sé paciente, claro y pedagógico."""

# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """
    Load Google Gemini model via LangChain (cached as resource).
    Use @st.cache_resource for expensive objects like models.
    """
    api_key = _get_api_key()
    if not api_key:
        st.error(
            "❌ GOOGLE_API_KEY not found!\n\n"
            "**Streamlit Cloud:** Add it in App Settings → Secrets:\n"
            "```\nGOOGLE_API_KEY = 'your-key-here'\n```\n\n"
            "**Local:** Create a `.env` file with `GOOGLE_API_KEY=your-key-here`"
        )
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature
    )
    return llm

@st.cache_resource
def load_explainer() -> CombinedExplainer:
    """
    Load SHAP/LIME explainer (cached as resource).
    This explainer uses a local model for demonstrating explainability.
    """
    return CombinedExplainer()

@st.cache_resource
def load_rag_system() -> RAGSystem:
    """
    Load RAG system (cached as resource).
    This loads the vectorstore with code style guides.
    """
    try:
        rag = initialize_rag_system(
            documents_path="./code styles",
            persist_directory="./chroma_db",
            rebuild=False,  # Don't rebuild, load existing
            embedding_model="gemini",
            google_api_key=_get_api_key()
        )
        return rag
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_feedback_data() -> pd.DataFrame:
    """
    Load feedback data (cached for 1 hour).
    Use @st.cache_data for data that can be serialized.
    """
    if 'feedback_db' in st.session_state and st.session_state.feedback_db:
        return pd.DataFrame(st.session_state.feedback_db)

    return pd.DataFrame({
        'timestamp': [],
        'message': [],
        'response': [],
        'rating': [],
        'comment': []
    })

@st.cache_data
def compute_explanation(_input_text: str, _response: str) -> Dict:
    """
    Compute SHAP/LIME explainability for the input question.
    Prefix with _ to exclude from cache key.

    Note: This uses a local classifier to demonstrate explainability techniques.
    The classifier analyzes the question type (loops, conditionals, variables, etc.)
    """
    explainer = load_explainer()

    # Get combined explanation (LIME + SHAP)
    explanation = explainer.explain_question(_input_text)

    # Get visualization data
    viz_data = explainer.get_visualization_data(explanation)

    # Extract basic metrics
    input_words = len(_input_text.split())
    response_words = len(_response.split())

    return {
        'input_tokens': input_words,
        'response_tokens': response_words,
        'confidence': explanation['summary']['confidence'],
        'category': explanation['summary']['category'],
        'lime_features': explanation['lime']['feature_weights'][:10],
        'shap_features': explanation['shap']['feature_importance'][:10],
        'highlighted_tokens': explanation['lime']['highlighted_tokens'],
        'probabilities': explanation['lime']['probabilities'],
        'viz_data': viz_data,
        'explanation_lime': explanation['lime']['explanation_text'],
        'explanation_shap': explanation['shap']['explanation_text']
    }

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""

    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Feedback database (in-memory for demo)
    if 'feedback_db' not in st.session_state:
        st.session_state.feedback_db = []

    # User preferences
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {
            'temperature': 0.7,
            'max_tokens': 500,
            'system_prompt': DEFAULT_SYSTEM_PROMPT
        }

    # Current explanation
    if 'current_explanation' not in st.session_state:
        st.session_state.current_explanation = None

    # Performance metrics
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'total_messages': 0,
            'avg_response_time': 0,
            'total_feedback': 0
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_response(message: str, temperature: float = 0.7) -> tuple:
    """
    Generate LLM response using Google Gemini with RAG and compute explanation.

    Args:
        message: User's message
        temperature: LLM temperature parameter

    Returns:
        Tuple of (response, explanation, response_time, sources)
    """
    start_time = time.time()

    try:
        # Load cached model and RAG system
        llm = load_llm(temperature)
        rag = load_rag_system()

        # Retrieve relevant context from RAG
        retrieved_sources = []
        context_text = ""

        if rag is not None:
            try:
                with st.spinner("🔍 Buscando en documentos..."):
                    sources = rag.retrieve(message, k=3)
                retrieved_sources = sources

                if sources:
                    context_text = "\n\n---CONTEXTO DE DOCUMENTOS---\n"
                    for i, source in enumerate(sources, 1):
                        context_text += f"\n[Fuente {i}: {source['source']}, Página {source['page']}]\n"
                        context_text += f"{source['content']}\n"
                    context_text += "\n---FIN DEL CONTEXTO---\n"
            except Exception as e:
                st.warning(f"RAG retrieval falló: {str(e)}")

        # Build enhanced system prompt with RAG context
        base_system_prompt = st.session_state.preferences['system_prompt']

        if context_text:
            enhanced_prompt = base_system_prompt + "\n\n" + context_text + """

INSTRUCCIONES CRÍTICAS — DEBES SEGUIRLAS SIEMPRE:
1. Responde ÚNICAMENTE usando la información del CONTEXTO DE DOCUMENTOS proporcionado arriba.
2. Cita SIEMPRE tus fuentes en el texto de la respuesta usando el formato [Fuente 1], [Fuente 2], etc.
3. NO uses tu conocimiento general ni información externa a los documentos.
4. Si la pregunta no puede responderse con el contexto dado, responde exactamente:
   "No tengo información sobre ese tema en mis documentos. Puedo ayudarte con preguntas
    sobre las guías de estilo de Google para C++, Objective-C y Markdown."
5. Nunca inventes ni supongas información que no esté en el contexto."""
        else:
            # No relevant context was found (all chunks were below relevance threshold)
            enhanced_prompt = base_system_prompt + """

INSTRUCCIÓN CRÍTICA: No se encontró información relevante en los documentos para esta pregunta.
Responde exactamente lo siguiente, sin agregar nada más:
"No tengo información sobre ese tema en mis documentos. Puedo ayudarte con preguntas
 sobre las guías de estilo de Google para C++, Objective-C y Markdown."
NO uses tu conocimiento general."""

        messages = [SystemMessage(content=enhanced_prompt)]

        # Add conversation history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add current message
        messages.append(HumanMessage(content=message))

        # Generate response
        with st.spinner("✍️ Generando respuesta..."):
            response = llm.invoke(messages)
            response_text = response.content

        # Compute explanation
        explanation = compute_explanation(message, response_text)

        response_time = time.time() - start_time

        # Update metrics
        st.session_state.metrics['total_messages'] += 1
        st.session_state.metrics['avg_response_time'] = (
            (st.session_state.metrics['avg_response_time'] *
             (st.session_state.metrics['total_messages'] - 1) + response_time) /
            st.session_state.metrics['total_messages']
        )

        # Track citation rate — check if LLM actually cited sources in the response text,
        # not just whether RAG retrieved chunks. This gives an accurate metric.
        citation_patterns = [r'\[Fuente\s*\d+\]', r'según la guía', r'de acuerdo a la guía',
                             r'la guía (de|del)', r'\[Source\s*\d+\]']
        has_inline_citation = any(
            re.search(pattern, response_text, re.IGNORECASE)
            for pattern in citation_patterns
        )
        if 'total_with_sources' not in st.session_state.metrics:
            st.session_state.metrics['total_with_sources'] = 0
        if has_inline_citation:
            st.session_state.metrics['total_with_sources'] += 1

        return response_text, explanation, response_time, retrieved_sources

    except Exception as e:
        error_msg = f"Error al generar respuesta: {str(e)}"
        st.error(error_msg)
        return error_msg, {}, 0, []

def save_feedback(message: str, response: str, rating: str, comment: str):
    """Save user feedback to session state."""
    feedback_entry = {
        'timestamp': datetime.now(),
        'message': message,
        'response': response,
        'rating': rating,
        'comment': comment
    }

    st.session_state.feedback_db.append(feedback_entry)
    st.session_state.metrics['total_feedback'] += 1

    # Clear cache to reload feedback data
    load_feedback_data.clear()

# ============================================================================
# PAGE: CHAT INTERFACE
# ============================================================================

def page_chat():
    """Main chat interface page."""
    st.title("🎓 Asistente de Guías de Estilo de Código")
    st.markdown(
        "Consulta las guías de estilo de Google para **C++**, **Objective-C** y **Markdown**. "
        "Las respuestas se generan únicamente a partir de los documentos indexados."
    )

    with st.expander("💡 ¿Qué puedo preguntarte? (ejemplos)", expanded=False):
        st.markdown("""
**Preguntas fáciles — respuesta directa:**
- ¿Cuál es el estándar de indentación recomendado en C++?
- ¿Cómo se deben nombrar las constantes en Objective-C?
- ¿Qué reglas establece la guía de Markdown para los encabezados?

**Preguntas moderadas — combinan varias fuentes:**
- ¿Qué diferencias existen entre las convenciones de nombres en C++ y Objective-C?
- ¿Cómo manejan los comentarios las tres guías de estilo?
- ¿Qué dice la guía de Markdown sobre el uso de listas anidadas?

**Nota:** El asistente solo responde sobre los documentos disponibles.
Para temas fuera de este alcance indicará que no tiene información.
        """)

    # ========================================================================
    # METRICS PANEL (Module 16)
    # ========================================================================
    st.markdown("### 📊 Métricas del Sistema")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_time = st.session_state.metrics.get('avg_response_time', 0)
        st.metric(
            label="⏱️ Tiempo Promedio",
            value=f"{avg_time:.2f}s",
            delta="-0.5s" if avg_time < 3 else None,
            help="Tiempo promedio de respuesta. Meta: < 3 segundos"
        )

    with col2:
        total_queries = st.session_state.metrics.get('total_messages', 0)
        st.metric(
            label="💬 Total Queries",
            value=total_queries,
            help="Número total de preguntas realizadas"
        )

    with col3:
        total_feedback = st.session_state.metrics.get('total_feedback', 0)
        positive_feedback = len([f for f in st.session_state.feedback_db if '👍' in f.get('rating', '')])

        if total_feedback > 0:
            satisfaction = (positive_feedback / total_feedback) * 100
            st.metric(
                label="😊 Satisfacción",
                value=f"{satisfaction:.1f}%",
                delta="Bueno" if satisfaction >= 60 else "Mejorar",
                help="Porcentaje de feedback positivo. Meta: > 60%"
            )
        else:
            st.metric(
                label="😊 Satisfacción",
                value="N/A",
                help="Sin feedback aún"
            )

    with col4:
        total_with_sources = st.session_state.metrics.get('total_with_sources', 0)

        if total_queries > 0:
            citation_rate = (total_with_sources / total_queries) * 100
            st.metric(
                label="📚 Citación",
                value=f"{citation_rate:.1f}%",
                delta="Excelente" if citation_rate >= 80 else "Normal",
                help="Porcentaje de respuestas con fuentes. Meta: > 80%"
            )
        else:
            st.metric(
                label="📚 Citación",
                value="N/A",
                help="Sin queries aún"
            )

    st.divider()

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Configuración")

        temperature = st.slider(
            "Temperatura (Creatividad)",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.preferences['temperature'],
            step=0.1,
            help="Valores más altos hacen la salida más creativa"
        )
        st.session_state.preferences['temperature'] = temperature

        max_tokens = st.slider(
            "Tokens Máximos",
            min_value=50,
            max_value=2000,
            value=st.session_state.preferences['max_tokens'],
            step=50
        )
        st.session_state.preferences['max_tokens'] = max_tokens

        with st.expander("Prompt del Sistema"):
            system_prompt = st.text_area(
                "Instrucciones del Sistema",
                value=st.session_state.preferences['system_prompt'],
                height=200
            )
            st.session_state.preferences['system_prompt'] = system_prompt

        st.divider()

        if st.button("🗑️ Limpiar Historial", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_explanation = None
            st.rerun()

    # Main chat area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Conversación")

        # Display chat history
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Escribe tu pregunta sobre programación aquí..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Generate response with RAG (spinners shown inside generate_response)
            response, explanation, response_time, sources = generate_response(
                prompt,
                temperature
            )

            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.current_explanation = {
                'input': prompt,
                'output': response,
                'details': explanation,
                'response_time': response_time,
                'sources': sources  # Store retrieved sources
            }

            # Display assistant message
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)

            st.rerun()

    with col2:
        st.subheader("🔍 Explicabilidad SHAP/LIME")

        if st.session_state.current_explanation:
            exp = st.session_state.current_explanation
            details = exp['details']

            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confianza", f"{details.get('confidence', 0):.2f}")
            with col_b:
                st.metric("Tiempo", f"{exp['response_time']:.2f}s")

            st.divider()

            # RAG Sources (Module 16)
            sources = exp.get('sources', [])
            if sources:
                st.markdown("**📚 Fuentes Recuperadas (RAG):**")
                for i, source in enumerate(sources, 1):
                    with st.expander(f"📄 {source['source']} - Página {source['page']}"):
                        st.caption(f"**Relevancia:** {source['score']:.4f}")
                        st.text(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
            else:
                st.markdown("**📚 Fuentes Recuperadas:**")
                st.info("No se recuperaron fuentes del knowledge base")

            st.divider()

            # Question category
            st.markdown("**📋 Categoría Detectada:**")
            st.info(f"**{details.get('category', 'N/A')}**")

            st.divider()

            # LIME Top Features
            st.markdown("**🔬 LIME - Top 5 Palabras Influyentes:**")
            lime_features = details.get('lime_features', [])[:5]
            for word, weight in lime_features:
                # Clean word
                clean_word = word.strip('<>=')
                direction = "↑" if weight > 0 else "↓"
                color = "green" if weight > 0 else "red"
                st.markdown(f":{color}[{direction} **{clean_word}**: {weight:.3f}]")

            st.divider()

            # SHAP Top Features
            st.markdown("**⚡ SHAP - Top 5 Importancia Global:**")
            shap_features = details.get('shap_features', [])[:5]
            for word, importance in shap_features:
                st.markdown(f"- **{word}**: {importance:.3f}")

            st.divider()

            # Feedback section
            st.markdown("**📊 Proporcionar Retroalimentación:**")

            rating = st.radio(
                "Califica esta respuesta:",
                options=["👍 Útil", "👎 No útil"],
                key=f"rating_{len(st.session_state.messages)}"
            )

            comment = st.text_area(
                "Comentario opcional:",
                placeholder="¿Qué fue bueno o se puede mejorar?",
                key=f"comment_{len(st.session_state.messages)}"
            )

            if st.button("📝 Enviar Retroalimentación", use_container_width=True):
                save_feedback(
                    exp['input'],
                    exp['output'],
                    rating,
                    comment
                )
                st.success("✅ ¡Retroalimentación guardada!")
        else:
            st.info("Envía un mensaje para ver la explicación SHAP/LIME y proporcionar retroalimentación.")

# ============================================================================
# PAGE: EXPLAINABILITY ANALYSIS
# ============================================================================

def page_explainability():
    """Detailed explainability analysis page."""
    st.title("🔍 Análisis de Explicabilidad")
    st.markdown("Análisis profundo de las decisiones del modelo y patrones de comportamiento.")

    if not st.session_state.messages:
        st.info("No hay conversaciones aún. Ve a la página de Chat para comenzar.")
        return

    # Get recent conversations
    conversations = []
    for i in range(0, len(st.session_state.messages), 2):
        if i+1 < len(st.session_state.messages):
            conversations.append({
                'user': st.session_state.messages[i]['content'],
                'assistant': st.session_state.messages[i+1]['content']
            })

    if not conversations:
        st.warning("No se encontraron conversaciones completas.")
        return

    # Select conversation to analyze
    selected_idx = st.selectbox(
        "Selecciona conversación para analizar:",
        range(len(conversations)),
        format_func=lambda i: f"Conv {i+1}: {conversations[i]['user'][:50]}..."
    )

    selected_conv = conversations[selected_idx]

    # Display conversation
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Entrada del Usuario")
        st.markdown(f"```\n{selected_conv['user']}\n```")

        st.subheader("🤖 Respuesta del Modelo")
        st.markdown(f"```\n{selected_conv['assistant']}\n```")

    with col2:
        st.subheader("📊 Explicabilidad SHAP/LIME")

        # Compute explanation
        exp = compute_explanation(selected_conv['user'], selected_conv['assistant'])

        # Display metrics
        st.metric("Tokens de Entrada", exp['input_tokens'])
        st.metric("Tokens de Respuesta", exp['response_tokens'])
        st.metric("Confianza del Clasificador", f"{exp['confidence']:.0%}")
        st.metric("Categoría Detectada", exp['category'])

        st.divider()

        st.markdown("**📋 Distribución de Probabilidades:**")
        for cat, prob in list(exp['probabilities'].items())[:5]:
            st.progress(prob, text=f"{cat}: {prob:.2%}")

    # Visualization section
    st.divider()
    st.subheader("📈 Visualizaciones SHAP/LIME")

    # LIME visualization
    st.markdown("### 🔬 LIME - Importancia Local de Palabras")
    st.caption(exp['explanation_lime'])

    lime_words = [w[0].strip('<>=') for w in exp['lime_features']]
    lime_weights = [w[1] for w in exp['lime_features']]
    lime_colors = ['green' if w > 0 else 'red' for w in lime_weights]

    fig_lime = go.Figure(data=[
        go.Bar(
            x=lime_weights,
            y=lime_words,
            orientation='h',
            marker_color=lime_colors,
            text=[f"{w:.3f}" for w in lime_weights],
            textposition='outside'
        )
    ])
    fig_lime.update_layout(
        title="LIME Feature Weights",
        xaxis_title="Peso (+ ayuda a predecir, - va en contra)",
        yaxis_title="Palabras",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_lime, use_container_width=True)

    st.divider()

    # SHAP visualization
    st.markdown("### ⚡ SHAP - Importancia Global de Features")
    st.caption(exp['explanation_shap'])

    shap_words = [w[0] for w in exp['shap_features']]
    shap_importance = [w[1] for w in exp['shap_features']]

    fig_shap = go.Figure(data=[
        go.Bar(
            x=shap_importance,
            y=shap_words,
            orientation='h',
            marker_color='lightblue',
            text=[f"{w:.3f}" for w in shap_importance],
            textposition='outside'
        )
    ])
    fig_shap.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="Importancia (drop in confidence si se remueve)",
        yaxis_title="Palabras",
        height=400
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.divider()

    # Highlighted text
    st.markdown("### 🎨 Texto con Palabras Resaltadas (LIME)")
    st.caption("Verde = ayuda a la predicción | Rojo = va en contra")

    # Create highlighted text HTML
    highlighted_html = selected_conv['user']
    for token in exp['highlighted_tokens']:
        word = token['word']
        weight = abs(token['weight'])
        color = token['color']

        intensity = min(weight * 3, 1.0)  # Scale up for visibility
        if color == 'green':
            bg_color = f'rgba(0, 255, 0, {intensity * 0.4})'
        else:
            bg_color = f'rgba(255, 0, 0, {intensity * 0.4})'

        highlighted = f'<span style="background-color: {bg_color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{word}</span>'
        # Use word boundaries to avoid partial replacements
        import re
        highlighted_html = re.sub(rf'\b{re.escape(word)}\b', highlighted, highlighted_html, count=1)

    st.markdown(highlighted_html, unsafe_allow_html=True)

    st.success("✅ **Implementación completa**: SHAP y LIME funcionando con modelo local de demostración.")

# ============================================================================
# PAGE: FEEDBACK DASHBOARD
# ============================================================================

def page_feedback():
    """User feedback and quality monitoring page."""
    st.title("📊 Dashboard de Retroalimentación")
    st.markdown("Monitorea la retroalimentación de usuarios y la calidad de respuestas.")

    # Load feedback data
    feedback_df = load_feedback_data()

    if feedback_df.empty:
        st.info("No se ha recopilado retroalimentación aún. ¡Chatea con el tutor y proporciona retroalimentación!")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Retroalimentación", len(feedback_df))

    with col2:
        positive = len(feedback_df[feedback_df['rating'].str.contains('👍')])
        st.metric("Positiva", positive)

    with col3:
        negative = len(feedback_df[feedback_df['rating'].str.contains('👎')])
        st.metric("Negativa", negative)

    with col4:
        if len(feedback_df) > 0:
            satisfaction = (positive / len(feedback_df)) * 100
            st.metric("Satisfacción", f"{satisfaction:.1f}%")

    st.divider()

    # Feedback visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de Retroalimentación")

        rating_counts = feedback_df['rating'].value_counts()
        fig = px.pie(
            values=rating_counts.values,
            names=rating_counts.index,
            title="Retroalimentación Positiva vs Negativa"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Retroalimentación a lo Largo del Tiempo")

        feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
        daily_counts = feedback_df.groupby('date').size().reset_index(name='count')

        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Conteo de Retroalimentación por Día",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Recent feedback
    st.subheader("Retroalimentación Reciente")

    # Display feedback table
    display_df = feedback_df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['message'] = display_df['message'].str[:50] + '...'
    display_df['response'] = display_df['response'].str[:50] + '...'

    st.dataframe(
        display_df[['timestamp', 'message', 'response', 'rating', 'comment']],
        use_container_width=True,
        hide_index=True
    )

    # Export feedback
    st.divider()
    if st.button("📥 Exportar Datos de Retroalimentación"):
        csv = feedback_df.to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name=f"feedback_cimaprompter_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============================================================================
# PAGE: MONITORING
# ============================================================================

def page_monitoring():
    """System monitoring and performance metrics page."""
    st.title("📈 Monitoreo del Sistema")
    st.markdown("Rastrea el rendimiento de la aplicación y métricas de uso.")

    # Metrics overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total de Mensajes",
            st.session_state.metrics['total_messages']
        )

    with col2:
        st.metric(
            "Tiempo Promedio de Respuesta",
            f"{st.session_state.metrics['avg_response_time']:.2f}s"
        )

    with col3:
        st.metric(
            "Total de Retroalimentación",
            st.session_state.metrics['total_feedback']
        )

    st.divider()

    # Cache status
    st.subheader("Estado de Caché")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Caché del Modelo**")
        st.success("✅ Modelo Gemini cargado y en caché")
        st.markdown("**Caché de Datos de Retroalimentación**")
        st.info("ℹ️ TTL: 1 hora")

    with col2:
        if st.button("🔄 Limpiar Todos los Cachés"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("¡Cachés limpiados!")
            st.rerun()

    st.divider()

    # Session state inspection
    with st.expander("🔍 Estado de Sesión (Debug)"):
        st.json({
            'messages_count': len(st.session_state.messages),
            'feedback_count': len(st.session_state.feedback_db),
            'preferences': st.session_state.preferences,
            'metrics': st.session_state.metrics
        })

    st.divider()

    # System recommendations
    st.subheader("Recomendaciones de Optimización")

    if st.session_state.metrics['avg_response_time'] > 2.0:
        st.warning("⚠️ El tiempo promedio de respuesta es alto. Considera implementar respuestas en streaming.")
    else:
        st.success("✅ Los tiempos de respuesta están dentro del rango aceptable.")

    if st.session_state.metrics['total_messages'] > 0:
        feedback_rate = st.session_state.metrics['total_feedback'] / st.session_state.metrics['total_messages']
        if feedback_rate < 0.3:
            st.info("ℹ️ La tasa de retroalimentación es baja. Considera hacer la retroalimentación más prominente en la UI.")
        else:
            st.success("✅ Buena tasa de participación en retroalimentación.")

# ============================================================================
# PAGE: DOCUMENTATION
# ============================================================================

def page_documentation():
    """Documentation and team information page."""
    st.title("📚 Documentación")

    tab1, tab2, tab3 = st.tabs(["Acerca de", "Técnico", "Equipo"])

    with tab1:
        st.markdown("""
        ## Acerca de Esta Aplicación

        **Cimaprompter - Trustworthy AI Explainer Dashboard**

        El Trustworthy AI Explainer Dashboard de Cimaprompter es una interfaz integral para
        interactuar con un tutor de IA para lógica de programación, manteniendo transparencia
        y responsabilidad.

        ### Características

        - **💬 Chat Interactivo**: IA conversacional con memoria de contexto
        - **🔍 Explicabilidad**: Análisis en tiempo real de las decisiones del modelo
        - **📊 Sistema de Retroalimentación**: Recopilación de calificaciones y comentarios de usuarios
        - **📈 Monitoreo**: Métricas de rendimiento y análisis de uso
        - **📄 Dashboard Multi-página**: Interfaz organizada para diferentes tareas

        ### Cómo Usar

        1. **Chat**: Ve a la página de Chat para interactuar con el tutor de IA
        2. **Revisar**: Verifica las explicaciones para cada respuesta
        3. **Retroalimentación**: Califica las respuestas para ayudar a mejorar el sistema
        4. **Analizar**: Usa la página de Explicabilidad para análisis profundo
        5. **Monitorear**: Rastrea el rendimiento del sistema en la página de Monitoreo

        ### Objetivo Pedagógico

        Este tutor está diseñado para **guiar** a estudiantes de preparatoria y universidad
        en el aprendizaje de lógica de programación. No proporciona soluciones directas,
        sino que explica conceptos y razonamiento para fomentar el aprendizaje activo.
        """)

    with tab2:
        st.markdown("""
        ## Documentación Técnica

        ### Arquitectura

        ```
        Frontend: Streamlit Multi-página
        Backend: Google Gemini (vía LangChain)
        Explicabilidad: SHAP/LIME (próximamente)
        Estado: st.session_state
        Caché: @st.cache_resource, @st.cache_data
        ```

        ### Stack Tecnológico

        - `streamlit>=1.28.0` - Framework de dashboard
        - `langchain==0.3.0` - Orquestación de LLM
        - `langchain-google-genai` - Integración con Gemini
        - `google-generativeai` - API de Google Gemini
        - `pandas<3.0.0`, `numpy<2.0.0` - Procesamiento de datos
        - `plotly` - Visualizaciones interactivas
        - `shap`, `lime` - Explicabilidad (próximamente)

        ### Gestión de Estado

        Esta aplicación usa `st.session_state` para persistir:
        - Historial de conversación
        - Preferencias del usuario
        - Datos de retroalimentación
        - Métricas de rendimiento

        ### Estrategia de Caché

        - `@st.cache_resource`: Carga del modelo (costoso, no serializado)
        - `@st.cache_data`: Carga de datos (serializable, con TTL)

        ### Despliegue

        **Local:**
        ```bash
        # Activar entorno conda
        conda activate cimaprompter

        # Ejecutar aplicación
        streamlit run Cimaprompter_Module15_StreamlitApp.py
        ```

        **Streamlit Community Cloud:**
        ```bash
        1. Push a GitHub
        2. Conecta el repo en Streamlit Cloud
        3. Agrega GOOGLE_API_KEY en Secrets
        4. Despliega
        ```

        **Docker:**
        ```dockerfile
        FROM python:3.11-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install -r requirements.txt
        COPY Cimaprompter_Module15_StreamlitApp.py .
        EXPOSE 8501
        CMD ["streamlit", "run", "Cimaprompter_Module15_StreamlitApp.py"]
        ```

        ### Variables de Entorno

        ```bash
        GOOGLE_API_KEY=your-gemini-api-key-here
        ```

        ### Configuración de Secrets (Streamlit Cloud)

        ```toml
        # .streamlit/secrets.toml
        GOOGLE_API_KEY = "your-gemini-api-key-here"
        ```
        """)

    with tab3:
        st.markdown("""
        ## Información del Equipo

        **Equipo**: Cimaprompter

        **Institución**: Universidad Autónoma de Baja California (UABC)

        **Integrantes del Equipo**:
        - Miguel Ángel González Mandujano
        - Monica Valenzuela Delgado
        - Karina Caro Corrales
        - Juan Francisco Flores Resendiz

        **Módulo**: 15 - App Prototyping with Streamlit

        **Proyecto**: Trustworthy AI Explainer Dashboard

        **Objetivo**: Desarrollar un sistema de tutoría de IA explicable para la enseñanza
        de lógica de programación a estudiantes de preparatoria y universidad.

        ### Contribuciones del Equipo

        - **Prototipo Gradio**: Sistema de chat interactivo con Google Gemini
        - **Dashboard Streamlit**: Aplicación multi-página con análisis y monitoreo
        - **Integración de Explicabilidad**: Framework para análisis SHAP/LIME
        - **Sistema de Retroalimentación**: Recopilación y análisis de feedback de usuarios

        ### Tecnologías Utilizadas

        - Google Gemini 2.5 Flash (LLM)
        - LangChain 0.3.0 (Orquestación)
        - Gradio 4.0+ (Prototipo)
        - Streamlit 1.28+ (Dashboard)
        - Plotly (Visualizaciones)
        - SHAP/LIME (Explicabilidad)

        ### Contacto

        Para preguntas sobre el proyecto, contacta al equipo Cimaprompter.
        """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""

    # Initialize session state
    initialize_session_state()

    # Sidebar navigation
    with st.sidebar:
        st.title("🎓 Cimaprompter")
        st.markdown("*Trustworthy AI Explainer*")
        st.markdown("---")

        page = st.radio(
            "Navegación",
            ["💬 Chat", "🔍 Explicabilidad", "📊 Retroalimentación", "📈 Monitoreo", "📚 Documentación"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.caption(f"Sesión: {len(st.session_state.messages)} mensajes")
        st.caption(f"Retroalimentación: {len(st.session_state.feedback_db)} entradas")

    # Route to selected page
    if page == "💬 Chat":
        page_chat()
    elif page == "🔍 Explicabilidad":
        page_explainability()
    elif page == "📊 Retroalimentación":
        page_feedback()
    elif page == "📈 Monitoreo":
        page_monitoring()
    elif page == "📚 Documentación":
        page_documentation()

if __name__ == "__main__":
    main()
