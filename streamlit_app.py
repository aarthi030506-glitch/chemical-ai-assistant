import streamlit as st
import os
import math

from dotenv import load_dotenv
from google import genai

from tools import (
    calculate_reynolds_number,
    calculate_ideal_gas_volume,
    calculate_heat_duty
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chemical Engineering AI Assistant",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.75;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.3);
    margin-bottom: 15px;
}

.answer-box {
    padding: 25px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.3);
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD API
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ Gemini API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🧪 Chemical Engineering AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered learning assistant for Chemical Engineering concepts, '
    'calculations and problem solving.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Assistant")

    st.markdown("""
    ### What can I do?

    📚 **Concepts**
    - Distillation
    - Heat transfer
    - Mass transfer
    - Fluid mechanics
    - Thermodynamics

    🧮 **Calculations**
    - Reynolds number
    - Ideal gas volume
    - Heat duty

    🤖 **AI Assistance**
    - Explain concepts
    - Solve engineering problems
    - Give simple explanations
    """)

    st.divider()

    st.info(
        "💡 Tip: Ask your question with all the required "
        "values when you need a calculation."
    )


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def create_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    dot_product = sum(
        x * y for x, y in zip(a, b)
    )

    magnitude_a = math.sqrt(
        sum(x * x for x in a)
    )

    magnitude_b = math.sqrt(
        sum(y * y for y in b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_knowledge():

    with open(
        "knowledge.txt",
        "r",
        encoding="utf-8"
    ) as file:

        knowledge = file.read()

    chunks = [
        chunk.strip()
        for chunk in knowledge.split("\n\n")
        if chunk.strip()
    ]

    embeddings = []

    for chunk in chunks:

        embedding = create_embedding(chunk)

        embeddings.append(
            (chunk, embedding)
        )

    return embeddings


chunk_embeddings = load_knowledge()


# ============================================================
# TOOL FUNCTIONS
# ============================================================

def reynolds_tool(
    density: float,
    velocity: float,
    diameter: float,
    viscosity: float
) -> float:

    return calculate_reynolds_number(
        density,
        velocity,
        diameter,
        viscosity
    )


def ideal_gas_tool(
    pressure: float,
    moles: float,
    temperature: float
) -> float:

    return calculate_ideal_gas_volume(
        pressure,
        moles,
        temperature
    )


def heat_duty_tool(
    mass_flow_rate: float,
    specific_heat: float,
    temperature_change: float
) -> float:

    return calculate_heat_duty(
        mass_flow_rate,
        specific_heat,
        temperature_change
    )


tools = [
    reynolds_tool,
    ideal_gas_tool,
    heat_duty_tool
]


# ============================================================
# GEMINI CHAT
# ============================================================

chat = client.chats.create(
    model="gemini-3.5-flash",
    config={
        "tools": tools
    }
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader("💬 Ask a Chemical Engineering Question")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <b>📚 Concept</b><br>
    What is distillation?
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <b>🧮 Calculation</b><br>
    Calculate Reynolds number
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <b>🔥 Heat Transfer</b><br>
    What is heat duty?
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# USER INPUT
# ============================================================

question = st.text_input(
    "Enter your question:",
    placeholder="Example: Calculate Reynolds number for density 1000, velocity 2..."
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🤖 Ask AI",
    type="primary",
    use_container_width=True
):

    if question.strip() == "":
        st.warning("⚠️ Please enter a question.")

    else:

        with st.spinner("🔍 Analyzing your question..."):

            # ------------------------------------------------
            # CREATE QUESTION EMBEDDING
            # ------------------------------------------------

            question_embedding = create_embedding(
                question
            )

            # ------------------------------------------------
            # CALCULATE SIMILARITIES
            # ------------------------------------------------

            scores = []

            for chunk, embedding in chunk_embeddings:

                similarity = cosine_similarity(
                    question_embedding,
                    embedding
                )

                scores.append(
                    (similarity, chunk)
                )

            # ------------------------------------------------
            # SORT RESULTS
            # ------------------------------------------------

            scores.sort(
                reverse=True,
                key=lambda x: x[0]
            )

            # ------------------------------------------------
            # TOP 3 KNOWLEDGE CHUNKS
            # ------------------------------------------------

            top_chunks = scores[:3]

            context = "\n\n".join(
                chunk
                for score, chunk in top_chunks
            )

            # ------------------------------------------------
            # AI PROMPT
            # ------------------------------------------------

            prompt = f"""
You are a Chemical Engineering AI Assistant.

Use the retrieved reference information for
conceptual questions.

For numerical engineering calculations,
use the appropriate Python tool.

REFERENCE INFORMATION:

{context}

USER QUESTION:

{question}

Give a clear, accurate and educational answer.

For calculations:
1. Show the formula.
2. Substitute the values.
3. Give the final answer with units.
4. Briefly explain what the result means.

For conceptual questions:
Explain the concept in simple terms and give
a Chemical Engineering example.
"""

            # ------------------------------------------------
            # GEMINI RESPONSE
            # ------------------------------------------------

            response = chat.send_message(
                prompt
            )

        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.markdown(
            '<div class="answer-box">',
            unsafe_allow_html=True
        )

        st.subheader("🤖 Assistant")

        st.write(response.text)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # RETRIEVED KNOWLEDGE
        # ====================================================

        with st.expander("📚 View Retrieved Knowledge"):

            for score, chunk in top_chunks:

                st.write(chunk)

                st.caption(
                    f"Similarity score: {score:.3f}"
                )

                st.divider()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧪 Chemical Engineering AI Assistant | "
    "Built with Python, Gemini API, RAG & Streamlit"
)