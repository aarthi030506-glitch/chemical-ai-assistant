import streamlit as st
import os
import re

from dotenv import load_dotenv
from google import genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chemical Engineering AI Assistant",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
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
# LOAD GEMINI API KEY
# ============================================================

load_dotenv()


api_key = None


# Check Streamlit secrets first
try:

    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

except Exception:
    pass


# Check .env file
if not api_key:

    api_key = os.getenv("GEMINI_API_KEY")


# Stop if API key is missing
if not api_key:

    st.error("❌ Gemini API key not found.")

    st.info(
        "Add your GEMINI_API_KEY to your .env file."
    )

    st.stop()


# Create Gemini client
client = genai.Client(
    api_key=api_key
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="title">
    🧪 Chemical Engineering AI Assistant
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="subtitle">
    AI-powered Chemical Engineering learning assistant using
    Retrieval-Augmented Generation (RAG), TF-IDF retrieval,
    Cosine Similarity and Gemini AI.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Assistant Features")


    st.markdown("""
### 📚 Chemical Engineering Concepts

- Fluid Mechanics
- Heat Transfer
- Mass Transfer
- Thermodynamics
- Reaction Engineering
- Separation Processes
- Process Control
- Mass and Energy Balances


### 🧠 Local RAG System

- Knowledge Base
- Text Chunking
- TF-IDF Vectorization
- Cosine Similarity
- Top Relevant Chunk Retrieval


### 🧮 Engineering Calculations

- Reynolds Number
- Ideal Gas Law
- Heat Duty


### 🛡️ Domain Protection

The assistant is designed specifically for
Chemical Engineering questions.
""")


    st.divider()


    st.info(
        "💡 For numerical questions, include "
        "all values and units clearly."
    )

# ============================================================
# LOCAL KNOWLEDGE BASE
# ============================================================

@st.cache_resource(show_spinner=False)
def create_local_rag():

    # Read knowledge.txt

    with open(
        "knowledge.txt",
        "r",
        encoding="utf-8"
    ) as file:

        knowledge = file.read()


    # Split knowledge into chunks

    chunks = [

        chunk.strip()

        for chunk in knowledge.split("\n\n")

        if chunk.strip()

    ]


    # Check knowledge

    if not chunks:

        raise ValueError(
            "knowledge.txt is empty."
        )


    # Create local TF-IDF vectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )


    # Create vectors locally
    # NO GEMINI EMBEDDING API IS USED

    chunk_vectors = vectorizer.fit_transform(
        chunks
    )


    return (
        chunks,
        vectorizer,
        chunk_vectors
    )


# Load Local RAG System

try:

    (
        knowledge_chunks,
        vectorizer,
        chunk_vectors
    ) = create_local_rag()


except FileNotFoundError:

    st.error(
        "❌ knowledge.txt file was not found."
    )

    st.info(
        "Make sure knowledge.txt is in the "
        "same folder as streamlit_app.py"
    )

    st.stop()


except Exception as e:

    st.error(
        f"❌ Error loading knowledge base: {e}"
    )

    st.stop()


# ============================================================
# RETRIEVE RELEVANT KNOWLEDGE
# ============================================================

def retrieve_knowledge(
    question,
    top_k=3
):

    # Convert question into TF-IDF vector

    question_vector = vectorizer.transform(
        [question]
    )


    # Calculate cosine similarity

    similarity_scores = cosine_similarity(
        question_vector,
        chunk_vectors
    ).flatten()


    # Sort chunks from highest relevance

    ranked_indices = (
        similarity_scores
        .argsort()[::-1]
    )


    results = []


    # Select top relevant chunks

    for index in ranked_indices[:top_k]:

        results.append({

            "chunk": knowledge_chunks[index],

            "score": float(
                similarity_scores[index]
            )

        })


    return results


# ============================================================
# ENGINEERING CALCULATION FUNCTIONS
# ============================================================

def calculate_reynolds_number(
    density,
    velocity,
    diameter,
    viscosity
):

    return (
        density
        * velocity
        * diameter
        / viscosity
    )


def calculate_ideal_gas_volume(
    pressure,
    moles,
    temperature
):

    R = 8.314

    return (
        moles
        * R
        * temperature
        / pressure
    )


def calculate_heat_duty(
    mass_flow_rate,
    specific_heat,
    temperature_change
):

    return (
        mass_flow_rate
        * specific_heat
        * temperature_change
    )

# ============================================================
# CHEMICAL ENGINEERING CALCULATOR
# ============================================================

st.divider()

st.markdown("## 🧮 Engineering Calculator")

st.caption(
    "Perform essential Chemical Engineering calculations "
    "using built-in engineering formulas."
)


calculation_type = st.selectbox(
    "Select a calculation:",
    [
        "Reynolds Number",
        "Ideal Gas Volume",
        "Heat Duty"
    ]
)


# ============================================================
# REYNOLDS NUMBER
# ============================================================

if calculation_type == "Reynolds Number":

    st.markdown(
        "### Reynolds Number Calculator"
    )

    st.caption(
        "Select the value you want to calculate. "
        "Enter the remaining known values."
    )


    unknown = st.selectbox(
        "What do you want to calculate?",
        [
            "Reynolds Number",
            "Density",
            "Velocity",
            "Pipe Diameter",
            "Dynamic Viscosity"
        ],
        key="reynolds_unknown"
    )


    # ========================================================
    # CALCULATE REYNOLDS NUMBER
    # ========================================================

    if unknown == "Reynolds Number":

        col1, col2 = st.columns(2)


        with col1:

            density = st.number_input(
                "Density (kg/m³)",
                min_value=0.000001,
                value=1000.0,
                key="re_density"
            )

            velocity = st.number_input(
                "Velocity (m/s)",
                min_value=0.000001,
                value=1.0,
                key="re_velocity"
            )


        with col2:

            diameter = st.number_input(
                "Pipe Diameter (m)",
                min_value=0.000001,
                value=0.05,
                key="re_diameter"
            )

            viscosity = st.number_input(
                "Dynamic Viscosity (Pa·s)",
                min_value=0.000001,
                value=0.001,
                format="%.6f",
                key="re_viscosity"
            )


        if st.button(
            "Calculate Reynolds Number",
            use_container_width=True,
            key="calculate_reynolds"
        ):

            reynolds_number = (
                density
                * velocity
                * diameter
                / viscosity
            )


            st.success(
                f"Reynolds Number = "
                f"{reynolds_number:,.2f}"
            )


            if reynolds_number < 2300:

                st.info(
                    "🟢 Flow is likely Laminar."
                )


            elif reynolds_number < 4000:

                st.warning(
                    "🟡 Flow is in the Transitional region."
                )


            else:

                st.error(
                    "🔴 Flow is likely Turbulent."
                )


    # ========================================================
    # CALCULATE DENSITY
    # ========================================================

    elif unknown == "Density":

        reynolds_number = st.number_input(
            "Reynolds Number",
            min_value=0.000001,
            value=50000.0,
            key="density_re"
        )

        velocity = st.number_input(
            "Velocity (m/s)",
            min_value=0.000001,
            value=1.0,
            key="density_velocity"
        )

        diameter = st.number_input(
            "Pipe Diameter (m)",
            min_value=0.000001,
            value=0.05,
            key="density_diameter"
        )

        viscosity = st.number_input(
            "Dynamic Viscosity (Pa·s)",
            min_value=0.000001,
            value=0.001,
            format="%.6f",
            key="density_viscosity"
        )


        if st.button(
            "Calculate Density",
            use_container_width=True,
            key="calculate_density"
        ):

            density = (
                reynolds_number
                * viscosity
                / (velocity * diameter)
            )


            st.success(
                f"Density = {density:,.2f} kg/m³"
            )


    # ========================================================
    # CALCULATE VELOCITY
    # ========================================================

    elif unknown == "Velocity":

        reynolds_number = st.number_input(
            "Reynolds Number",
            min_value=0.000001,
            value=50000.0,
            key="velocity_re"
        )

        density = st.number_input(
            "Density (kg/m³)",
            min_value=0.000001,
            value=1000.0,
            key="velocity_density"
        )

        diameter = st.number_input(
            "Pipe Diameter (m)",
            min_value=0.000001,
            value=0.05,
            key="velocity_diameter"
        )

        viscosity = st.number_input(
            "Dynamic Viscosity (Pa·s)",
            min_value=0.000001,
            value=0.001,
            format="%.6f",
            key="velocity_viscosity"
        )


        if st.button(
            "Calculate Velocity",
            use_container_width=True,
            key="calculate_velocity"
        ):

            velocity = (
                reynolds_number
                * viscosity
                / (density * diameter)
            )


            st.success(
                f"Velocity = {velocity:,.6f} m/s"
            )


    # ========================================================
    # CALCULATE PIPE DIAMETER
    # ========================================================

    elif unknown == "Pipe Diameter":

        reynolds_number = st.number_input(
            "Reynolds Number",
            min_value=0.000001,
            value=50000.0,
            key="diameter_re"
        )

        density = st.number_input(
            "Density (kg/m³)",
            min_value=0.000001,
            value=1000.0,
            key="diameter_density"
        )

        velocity = st.number_input(
            "Velocity (m/s)",
            min_value=0.000001,
            value=1.0,
            key="diameter_velocity"
        )

        viscosity = st.number_input(
            "Dynamic Viscosity (Pa·s)",
            min_value=0.000001,
            value=0.001,
            format="%.6f",
            key="diameter_viscosity"
        )


        if st.button(
            "Calculate Pipe Diameter",
            use_container_width=True,
            key="calculate_diameter"
        ):

            diameter = (
                reynolds_number
                * viscosity
                / (density * velocity)
            )


            st.success(
                f"Pipe Diameter = "
                f"{diameter:.6f} m"
            )


    # ========================================================
    # CALCULATE DYNAMIC VISCOSITY
    # ========================================================

    elif unknown == "Dynamic Viscosity":

        reynolds_number = st.number_input(
            "Reynolds Number",
            min_value=0.000001,
            value=50000.0,
            key="viscosity_re"
        )

        density = st.number_input(
            "Density (kg/m³)",
            min_value=0.000001,
            value=1000.0,
            key="viscosity_density"
        )

        velocity = st.number_input(
            "Velocity (m/s)",
            min_value=0.000001,
            value=1.0,
            key="viscosity_velocity"
        )

        diameter = st.number_input(
            "Pipe Diameter (m)",
            min_value=0.000001,
            value=0.05,
            key="viscosity_diameter"
        )


        if st.button(
            "Calculate Dynamic Viscosity",
            use_container_width=True,
            key="calculate_viscosity"
        ):

            viscosity = (
                density
                * velocity
                * diameter
                / reynolds_number
            )


            st.success(
                f"Dynamic Viscosity = "
                f"{viscosity:.8f} Pa·s"
            )

# ============================================================
# IDEAL GAS LAW
# ============================================================

elif calculation_type == "Ideal Gas Volume":

    st.markdown(
        "### Ideal Gas Law Calculator"
    )

    st.caption(
        "Select the value you want to calculate. "
        "Enter the remaining known values."
    )


    unknown = st.selectbox(
        "What do you want to calculate?",
        [
            "Volume",
            "Pressure",
            "Number of Moles",
            "Temperature"
        ],
        key="ideal_gas_unknown"
    )


    # ========================================================
    # CALCULATE VOLUME
    # ========================================================

    if unknown == "Volume":

        col1, col2, col3 = st.columns(3)


        with col1:

            pressure = st.number_input(
                "Pressure (Pa)",
                min_value=0.000001,
                value=101325.0,
                key="gas_volume_pressure"
            )


        with col2:

            moles = st.number_input(
                "Number of Moles (mol)",
                min_value=0.000001,
                value=1.0,
                key="gas_volume_moles"
            )


        with col3:

            temperature = st.number_input(
                "Temperature (K)",
                min_value=0.000001,
                value=298.15,
                key="gas_volume_temperature"
            )


        if st.button(
            "Calculate Gas Volume",
            use_container_width=True,
            key="calculate_gas_volume"
        ):

            R = 8.314

            volume = (
                moles
                * R
                * temperature
                / pressure
            )


            st.success(
                f"Gas Volume = {volume:.6f} m³"
            )

            st.caption(
                "Formula: V = nRT / P"
            )


    # ========================================================
    # CALCULATE PRESSURE
    # ========================================================

    elif unknown == "Pressure":

        volume = st.number_input(
            "Volume (m³)",
            min_value=0.000001,
            value=0.024465,
            format="%.6f",
            key="gas_pressure_volume"
        )

        moles = st.number_input(
            "Number of Moles (mol)",
            min_value=0.000001,
            value=1.0,
            key="gas_pressure_moles"
        )

        temperature = st.number_input(
            "Temperature (K)",
            min_value=0.000001,
            value=298.15,
            key="gas_pressure_temperature"
        )


        if st.button(
            "Calculate Pressure",
            use_container_width=True,
            key="calculate_gas_pressure"
        ):

            R = 8.314

            pressure = (
                moles
                * R
                * temperature
                / volume
            )


            st.success(
                f"Pressure = {pressure:,.2f} Pa"
            )

            st.caption(
                "Formula: P = nRT / V"
            )


    # ========================================================
    # CALCULATE NUMBER OF MOLES
    # ========================================================

    elif unknown == "Number of Moles":

        pressure = st.number_input(
            "Pressure (Pa)",
            min_value=0.000001,
            value=101325.0,
            key="gas_moles_pressure"
        )

        volume = st.number_input(
            "Volume (m³)",
            min_value=0.000001,
            value=0.024465,
            format="%.6f",
            key="gas_moles_volume"
        )

        temperature = st.number_input(
            "Temperature (K)",
            min_value=0.000001,
            value=298.15,
            key="gas_moles_temperature"
        )


        if st.button(
            "Calculate Number of Moles",
            use_container_width=True,
            key="calculate_gas_moles"
        ):

            R = 8.314

            moles = (
                pressure
                * volume
                / (R * temperature)
            )


            st.success(
                f"Number of Moles = {moles:.6f} mol"
            )

            st.caption(
                "Formula: n = PV / RT"
            )


    # ========================================================
    # CALCULATE TEMPERATURE
    # ========================================================

    elif unknown == "Temperature":

        pressure = st.number_input(
            "Pressure (Pa)",
            min_value=0.000001,
            value=101325.0,
            key="gas_temperature_pressure"
        )

        volume = st.number_input(
            "Volume (m³)",
            min_value=0.000001,
            value=0.024465,
            format="%.6f",
            key="gas_temperature_volume"
        )

        moles = st.number_input(
            "Number of Moles (mol)",
            min_value=0.000001,
            value=1.0,
            key="gas_temperature_moles"
        )


        if st.button(
            "Calculate Temperature",
            use_container_width=True,
            key="calculate_gas_temperature"
        ):

            R = 8.314

            temperature = (
                pressure
                * volume
                / (moles * R)
            )


            st.success(
                f"Temperature = {temperature:.2f} K"
            )

            st.caption(
                "Formula: T = PV / nR"
            )
# ============================================================
# HEAT DUTY
# ============================================================

elif calculation_type == "Heat Duty":

    st.markdown(
        "### Heat Duty Calculator"
    )

    st.caption(
        "Select the value you want to calculate. "
        "Enter the remaining known values."
    )


    unknown = st.selectbox(
        "What do you want to calculate?",
        [
            "Heat Duty",
            "Mass Flow Rate",
            "Specific Heat Capacity",
            "Temperature Change"
        ],
        key="heat_duty_unknown"
    )


    # ========================================================
    # CALCULATE HEAT DUTY
    # ========================================================

    if unknown == "Heat Duty":

        col1, col2, col3 = st.columns(3)


        with col1:

            mass_flow_rate = st.number_input(
                "Mass Flow Rate (kg/s)",
                min_value=0.000001,
                value=1.0,
                key="heat_mass_flow"
            )


        with col2:

            specific_heat = st.number_input(
                "Specific Heat Capacity (J/kg·K)",
                min_value=0.000001,
                value=4180.0,
                key="heat_specific_heat"
            )


        with col3:

            temperature_change = st.number_input(
                "Temperature Change (K)",
                min_value=0.000001,
                value=10.0,
                key="heat_temperature_change"
            )


        if st.button(
            "Calculate Heat Duty",
            use_container_width=True,
            key="calculate_heat_duty"
        ):

            heat_duty = (
                mass_flow_rate
                * specific_heat
                * temperature_change
            )


            st.success(
                f"Heat Duty = {heat_duty:,.2f} W"
            )

            st.caption(
                "Formula: Q = ṁ × Cp × ΔT"
            )


    # ========================================================
    # CALCULATE MASS FLOW RATE
    # ========================================================

    elif unknown == "Mass Flow Rate":

        heat_duty = st.number_input(
            "Heat Duty (W)",
            min_value=0.000001,
            value=41800.0,
            key="mass_heat_duty"
        )

        specific_heat = st.number_input(
            "Specific Heat Capacity (J/kg·K)",
            min_value=0.000001,
            value=4180.0,
            key="mass_specific_heat"
        )

        temperature_change = st.number_input(
            "Temperature Change (K)",
            min_value=0.000001,
            value=10.0,
            key="mass_temperature_change"
        )


        if st.button(
            "Calculate Mass Flow Rate",
            use_container_width=True,
            key="calculate_mass_flow"
        ):

            mass_flow_rate = (
                heat_duty
                / (specific_heat * temperature_change)
            )


            st.success(
                f"Mass Flow Rate = "
                f"{mass_flow_rate:.6f} kg/s"
            )

            st.caption(
                "Formula: ṁ = Q / (Cp × ΔT)"
            )


    # ========================================================
    # CALCULATE SPECIFIC HEAT CAPACITY
    # ========================================================

    elif unknown == "Specific Heat Capacity":

        heat_duty = st.number_input(
            "Heat Duty (W)",
            min_value=0.000001,
            value=41800.0,
            key="cp_heat_duty"
        )

        mass_flow_rate = st.number_input(
            "Mass Flow Rate (kg/s)",
            min_value=0.000001,
            value=1.0,
            key="cp_mass_flow"
        )

        temperature_change = st.number_input(
            "Temperature Change (K)",
            min_value=0.000001,
            value=10.0,
            key="cp_temperature_change"
        )


        if st.button(
            "Calculate Specific Heat Capacity",
            use_container_width=True,
            key="calculate_specific_heat"
        ):

            specific_heat = (
                heat_duty
                / (mass_flow_rate * temperature_change)
            )


            st.success(
                f"Specific Heat Capacity = "
                f"{specific_heat:,.2f} J/kg·K"
            )

            st.caption(
                "Formula: Cp = Q / (ṁ × ΔT)"
            )


    # ========================================================
    # CALCULATE TEMPERATURE CHANGE
    # ========================================================

    elif unknown == "Temperature Change":

        heat_duty = st.number_input(
            "Heat Duty (W)",
            min_value=0.000001,
            value=41800.0,
            key="deltaT_heat_duty"
        )

        mass_flow_rate = st.number_input(
            "Mass Flow Rate (kg/s)",
            min_value=0.000001,
            value=1.0,
            key="deltaT_mass_flow"
        )

        specific_heat = st.number_input(
            "Specific Heat Capacity (J/kg·K)",
            min_value=0.000001,
            value=4180.0,
            key="deltaT_specific_heat"
        )


        if st.button(
            "Calculate Temperature Change",
            use_container_width=True,
            key="calculate_temperature_change"
        ):

            temperature_change = (
                heat_duty
                / (mass_flow_rate * specific_heat)
            )


            st.success(
                f"Temperature Change = "
                f"{temperature_change:.4f} K"
            )

            st.caption(
                "Formula: ΔT = Q / (ṁ × Cp)"
            )

# ============================================================
# CHEMICAL ENGINEERING DOMAIN KEYWORDS
# ============================================================

chemical_keywords = [

    # General

    "chemical engineering",
    "chemical engineer",


    # Fluid Mechanics

    "reynolds",
    "fluid",
    "flow",
    "laminar",
    "turbulent",
    "viscosity",
    "density",
    "velocity",
    "pressure",
    "pressure drop",
    "pipe",
    "pipeline",
    "pump",
    "compressor",
    "bernoulli",
    "friction factor",


    # Heat Transfer

    "heat",
    "heat transfer",
    "heat exchanger",
    "heat duty",
    "conduction",
    "convection",
    "radiation",
    "thermal conductivity",
    "nusselt",
    "prandtl",


    # Mass Transfer

    "mass transfer",
    "diffusion",
    "fick",
    "schmidt",
    "sherwood",


    # Thermodynamics

    "thermodynamics",
    "enthalpy",
    "entropy",
    "gibbs",
    "ideal gas",
    "gas law",
    "temperature",
    "phase equilibrium",
    "vapor pressure",


    # Mass and Energy Balance

    "mass balance",
    "material balance",
    "energy balance",
    "material flow",


    # Separation Processes

    "distillation",
    "absorption",
    "stripping",
    "extraction",
    "evaporation",
    "filtration",
    "drying",
    "adsorption",
    "membrane",
    "separation",
    "crystallization",


    # Reaction Engineering

    "reaction",
    "chemical reaction",
    "reactor",
    "cstr",
    "pfr",
    "batch reactor",
    "reaction rate",
    "kinetics",
    "catalyst",


    # Process Control

    "process control",
    "pid",
    "controller",
    "instrumentation",
    "control valve",


    # Industrial Chemical Engineering

    "refinery",
    "petroleum",
    "process plant",
    "chemical plant",
    "process engineering"

]


# ============================================================
# DOMAIN CHECK FUNCTION
# ============================================================

def is_chemical_engineering_question(
    question
):

    question_lower = question.lower()


    # Keyword check

    keyword_match = any(

        keyword in question_lower

        for keyword in chemical_keywords

    )


    # Retrieve knowledge and check similarity

    try:

        results = retrieve_knowledge(
            question,
            top_k=1
        )


        best_score = results[0]["score"]


    except Exception:

        best_score = 0


    # Accept if keyword OR meaningful RAG similarity

    if keyword_match:

        return True


    if best_score > 0.12:

        return True


    return False
# ============================================================
# SESSION STATE - QUESTION HISTORY
# ============================================================

if "question_history" not in st.session_state:
    st.session_state.question_history = []

# ============================================================
# AI ASSISTANT SECTION
# ============================================================

st.divider()

st.markdown("## 🤖 AI Assistant")

st.caption(
    "Ask domain-focused Chemical Engineering questions and "
    "receive AI-generated answers using the local RAG system."
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader("💬 Ask a Chemical Engineering Question")

st.caption("Try one of these example questions or enter your own question below.")


# Store selected example question
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("### 📚 Concept")

    if st.button(
        "What is distillation?",
        use_container_width=True
    ):
        st.session_state.selected_question = (
            "What is distillation?"
        )


with col2:

    st.markdown("### 🧮 Fluid Mechanics")

    if st.button(
        "Explain Reynolds number",
        use_container_width=True
    ):
        st.session_state.selected_question = (
            "Explain Reynolds number"
        )


with col3:

    st.markdown("### 🔥 Heat Transfer")

    if st.button(
        "What is heat duty?",
        use_container_width=True
    ):
        st.session_state.selected_question = (
            "What is heat duty?"
        )

# ============================================================
# USER QUESTION INPUT
# ============================================================

question = st.text_input(
    "Enter your question:",
    value=st.session_state.selected_question,
    placeholder=(
        "Example: What is the difference between "
        "laminar and turbulent flow?"
    )
)
st.session_state.selected_question = question


# ============================================================
# ASK AI BUTTON
# ============================================================

if st.button(
    "🤖 Ask AI",
    type="primary",
    use_container_width=True
):

    # ========================================================
    # EMPTY QUESTION CHECK
    # ========================================================

    if not question.strip():

        st.warning(
            "⚠️ Please enter a Chemical Engineering question."
        )

        st.stop()


    # ========================================================
    # DOMAIN CHECK
    # ========================================================

    if not is_chemical_engineering_question(question):

        st.warning(
            "⚠️ This question does not appear to be related "
            "to Chemical Engineering. Please ask a "
            "Chemical Engineering question."
        )

        st.stop()


    # ========================================================
    # AI PROCESSING STATUS
    # ========================================================

    with st.status(
        "🤖 Processing your question...",
        expanded=True
    ) as status:


        # ====================================================
        # RETRIEVE KNOWLEDGE
        # ====================================================

        st.write(
            "🔍 Searching the Chemical Engineering knowledge base..."
        )


        try:

            retrieved_results = retrieve_knowledge(
                question,
                top_k=3
            )


        except Exception as e:

            st.error(
                f"❌ Knowledge retrieval error: {e}"
            )

            st.stop()


        st.write(
            "📚 Retrieving the most relevant engineering concepts..."
        )


        # Best similarity score

        best_score = retrieved_results[0]["score"]
        # ====================================================
        # MINIMUM RETRIEVAL THRESHOLD
        # ====================================================

        MIN_RETRIEVAL_THRESHOLD = 0.10

        if best_score < MIN_RETRIEVAL_THRESHOLD:

            st.warning(
                "⚠️ Limited relevant information was found "
                "in the local Chemical Engineering knowledge base. "
                "The answer may rely on the AI model's general "
                "Chemical Engineering knowledge."
            )
        # ====================================================
        # RAG CONFIDENCE CALCULATION
        # ====================================================

        confidence_percentage = min(
            max(best_score * 100, 0),
            100
        )

        if best_score >= 0.50:

            confidence_level = "High"

        elif best_score >= 0.25:

            confidence_level = "Medium"

        else:

            confidence_level = "Low"


        # Combine retrieved knowledge chunks

        context = "\n\n".join(
            result["chunk"]
            for result in retrieved_results
        )


        st.write(
            "🧠 Preparing context for the AI model..."
        )


        # ====================================================
        # CREATE GEMINI PROMPT
        # ====================================================

        prompt = f"""

You are an AI assistant specialized ONLY in
Chemical Engineering.

Your task is to answer the user's question clearly,
accurately and educationally.

You have been provided with retrieved Chemical Engineering
reference information from a local knowledge base.

Use this information as your primary context.

Do not answer unrelated questions.

For conceptual questions:

1. Give a clear definition.
2. Explain the working principle.
3. Explain important engineering factors.
4. Give a relevant Chemical Engineering example.

For comparison questions:

1. Clearly explain both concepts.
2. Present the important differences.
3. Use a table if useful.
4. Give a practical Chemical Engineering example.

For numerical or calculation questions:

1. Identify the correct formula.
2. Clearly show the calculation steps.
3. Include units.
4. Give the final answer.
5. Briefly explain what the result means.

IMPORTANT FORMATTING RULES:

- Use clean Markdown.
- Use headings where useful.
- Use bullet points where useful.
- Do NOT generate HTML.
- Do NOT generate SVG.
- Do NOT generate localhost links.
- Do NOT generate anchor links.
- Do NOT include unnecessary URLs.
- Do NOT include [svg](...) links.


RETRIEVED CHEMICAL ENGINEERING REFERENCE:

{context}


USER QUESTION:

{question}


Now provide a clear and professional answer.

"""


        # ====================================================
        # GENERATE GEMINI RESPONSE
        # ====================================================

        st.write(
            "🤖 Generating your AI-powered answer..."
        )


        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            answer = response.text
            # Save question to history

            if question not in st.session_state.question_history:

                st.session_state.question_history.append(
                    question
                )

                # Keep only the latest 5 questions
                st.session_state.question_history = (
                    st.session_state.question_history[-5:]
                )


        except Exception as e:

            st.error(
                f"❌ Gemini API Error:\n\n{e}"
            )

            st.stop()


        # ====================================================
        # COMPLETE STATUS
        # ====================================================

        status.update(
            label="✅ Answer generated successfully!",
            state="complete",
            expanded=False
        )


    # ========================================================
    # CLEAN UNWANTED SVG / LOCALHOST LINKS
    # ========================================================

    answer = re.sub(
        r"\[svg\]\(http://localhost:8501/#[^)]+\)",
        "",
        answer
    )


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    st.markdown(
        """
        <div class="answer-box">
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "🤖 Assistant Answer"
    )

    st.markdown(
        answer
    )

    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True
    )


    # ============================================================
    # RAG CONFIDENCE INDICATOR
    # ============================================================

    st.divider()

    st.subheader(
        "📊 Knowledge Retrieval Confidence"
    )

    st.progress(
        int(confidence_percentage)
    )

    if confidence_level == "High":

        st.success(
            f"🟢 High Confidence — "
            f"{confidence_percentage:.1f}% relevant knowledge found"
        )

    elif confidence_level == "Medium":

        st.warning(
            f"🟡 Medium Confidence — "
            f"{confidence_percentage:.1f}% relevant knowledge found"
        )

    else:

        st.error(
            f"🔴 Low Confidence — "
            f"{confidence_percentage:.1f}% relevant knowledge found"
        )
    # ============================================================
    # RECENT QUESTION HISTORY
    # ============================================================

    if st.session_state.question_history:

        st.divider()

        st.subheader("🕘 Recent Questions")

        st.caption(
            "Your recently asked Chemical Engineering questions."
        )


        for index, previous_question in enumerate(
            reversed(st.session_state.question_history),
            start=1
        ):

            st.markdown(
                f"**{index}.** {previous_question}"
            )


        if st.button(
            "🗑️ Clear Question History"
        ):

            st.session_state.question_history = []

            st.rerun()


    # ========================================================
    # SHOW RETRIEVED KNOWLEDGE
    # ========================================================

    with st.expander(
        "📚 View Retrieved Knowledge"
    ):

        st.caption(
            f"Best similarity score: "
            f"{best_score:.3f}"
        )


        for number, result in enumerate(
            retrieved_results,
            start=1
        ):

            st.markdown(
                f"### Knowledge Chunk {number}"
            )


            st.write(
                result["chunk"]
            )


            st.caption(
                f"Similarity Score: "
                f"{result['score']:.3f}"
            )


            st.divider()
# ============================================================
# ABOUT THE SYSTEM
# ============================================================

st.divider()

st.markdown("## 📊 About This System")

st.caption(
    "Architecture and technologies used in the Chemical Engineering AI Assistant."
)


col1, col2 = st.columns(2)


with col1:

    st.markdown("### 🧠 AI & RAG Workflow")

    st.markdown(
        """
        1. User enters a Chemical Engineering question.
        2. Domain validation checks whether the question is relevant.
        3. TF-IDF converts the question and knowledge into vectors.
        4. Cosine similarity finds the most relevant knowledge.
        5. Top relevant chunks are retrieved.
        6. Retrieval confidence is calculated.
        7. Relevant context is sent to Gemini AI.
        8. Gemini generates a professional answer.
        """
    )


with col2:

    st.markdown("### ⚙️ Technology Stack")

    st.markdown(
        """
        - 🐍 Python
        - 🎨 Streamlit
        - 🤖 Google Gemini API
        - 📚 Scikit-learn
        - 🔍 TF-IDF Vectorization
        - 📊 Cosine Similarity
        - 🧠 Retrieval-Augmented Generation (RAG)
        """
    )


st.divider()

st.markdown("### ✨ Key Features")

st.markdown(
    """
    - Domain-focused Chemical Engineering AI Assistant
    - Local knowledge base
    - TF-IDF-based knowledge retrieval
    - Cosine similarity ranking
    - Retrieval confidence indicator
    - Minimum retrieval threshold
    - Gemini AI-powered answers
    - Engineering calculation tools
    - Recent question history
    - Transparent retrieved knowledge display
    """
)


st.caption(
    "Built as an AI-powered Chemical Engineering learning and engineering support system."
)