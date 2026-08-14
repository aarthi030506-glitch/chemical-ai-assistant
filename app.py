import os
import math

from dotenv import load_dotenv
from google import genai

from tools import (
    calculate_reynolds_number,
    calculate_ideal_gas_volume,
    calculate_heat_duty
)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


# -----------------------------
# TOOL FUNCTIONS
# -----------------------------

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


# -----------------------------
# RAG FUNCTIONS
# -----------------------------

def create_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


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


# -----------------------------
# LOAD KNOWLEDGE BASE
# -----------------------------

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


print("Creating knowledge embeddings...")

chunk_embeddings = []

for chunk in chunks:

    embedding = create_embedding(chunk)

    chunk_embeddings.append(
        (chunk, embedding)
    )


print(
    "Knowledge base loaded!",
    len(chunks),
    "chunks."
)


# -----------------------------
# CREATE CHAT
# -----------------------------

chat = client.chats.create(
    model="gemini-3.5-flash",

    config={
        "tools": tools
    }
)


# -----------------------------
# MAIN ASSISTANT
# -----------------------------

print("\n🧪 Chemical Engineering AI Assistant")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":

        print("Assistant: Goodbye!")

        break


    # Create embedding for question
    question_embedding = create_embedding(
        question
    )


    # Calculate similarity
    scores = []

    for chunk, embedding in chunk_embeddings:

        similarity = cosine_similarity(
            question_embedding,
            embedding
        )

        scores.append(
            (similarity, chunk)
        )


    # Sort by similarity
    scores.sort(
        reverse=True,
        key=lambda x: x[0]
    )


    # Select top 3 chunks
    top_chunks = scores[:3]


    context = "\n\n".join(
        chunk
        for score, chunk in top_chunks
    )


    # Give retrieved information to Gemini
    prompt = f"""
You are a Chemical Engineering AI Assistant.

You have access to a Chemical Engineering knowledge base
and Python calculation tools.

Use the retrieved reference information when answering
conceptual questions.

For numerical engineering calculations, use the appropriate
Python calculation tool instead of doing the arithmetic yourself.

Retrieved reference information:

{context}

User question:

{question}

Answer clearly and explain the result in simple terms.
"""


    response = chat.send_message(prompt)


    print("\nAssistant:")
    print(response.text)
    print()
