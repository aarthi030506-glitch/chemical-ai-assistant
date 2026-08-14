import os
import math
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


# Create embeddings
def create_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# Calculate cosine similarity
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


# Load knowledge base
with open("knowledge.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()


# Split knowledge into chunks
chunks = [chunk.strip() for chunk in knowledge.split("\n\n") if chunk.strip()]

# Create embeddings for chunks
chunk_embeddings = []

for chunk in chunks:
    embedding = create_embedding(chunk)
    chunk_embeddings.append((chunk, embedding))

print("Knowledge base loaded!")
print("Number of chunks:", len(chunks))


# Get user's question
question = input("\nAsk a Chemical Engineering question: ")

# Create embedding for question
question_embedding = create_embedding(question)


# Find the most relevant chunks
scores = []

for chunk, embedding in chunk_embeddings:
    similarity = cosine_similarity(question_embedding, embedding)
    scores.append((similarity, chunk))

scores.sort(reverse=True, key=lambda x: x[0])


# Take the top 3 relevant chunks
top_chunks = scores[:3]

context = "\n\n".join(chunk for score, chunk in top_chunks)


# Send retrieved information to Gemini
prompt = f"""
You are a Chemical Engineering AI Assistant.

Answer the user's question using the provided reference information.

REFERENCE INFORMATION:
{context}

USER QUESTION:
{question}

Instructions:
- Give a clear and accurate answer.
- Prefer the reference information when answering.
- If the reference information does not contain enough information, say that the available reference material is insufficient.
- Do not invent information.
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)


print("\nAssistant:")
print(response.text)