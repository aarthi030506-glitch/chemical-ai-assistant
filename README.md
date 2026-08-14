# 🧪 Chemical Engineering AI Assistant

An AI-powered Chemical Engineering assistant built using **Python, Google Gemini, RAG, embeddings, function calling, and Streamlit**.

The application can answer Chemical Engineering questions using a custom knowledge base and perform engineering calculations using Python tools.

---

## 🎯 Project Overview

The goal of this project is to build an AI assistant specifically for Chemical Engineering.

Instead of relying only on the general knowledge of an LLM, the application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from a custom Chemical Engineering knowledge base.

For numerical calculations, the AI can use dedicated Python functions instead of relying on the language model to perform arithmetic.

---

## 🚀 Features

### 📚 RAG-based Question Answering

The application:

1. Loads Chemical Engineering reference material.
2. Splits the content into smaller chunks.
3. Creates embeddings for each chunk.
4. Converts the user's question into an embedding.
5. Calculates cosine similarity.
6. Retrieves the most relevant information.
7. Sends the retrieved information to Gemini.
8. Generates the final answer.

### 🧮 Engineering Calculations

The assistant currently includes Python tools for:

- Reynolds Number
- Ideal Gas Law
- Heat Duty

### 🤖 Gemini Function Calling

Gemini can identify when a numerical calculation is required and call the appropriate Python function.

For example:

```text
Calculate Reynolds number for density 1000,
velocity 2, diameter 0.05 and viscosity 0.001.