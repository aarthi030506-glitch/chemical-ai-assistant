# 🧪 Chemical Engineering AI Assistant

An AI-powered Chemical Engineering application built using **Python, Streamlit, Scikit-learn, and Google Gemini AI**.

The application helps users ask Chemical Engineering questions and receive AI-generated answers supported by relevant information retrieved from a local knowledge base.

---

## 🚀 Features

### 🤖 AI-Powered Assistant

- Answers Chemical Engineering-related questions
- Performs Chemical Engineering domain validation
- Retrieves relevant information from a local knowledge base
- Uses Google Gemini AI to generate structured answers
- Displays recently asked questions

### 📚 Local Knowledge Retrieval

The application uses a local retrieval system to find relevant information before generating an AI response.

The retrieval process uses:

- TF-IDF Vectorization
- Cosine Similarity
- Top relevant knowledge chunks
- Retrieval confidence indicator

### 🧮 Chemical Engineering Calculator

The application includes three calculation tools:

- Reynolds Number
- Ideal Gas Volume
- Heat Duty

---

## 🏗️ System Architecture

![Chemical Engineering AI Assistant Architecture](./assets/architecture_diagram.png)

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| 🐍 **Python** | Core programming language |
| 🌐 **Streamlit** | Interactive web application interface |
| 🤖 **Google Gemini AI** | AI-powered answer generation |
| 🧠 **Scikit-learn** | Machine learning and local knowledge retrieval |
| 📚 **TF-IDF Vectorization** | Converts Chemical Engineering text into numerical representations |
| 🔍 **Cosine Similarity** | Finds and ranks the most relevant knowledge |
| 📄 **Local Knowledge Base** | Stores Chemical Engineering reference information in `knowledge.txt` |
| 🔐 **Git & GitHub** | Version control and project hosting |

## 🧠 Key Technical Learnings

- Built a **Retrieval-Augmented Generation (RAG)** system for Chemical Engineering questions.
- Learned how to use **TF-IDF Vectorization** to convert text from a local knowledge base into numerical representations.
- Used **Cosine Similarity** to retrieve and rank the most relevant knowledge chunks.
- Integrated **Google Gemini AI** to generate clear and structured answers using retrieved context.
- Designed the system to avoid repeated cloud embedding requests after encountering **API quota limitations**.
- Implemented a **local knowledge retrieval system**, reducing dependency on external embedding APIs.
- Used **Streamlit** to develop an interactive AI-powered web application.
- Implemented **domain validation** to keep the assistant focused on Chemical Engineering questions.
- Added a **RAG confidence indicator** based on the relevance score of retrieved knowledge.
- Integrated basic **Chemical Engineering calculation tools** for Reynolds Number, Ideal Gas Volume, and Heat Duty.
- Learned to manage sensitive API credentials securely using `.streamlit/secrets.toml` and `.gitignore`.
- Used **Git and GitHub** for version control and project documentation.

## 🔮 Future Improvements

- 📄 Add support for **PDF, DOCX, and other document uploads** to expand the knowledge base.
- 🧠 Implement **advanced semantic search and embedding-based retrieval** when scalable API or local embedding resources are available.
- 💬 Add **multi-turn conversational memory** for follow-up questions and more natural interactions.
- 📚 Expand the **Chemical Engineering knowledge base** with additional subjects and reference materials.
- 🧮 Add more advanced **Chemical Engineering calculation and problem-solving capabilities**.
- 📊 Include **interactive graphs and visualizations** for engineering concepts and calculations.
- 📁 Allow users to upload their own **Chemical Engineering notes and study materials**.
- ☁️ Deploy the application to a **cloud platform** for public access.
- 👤 Add **user profiles and personalized learning features**.
- 🎤 Explore **voice-based interaction** for a more accessible AI assistant.

## 👩‍💻 Author

**Aarthi**

Chemical Engineering Student | AI & Technology Enthusiast

Developed the **Chemical Engineering AI Assistant** as an AI-powered learning and engineering assistance system.