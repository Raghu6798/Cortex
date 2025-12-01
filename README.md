# Cortex Agent Runtime Backend

Welcome to the backend documentation for the Cortex Agent Runtime. This system serves as the central brain for managing, executing, and orchestrating AI agents. It provides a robust API for creating agents, managing their lifecycle, handling chat sessions, and securely executing code.

## 🌟 Overview

The Cortex Backend is designed to be a flexible and powerful runtime for AI applications. It abstracts away the complexities of managing different LLM providers, agent frameworks, and execution environments, allowing developers to focus on building intelligent behaviors.

At its core, it connects:
- **Users** who interact with agents.
- **AI Models** (LLMs) that power the intelligence.
- **Tools & Knowledge** that agents use to perform tasks.
- **Sandboxes** where code is safely executed.

## 🧩 Core Concepts

### 🤖 Agents
Agents are the primary entities in the system. An agent is defined by:
- **Identity**: Name, description, and configuration.
- **Brain**: The underlying LLM (e.g., GPT-4, Claude) and framework (e.g., LangChain, LlamaIndex).
- **Tools**: Capabilities enabled for the agent (e.g., web search, database access).
- **Memory**: Ability to retain context across conversations.

### 💬 Chat Sessions
Interactions with agents happen within **Sessions**. A session maintains the conversation history, context, and state. The backend manages the flow of messages between the user and the agent, ensuring context is preserved.

### 📦 Sandboxes
Security is paramount when agents write and execute code. **Sandboxes** provide isolated, secure environments (powered by E2B) where agents can run code without risking the host system. This is crucial for "Code Interpreter" style features.

### 📚 Knowledge Bases (RAG)
Agents often need access to private or specific data. **Knowledge Bases** allow you to upload documents and data, which are then indexed (Vector DB) and made available to agents via Retrieval Augmented Generation (RAG).

### 🔐 Secrets Management
To interact with external services (like GitHub, AWS, or LLM providers), agents need credentials. The backend provides a secure **Secrets Manager** to store and encrypt these sensitive values, injecting them into agents only when needed.

## 🚀 Key Features

- **Multi-Framework Support**: Run agents built with LangChain, LlamaIndex, Agno, and more.
- **Universal API**: A consistent API surface for managing different types of agents.
- **Observability**: Built-in tracking for token usage, latency, and errors.
- **Scalable Architecture**: Designed to handle multiple concurrent agent sessions.
- **Extensible Tooling**: Easy to add new tools and integrations.

## 📂 Project Structure

Here is a high-level view of the backend organization:

- **`app/api`**: The API layer. This contains the "Routes" which define the URL endpoints you can hit (e.g., `/agents`, `/sessions`).
- **`app/services`**: The business logic layer. This is where the actual work happens—creating agents, processing files, managing sandboxes.
- **`app/db`**: The database layer. Defines the data models (tables) for Agents, Sessions, etc.
- **`app/schemas`**: Data validation. Defines what data is expected in requests and what is returned in responses.
- **`app/integrations`**: Connectors to external services and LLM providers.

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- PostgreSQL (or compatible database)
- Docker (optional, for containerized deployment)

### Running the Server

> **Note**: Always ensure you are in the `/backend` directory before running commands.

1.  **Install Dependencies**:
    We use `uv` or `pip` to manage dependencies.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Setup**:
    Copy `.env.example` to `.env` and configure your database URL and API keys.

3.  **Run the Application**:
    You can use the `just` command runner if installed, or run directly with Python/Uvicorn.
    ```bash
    # Using just (Recommended)
    just run
    
    # Or directly
    python main.py
    ```

The API will be available at `http://localhost:8000`. You can view the interactive API documentation at `http://localhost:8000/docs`.