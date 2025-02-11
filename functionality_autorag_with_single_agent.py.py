import os
import sys
import requests
from getpass import getpass

# ------------------------------
# Utility Functions
# ------------------------------

def setup_directories():
    """Create local directories needed for data and vector storage."""
    try:
        os.makedirs("local_data", exist_ok=True)
        os.makedirs("local_data2", exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Creating directories: {e}")
        sys.exit(1)

def download_file(url: str, output_path: str):
    """Download a file from a URL to a given output path."""
    try:
        print(f"Downloading file from {url} ...")
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Saved file to {output_path}")
    except Exception as e:
        print(f"[ERROR] Downloading file from {url}: {e}")
        sys.exit(1)

def setup_openai_api_key():
    """Prompt for the OpenAI API key and set it in the environment."""
    try:
        api_key = getpass("Enter your OpenAI API key: ").strip()
        if not api_key:
            raise ValueError("API key cannot be empty!")
        os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"[ERROR] Setting OpenAI API key: {e}")
        sys.exit(1)

def initialize_embedding():
    """Initialize and return the OpenAI embedding instance."""
    try:
        from camel.embeddings import OpenAIEmbedding
        from camel.types import EmbeddingModelType

        embedding_instance = OpenAIEmbedding(
            model_type=EmbeddingModelType.TEXT_EMBEDDING_3_LARGE
        )
        return embedding_instance
    except Exception as e:
        print(f"[ERROR] Initializing embedding model: {e}")
        sys.exit(1)

# ------------------------------
# Core Agent Function
# ------------------------------

def single_agent(query: str, embedding_instance) -> str:
    """
    Runs a single agent with Auto RAG.
    It retrieves relevant context from given contents and then
    asks the ChatAgent to answer the query based on the retrieved info.
    """
    try:
        from camel.agents import ChatAgent
        from camel.messages import BaseMessage
        from camel.types import RoleType
        from camel.retrievers import AutoRetriever
        from camel.types import StorageType

        # Define system message for the assistant agent.
        assistant_sys_msg = BaseMessage(
            role_name="Assistant",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content=(
                "You are a helpful assistant that answers questions based on "
                "the retrieved context. If you can't answer, just say 'I don't know.'"
            ),
        )

        # Initialize the AutoRetriever using Qdrant as vector storage.
        auto_retriever = AutoRetriever(
            vector_storage_local_path="local_data2/",
            storage_type=StorageType.QDRANT,
            embedding_model=embedding_instance,
        )

        # Retrieve context from both a local file and a remote URL.
        retrieved_info = auto_retriever.run_vector_retriever(
            query=query,
            contents=[
                "local_data/camel_paper.pdf",  # local content
                "https://github.com/camel-ai/camel/wiki/Contributing-Guidlines",  # remote content
            ],
            top_k=1,
            return_detailed_info=False,
            similarity_threshold=0.5,
        )

        # Prepare the user message including the retrieved context.
        user_msg = BaseMessage.make_user_message(
            role_name="User", content=str(retrieved_info)
        )

        # Initialize the chat agent and get the assistant's response.
        agent = ChatAgent(assistant_sys_msg)
        assistant_response = agent.step(user_msg)
        return assistant_response.msg.content

    except Exception as e:
        print(f"[ERROR] in single_agent: {e}")
        sys.exit(1)

# ------------------------------
# Main Execution Flow
# ------------------------------

def main():
    # Create necessary directories.
    setup_directories()

    # Download the CAMEL paper (if not already present)
    camel_paper_path = "local_data/camel_paper.pdf"
    if not os.path.exists(camel_paper_path):
        paper_url = "https://arxiv.org/pdf/2303.17760.pdf"
        download_file(paper_url, camel_paper_path)
    else:
        print(f"Found local file: {camel_paper_path}")

    # Setup OpenAI API key.
    setup_openai_api_key()

    # Initialize the embedding instance.
    embedding_instance = initialize_embedding()

    # Define your query. You may extend this to take a command-line argument.
    query = "If I'm interested in contributing to the CAMEL project, what should I do?"

    print("\nRunning single agent with Auto RAG...")
    response = single_agent(query, embedding_instance)
    print("\n--- Assistant Response ---")
    print(response)

if __name__ == "__main__":
    main()
