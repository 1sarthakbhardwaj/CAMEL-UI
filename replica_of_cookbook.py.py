import os
import requests
from getpass import getpass
from dotenv import load_dotenv
import faulthandler
faulthandler.enable()
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.makedirs('local_data', exist_ok=True)

paper_path = "local_data/camel_paper.pdf"
if not os.path.exists(paper_path):
    url = "https://arxiv.org/pdf/2303.17760.pdf"
    response = requests.get(url)
    with open(paper_path, 'wb') as file:
         file.write(response.content)

from camel.embeddings import OpenAIEmbedding
from camel.types import EmbeddingModelType

embedding_instance = OpenAIEmbedding(model_type=EmbeddingModelType.TEXT_EMBEDDING_3_LARGE)

from camel.storages import QdrantStorage

storage_instance = QdrantStorage(
    vector_dim=embedding_instance.get_output_dim(),
    path="local_data",
    collection_name="camel_paper",
)

from camel.retrievers import VectorRetriever
try:
    vector_retriever = VectorRetriever(embedding_model=embedding_instance,storage=storage_instance)
    vector_retriever.process(content=paper_path,)
except Exception as e:
    print(f"Qdrant Error: {e}")

query = input("Enter your query: ")
retrieved_info = vector_retriever.query(query=query, top_k=1)
print("Retrieved Information:")
print(retrieved_info)