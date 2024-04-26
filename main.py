# from llama_index.llms.ollama import Ollama
# from llama_parse import LlamaParse
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
# from llama_index.core.embeddings import resolve_embed_model
# from dotenv import load_dotenv

# load_dotenv()

# llm = Ollama(model='mistral', request_timeout= 60.0)
 

 
# parser = LlamaParse(result_type='markdown')

# file_extractor = {".pdf": parser}
# document = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# embed_model = resolve_embed_model("local:BAAI/bge-m3")
# vector_index = VectorStoreIndex.from_documents(document, embed_model= embed_model)
# query_engine = vector_index.as_query_engine(llm=llm)

# result = query_engine.query("What are some of the routes in the API?")

# # result = llm.complete("Hello World")

# print(result)


from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv
import httpx
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    # Initialize Ollama with a longer request timeout
    llm = Ollama(model='mistral', request_timeout=60.0)

    # Initialize LlamaParse for processing PDF files
    parser = LlamaParse(result_type='markdown')

    # Define file extractor for PDF documents
    file_extractor = {".pdf": parser}

    # Load documents from a directory
    document_reader = SimpleDirectoryReader("./data", file_extractor=file_extractor)
    document = document_reader.load_data()

    # Resolve and initialize the embedding model
    embed_model = resolve_embed_model("local:BAAI/bge-m3")

    # Create a VectorStoreIndex from the documents and embedding model
    vector_index = VectorStoreIndex.from_documents(document, embed_model=embed_model)

    # Create a query engine from the vector index and Ollama instance
    query_engine = vector_index.as_query_engine(llm=llm)

    # Perform a sample query
    query = "What are some of the routes in the API?"
    result = query_engine.query(query)

    # Print the result
    print(result)

except httpx.ReadTimeout as e:
    logging.error(f"HTTP request timed out: {e}")
except Exception as e:
    logging.error(f"An error occurred: {e}")
