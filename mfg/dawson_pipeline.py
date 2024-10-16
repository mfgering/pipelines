import os
import logging
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
import openai
from utils.pipelines.main import get_last_user_message
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import llama_index.vector_stores.chroma
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings    
from llama_index.core.tools import QueryEngineTool
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode
from huggingface_hub import snapshot_download
from chromadb import Client
from chromadb.utils import persistent_client

log = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        DOCS_DIR: str = "dawson_docs"
        LLM_PROVIDER: str = "openai"
        LLM_MODEL: str = "gpt-4-turbo"
        OPENAI_API_KEY: str = "your-openai-api-key"
        EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "pipeline_example"

        # The name of the pipeline.
        self.name = "Dawson Docs Pipeline"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
            }
        )
        self._query_engine = None
        self._embedding_model_path = None

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        self.setup()
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        print(body)
        print(user)

        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        # This function is called after the OpenAI API response is completed. You can modify the messages after they are received from the OpenAI API.
        print(f"outlet:{__name__}")

        print(body)
        print(user)

        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")
        if self._query_engine is None:
            self.setup()
        # If you'd like to check for title generation, you can add the following check
        m = get_last_user_message(messages)
        if 'Create' in m and 'title' in m:
        #if body.get("title", False):
            print("Title Generation Request")

        print(messages)
        print(user_message)
        print(body)

# TODO: Do a local query to find RAG context
# Use local embeddings (SentenceTransformer or HuggingFace models) to find relevant documents from your local storage (e.g., a vector database, a directory of PDFs, etc.)
# Query local index to get context
# Extract context for openai query
# Encode context metadata in response 
# TODO: Need embedding_model, embedding_function = sentence_transformers.SentenceTransformer
        if self.valves.LLM_PROVIDER == "openai":
            openai.api_key = self.valves.OPENAI_API_KEY
            llm = OpenAI(model="gpt-4-turbo", temperature=0)
        else:
            llm = Ollama(model="llama3.1:8b", request_timeout=120.0)

        #Settings.llm = OpenAI(model="gpt-4-turbo", temperature=0)
        Settings.llm = llm
        #index = VectorStoreIndex.from_documents(documents) #TODO: Change this to use Nodes
        #index = self._index
        #query_engine = index.as_query_engine()
        #TODO: FIX this to query the index! It appears to only query the llm
        response = self._query_engine.query(user_message)

        return response.response

    def setup(self):
        # Strategy:
        #   1. Setup: 
        #        init vector db
        #          if exists:
        #             load index from db
        #          else:
        #             init embedding model
        #               get_embedding_model_path
        #             create embeddings for docs
        #             persist db
        #   2. Pipe:
        #        create embeddings for query
        #          get_embedding_model_path
        #        query db
        #        retrieve context/content
        #        complete query using context
        #        augment response to include citations
        #
        #self.get_embedding_model_path() # Need the embedding model for queries and to init the store
        self.init_db()
        # initialize client, setting path to save data
    
    def init_db(self):
        db_path = './chroma_db'
        collection_name = 'dawson'
        client = None
        is_db_new = False
        if os.path.exists(db_path):
            client = persistent_client(path=db_path)
        else:
            print("Creating new db")
            client = persistent_client(path=db_path)
            is_db_new = True
        collection = self.client.get_or_create_collection(collection_name)
        try:
            storage_context = StorageContext.from_defaults(persist_dir=db_path)
            index = load_index_from_storage(storage_context)
            pass
        except Exception as e:
            print(e)
            pass
        #TODO: reconcile this with the repo_id stuff
        collection_name = 'dawson'
        vector_store = ChromaVectorStore(collection_name=collection_name)
        if os.path.exists(db_path):
            pass
        db = chromadb.PersistentClient(path=db_path)
        # Create and init the collections if necessary
        try:
            chroma_collection = db.get_collection('dawson')
        except ValueError:

            chroma_collection = db.create_collection("dawson")
            nodes = []
            documents = SimpleDirectoryReader(self.valves.DOCS_DIR).load_data()
            for i, doc in enumerate(documents):
                node = TextNode(text=doc.get_content(), id=f"foo-{doc.node_id}")
                nodes.append(node)
        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.persist()
        #index = load_index_from_storage(storage_context)
        #TODO: THIS IS not right -- do not use openai; use the local model!
        #os.environ["OPENAI_API_KEY"] = self.valves.OPENAI_API_KEY
        #openai.api_key = self.valves.OPENAI_API_KEY
        self._index = VectorStoreIndex(nodes, storage_context=storage_context) 
        # configure retriever
        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=10,
        )
        self._response_synthesizer = get_response_synthesizer()
        # assemble query engine
        self._query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=self._response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        )
        pass
    
    def get_embedding_model_path(self):
        #NOTE: Tailored from rag.main.update_embedding_model()
        if not self._embedding_model_path:
            self._embedding_model_path = self.get_embedding_model_path()
        return self._embedding_model_path

    def get_embedding_model_path(self):
        # SENTENCE_TRANSFORMERS_HOME="/app/backend/data/cache/embedding/models"
        #cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME") or "pipelines/cache/embedding/models"
        cache_dir = f"{os.path.dirname(__file__)}/dawson_pipeline/cache/embedding/models"
        model = "sentence-transformers" + "/" + self.valves.EMBEDDING_MODEL
        model_repo_path = f"{cache_dir}/models--sentence-transformers--{self.valves.EMBEDDING_MODEL}"
        if os.path.exists(model_repo_path):
            return model_repo_path
        local_files_only = True
        snapshot_kwargs = {
            "cache_dir": cache_dir,
            "local_files_only": False,
            "repo_id": model,
        }
        # Attempt to query the huggingface_hub library to determine the local path and/or to update
        try:
            model_repo_path = snapshot_download(**snapshot_kwargs)
            log.debug(f"model_repo_path: {model_repo_path}")
            return model_repo_path
        except Exception as e:
            log.exception(f"Cannot determine model snapshot path: {e}")
            return model
        pass