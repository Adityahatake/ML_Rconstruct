import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def ingest_data():

    # Load environment variables
    load_dotenv()

    # 1. Load documents
    loader = DirectoryLoader(
        path="data",
        glob="**/*.txt"
    )
    documents = loader.load()

    # 2. Split documents
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = OpenAIEmbeddings()

    # 4. Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    print("Ingestion complete!")


if __name__ == "__main__":
    ingest_data()
