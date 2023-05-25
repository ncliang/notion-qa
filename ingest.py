"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Here we load in the data in the format that Notion exports it in.
# ps = list(Path("Notion_DB/").glob("**/*.md"))
ps = list(Path("ctbc/").glob("**/*.txt"))
documents = []
for p in ps:
    loader = TextLoader(p)
    documents.extend(loader.load())


# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# Here we create a vector store from the documents and save it to disk.
db = FAISS.from_documents(docs, embeddings)

db.save_local("docs.index")

