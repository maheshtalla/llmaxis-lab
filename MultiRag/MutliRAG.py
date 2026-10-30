import os 
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, unstructuredWordDocumentLoader, unstructuredfileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


## Start Local LLM

llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
    verbose=True,
)
print ("local llm started successfully")

DATA_PATH = "MultiRag\Knowledge"
documents = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        pages = loader.load()
        documents.extend(loader.load())

    elif file.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

    elif file.endswith(".docx"):
        loader = unstructuredWordDocumentLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

    elif file.endswith(".md"):
        loader = unstructuredfileLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

    else:
        print(f"Unsupported file format: {file}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30
)
chunks = text_splitter.split_documents(documents)
print(f"Total Chunks created: {len(chunks)}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

## Vertor DB : Store the embeddings in vector database and retrieve the relevant embeddings based on user query

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./multirag_db")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("vector db created successfully")

prompt = ChatPromptTemplate.from_chain_type([
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the document provided."),
    ("system", "If the answer is not in the document, say 'I don't know'."),
    ("system", "{history}")
    ("system", "{history}"),
    ("system", "{context}"),
    ("human", "{input}"),
   
    ]
)

Chat_history = []
print("\n Welcome to the PDF Question Answering System!,(Type 'exit' to quit.\n")