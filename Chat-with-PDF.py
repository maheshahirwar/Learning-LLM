from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Load PDF document
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100 # This means that each chunk will have 100 characters of overlap with the previous chunk to maintain context
)

docs = text_splitter.split_documents(documents)

# Embeddings and vector store

embeddings = OllamaEmbeddings(model="llama3")

vector_store = FAISS.from_documents(docs, embeddings)

# Now we can perform similarity search on the vector store to find relevant chunks of text based on a query

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# LLM
llm = ChatOllama(model="llama3")

# Prompt template
template = """
    You are an AI assistant.
    Answer ONLY from the provided context.
    If the answer is not in context, say:
    "Information not found in document."

    Context:
    {context}

    Question:
    {question}

    Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# User query
query = "What encryption is used?"

# Retrieve relevant docs
relevant_docs = retriever.invoke(query)

# Combine context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Construct final prompt
final_prompt = prompt.format(
    context=context,
    question=query
)

print("Final Prompt:\n", final_prompt)

# Call LLM
response = llm.invoke(final_prompt)

print(response.content)