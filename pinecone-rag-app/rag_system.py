import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

def main():
    # 1. Configuration Check
    google_key = os.getenv("GOOGLE_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not all([google_key, pinecone_key, index_name]):
        print("Error: Missing API keys or Index Name in .env file.")
        return

    # 2. Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_key)

    # 3. Create Index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=768, # Dimension for Google text-embedding-004
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    else:
        print(f"Index '{index_name}' already exists.")

    # 4. Load and Split Document
    print("Loading and splitting document...")
    loader = TextLoader("knowledge_base.txt")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(data)

    # 5. Embed and Index in Pinecone
    print("Embedding and indexing documents...")
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768,   )
    
    # This will both embed and upload the documents
    vectorstore = PineconeVectorStore.from_documents(
        splits, 
        embeddings, 
        index_name=index_name
    )

    # 6. Setup Retrieval Chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Ask Questions
    print("\n--- RAG System Ready ---")
    query = "What is the primary objective of the AREP LAB04?"
    print(f"Question: {query}")
    
    response = qa_chain.invoke(query)
    
    print("\nAnswer:")
    print(response)

if __name__ == "__main__":
    main()
