import os
from typing import List, Tuple
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

class RAGHelper:
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
        self.vectorstore = None
        self.qa_chain = None

    async def load_and_prepare(self, file_extensions: List[str]) -> None:
        all_docs = []

        for filename in os.listdir(self.pdf_folder):
            ext = os.path.splitext(filename)[-1].lower()
            filepath = os.path.join(self.pdf_folder, filename)

            if ext not in file_extensions:
                continue

            if ext == ".pdf":
                loader = PyPDFLoader(filepath)
            elif ext == ".txt":
                loader = TextLoader(filepath, encoding="utf-8")
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(filepath)
            else:
                continue

            try:
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

        if not all_docs:
            raise ValueError("❌ 沒有成功載入任何文件")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        documents = splitter.split_documents(all_docs)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma.from_documents(documents, embeddings)

    def setup_retrieval_chain(self):
        if not self.vectorstore:
            raise RuntimeError("❌ 請先執行 load_and_prepare() 載入向量資料庫")
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    def ask(self, query: str) -> Tuple[str, dict]:
        if not self.qa_chain:
            raise RuntimeError("❌ QA chain 尚未初始化，請先呼叫 setup_retrieval_chain()")
        result = self.qa_chain({"query": query})
        return result["result"], result
