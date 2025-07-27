import glob
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)


class RAGHelper:
    def __init__(self, folder_path, chunk_size=150, chunk_overlap=30):
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.retrieval_chain = None
        self.persist_dir = "chroma_db"

    def get_loader(self, path: str):
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(path)
        elif ext == ".txt":
            return TextLoader(path, encoding="utf-8")
        elif ext == ".docx":
            return UnstructuredWordDocumentLoader(path)
        elif ext == ".md":
            return UnstructuredMarkdownLoader(path)
        elif ext == ".csv":
            return CSVLoader(path)
        else:
            raise ValueError(f"不支援的檔案類型: {ext}")

    async def load_any_file_async(self, path: str):
        loader = self.get_loader(path)
        if hasattr(loader, "alazy_load"):
            pages = []
            async for page in loader.alazy_load():
                pages.append(page)
            return pages
        else:
            return loader.load()

    def _split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )
        return splitter.split_documents(documents)

    def _build_vectorstore(self, documents):
        print(f"建立向量資料庫，共 {len(documents)} 段文字")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_dir
        )
        self.vectorstore.persist()

    async def load_and_prepare(self, file_extensions=None):
        print("開始載入資料並建立向量資料庫...")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        if os.path.exists(self.persist_dir):
            print("偵測到現有向量庫，載入中...")
            self.vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=self.persist_dir
            )
            return

        if file_extensions is None:
            file_extensions = [".pdf"]

        all_chunks = []

        for ext in file_extensions:
            pattern = f"*{ext}"
            file_paths = glob.glob(os.path.join(self.folder_path, pattern))

            for path in file_paths:
                try:
                    print(f"讀取檔案：{os.path.basename(path)}")
                    pages = await self.load_any_file_async(path)
                    chunks = self._split_documents(pages)
                    all_chunks.extend(chunks)
                    print(f"{os.path.basename(path)} 分割為 {len(chunks)} 段")
                except Exception as e:
                    print(f"讀取 {os.path.basename(path)} 發生錯誤: {e}")

        if not all_chunks:
            raise ValueError("沒有任何文件被成功載入")

        self._build_vectorstore(all_chunks)

    def setup_retrieval_chain(self):
        if not self.vectorstore:
            raise ValueError("請先執行 load_and_prepare()")

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一個問答助手，根據以下內容回答問題。\n\n{context}"),
            ("human", "{input}")
        ])
        qa_chain = create_stuff_documents_chain(llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, qa_chain)

    def ask(self, query):
        if not self.retrieval_chain:
            raise ValueError("請先執行 setup_retrieval_chain()")
        result = self.retrieval_chain.invoke({"input": query})
        return result["answer"], result.get("context", [])
