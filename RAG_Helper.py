import os
import glob
from pathlib import Path
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

class RAGHelper:
    def __init__(self, folder: str, chunk_size=300, chunk_overlap=50):
        self.folder = folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.retrieval_chain = None

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

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )
        return splitter.split_documents(documents)

    def _build_vectorstore(self, documents: List[Document]):
        print(f"📦 建立向量資料庫，共 {len(documents)} 段")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.vectorstore.save_local("my_faiss_index")

    async def load_and_prepare(self, file_extensions: List[str] = None):
        print("📂 開始載入檔案...")

        if os.path.exists("my_faiss_index"):
            print("🔁 偵測到已存在向量資料庫，直接載入")
            self.vectorstore = FAISS.load_local(
                "my_faiss_index",
                OpenAIEmbeddings(model="text-embedding-3-small"),
                allow_dangerous_deserialization=True
            )
            return

        if file_extensions is None:
            file_extensions = ['.pdf']

        all_chunks = []
        for ext in file_extensions:
            pattern = f"*{ext}"
            for path in glob.glob(os.path.join(self.folder, pattern)):
                try:
                    print(f"📄 讀取中: {os.path.basename(path)}")
                    pages = await self.load_any_file_async(path)
                    chunks = self._split_documents(pages)
                    all_chunks.extend(chunks)
                    print(f"✅ 分割完成，共 {len(chunks)} 段")
                except Exception as e:
                    print(f"❌ 錯誤讀取 {os.path.basename(path)}: {e}")

        if not all_chunks:
            raise ValueError("❌ 沒有成功載入任何文件")
        self._build_vectorstore(all_chunks)

    def setup_retrieval_chain(self, short_context=False):
        if not self.vectorstore:
            raise RuntimeError("❗ 請先執行 load_and_prepare()")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3 if short_context else 5})

        system_prompt = (
            "你是一個問答助手。基於以下提供的內容來回答問題。"
            "如果內容中沒有相關資訊，請說「根據提供的資料無法回答這個問題」。"
            f"請用繁體中文{'簡潔' if short_context else ''}回答。\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        chain = create_stuff_documents_chain(llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, chain)

    def ask(self, query: str) -> Tuple[str, List[Document]]:
        if not self.retrieval_chain:
            raise RuntimeError("❗ 尚未初始化問答鏈")

        try:
            result = self.retrieval_chain.invoke({"input": query})
            return result["answer"], result["context"]
        except Exception as e:
            if "max_tokens_per_request" in str(e):
                print("⚠️ 上下文太長，使用縮短版問答鏈")
                self.setup_retrieval_chain(short_context=True)
                result = self.retrieval_chain.invoke({"input": query})
                return result["answer"], result["context"]
            raise e
