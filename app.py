# app.py
import os
from flask import Flask, render_template, request, jsonify
import asyncio
from RAG_Helper import RAGHelper
import webbrowser
import threading
app = Flask(__name__)
rag = RAGHelper(folder_path="pdfFiles")


# 初始化知識庫與檢索鏈
asyncio.run(rag.load_and_prepare(file_extensions=[".pdf", ".txt", ".docx"]))
rag.setup_retrieval_chain()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question")
        if not question:
            return jsonify({"error": "沒有輸入問題"}), 400

        answer, sources = rag.ask(question)

        source_info = []
        for doc in sources:
            content = doc.page_content.strip()[:100].replace("\n", "") + "…"  # 取前100字
            filename = os.path.basename(doc.metadata.get("source", "未知來源"))
            page = doc.metadata.get("page", "未知頁碼")
            source_info.append(f"{filename}（第 {page} 頁）：{content}")

        return jsonify({
            "answer": answer,
            "context": source_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


