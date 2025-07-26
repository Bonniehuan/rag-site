# app.py
from flask import Flask, render_template, request, jsonify
import asyncio
from RAG_Helper import RAGHelper
import webbrowser
import threading
app = Flask(__name__)
rag = RAGHelper(pdf_folder="pdfFiles")

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
        # 截斷來源內容為前 100 字
        context_texts = [doc.page_content.strip()[:100] + "…" for doc in sources]
        context_combined = "\n\n".join(context_texts)

        return jsonify({"answer": answer, "context": context_combined})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


