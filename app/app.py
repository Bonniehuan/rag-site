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
    user_input = request.json.get("question")  # <== 注意：前端要傳 "question"
    try:
        print(f"[使用者問題]：{user_input}")
        answer, context = rag.ask(user_input)
        sources = "\n".join([doc.page_content for doc in context.get("source_documents", [])])
        return jsonify({
            "answer": answer,
            "context": sources
        })
    except Exception as e:
        return jsonify({ "answer": None, "error": str(e) })

if __name__ == "__main__":
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000")  # ✅ 你本地的網址

    threading.Timer(1.0, open_browser).start()  # 延遲一秒啟動瀏覽器
    app.run(debug=True)
