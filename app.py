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
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8080))  # 預設 8080，但 Render 會傳自己的 PORT
    uvicorn.run(app, host="0.0.0.0", port=port)

