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
    user_input = request.json.get("question")
    try:
        answer, context = rag.ask(user_input)
        sources = []
        for doc in context:
            source = doc.metadata.get("source", "未知檔案")
            page = doc.metadata.get("page", "未知頁碼")
            content = doc.page_content.strip().replace("\n", " ")
            sources.append(f"<li><b>{source}（第 {page} 頁）</b>：{content}</li>")
        sources_html = "<ul>" + "".join(sources) + "</ul>"
        return jsonify({
            "answer": answer,
            "sources": sources_html
        })
    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


