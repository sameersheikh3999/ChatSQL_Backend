from flask import Flask, request, jsonify 
from flask_cors import CORS
# from chatbot import query_engine, load_memory, save_memory  # import memory funcs
from workinchatbot import query_engine, load_memory, save_memory, ask_query, is_correction

app = Flask(__name__)
CORS(app)

memory = load_memory()  # global memory list

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("message", "")

    try:
        response = query_engine.query(question)
        memory.append({
            "user": question,
            "ChatSQL": response.response
        })
        save_memory(memory)

        return jsonify({
            "sql": response.metadata.get("sql_query"),
            "response": response.response
        })
    except Exception as e:
        print("Error during query:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)


