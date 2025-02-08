from flask import Flask, request, jsonify
from openai_agent import Agent
app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Missing required fields: message"}), 400

    message = data['message']

    try:
        response = agent.chat(message)

        return jsonify({
            "role": "assistant",
            "content": str(response)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_chat', methods=['POST'])
def clear_chat():
    result = agent.clear_chat()
    return jsonify(result)

if __name__ == '__main__':
    agent = Agent()
    app.run(host='0.0.0.0', port=5556)
