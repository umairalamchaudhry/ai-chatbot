from flask import Flask, request, render_template, jsonify
import json
import asyncio
import aiohttp
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Load the knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Cache for frequently asked questions
response_cache = {}


async def fetch_ollama_response(session, prompt):
    """Fetch a response from the Ollama API asynchronously."""
    payload = {
        "model": "hf.co/bartowski/Dolphin3.0-Llama3.1-8B-GGUF:latest",  # Using the Dolphin3 model
        "prompt": prompt,
        "stream": False,
    }
    async with session.post(OLLAMA_API_URL, json=payload) as response:
        if response.status == 200:
            data = await response.json()
            return data["response"]
        else:
            return "Sorry, I couldn't generate a response. Please try again later."


def generate_response_with_ollama(user_query, language="en"):
    """Generate a response using Ollama, referencing the knowledge base."""
    # Check if the response is already cached
    if user_query in response_cache:
        return response_cache[user_query]

    # Create a prompt that includes the knowledge base
    prompt = f"""
    You are a helpful e-commerce customer support chatbot. Your task is to answer user queries based on the following knowledge base. If the answer is not in the knowledge base, generate a response on your own.

    Knowledge Base:
    {json.dumps(knowledge_base, indent=2)}

    User Query: {user_query}
    """

    # Fetch the response asynchronously
    async def get_response():
        async with aiohttp.ClientSession() as session:
            response = await fetch_ollama_response(session, prompt)
            if language == "ur":
                response = GoogleTranslator(source="auto", target="ur").translate(
                    response
                )
            response_cache[user_query] = response  # Cache the response
            return response

    # Run the async function synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(get_response())
    loop.close()

    return response


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.form["user_query"]
    language = request.form.get("language", "en")  # Default to English

    # Generate a response using Ollama, which references the knowledge base
    response = generate_response_with_ollama(user_query, language)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
