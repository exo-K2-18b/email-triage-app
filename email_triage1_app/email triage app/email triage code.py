# imports - just tools we need
from flask import Flask, request, jsonify, render_template
from groq import Groq
from dotenv import load_dotenv
import os
import json

# load API key from .env file
load_dotenv()

# start Flask
app = Flask(__name__)

# connect to Groq - YOUR code, you know this!
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# YOUR AI function - you already wrote this!
def analyze_email(email_text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an email triage assistant.
                Analyze emails and respond ONLY in this exact JSON format:
                {
                    "priority": "urgent/normal/low",
                    "category": "personal/work/newsletter/spam",
                    "summary": "one line summary",
                    "action": "reply/delete/read later/forward",
                    "reason": "one line explaining your decision"
                }"""
            },
            {
                "role": "user",
                "content": f"Analyze this email:\n\n{email_text}"
            }
        ]
    )
    return json.loads(response.choices[0].message.content)

# route 1 - show the homepage
@app.route("/")
def home():
    return render_template("index.html")  # just loads the HTML file

# route 2 - analyze the email
@app.route("/analyze", methods=["POST"])  # POST means receiving data
def analyze():
    data = request.get_json()        # get data from browser
    email_text = data["email"]       # extract email text
    result = analyze_email(email_text)  # run YOUR function
    return jsonify(result)           # send result back to browser

# start the server
if __name__ == "__main__":
    app.run(debug=True)