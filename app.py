from flask import Flask, render_template, request, jsonify
import ollama

app = Flask(__name__)

MODEL = "llama3.2:latest"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():

    data = request.get_json()

    customer_name = data.get("customer_name", "")
    subject = data.get("subject", "")
    email = data.get("email", "")
    tone = data.get("tone", "Friendly")

    # Step 1: Detect customer intent
    intent_prompt = f"""
Analyze the following customer email and identify its main intent.

Subject: {subject}
Email: {email}

Return ONLY the main intent in 2-5 words.
"""

    intent_response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": intent_prompt}
        ]
    )

    intent = intent_response["message"]["content"].strip()

    # Step 2: Generate customer reply
    reply_prompt = f"""
You are an intelligent customer support email assistant.

Customer Name: {customer_name}
Subject: {subject}
Customer Email: {email}
Detected Intent: {intent}

Write a professional and context-aware reply to the customer.

Requirements:
- Address the customer by name.
- Clearly respond to their issue.
- Be polite and helpful.
- Do not invent order numbers, dates, refunds, or other facts.
- Keep the reply concise.
- Do not explain your reasoning.
"""

    reply_response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": reply_prompt}
        ]
    )

    generated_reply = reply_response["message"]["content"].strip()

    # Step 3: Improve reply according to selected tone
    tone_prompt = f"""
Rewrite the following customer support reply in a {tone} tone.

Keep the meaning and important information unchanged.
Do not add imaginary information.
Return ONLY the improved email.

Reply:
{generated_reply}
"""

    tone_response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": tone_prompt}
        ]
    )

    improved_reply = tone_response["message"]["content"].strip()

    # Step 4: Generate short summary
    summary_prompt = f"""
Summarize this customer email conversation in approximately 50 words.

Customer Email:
{email}

Generated Reply:
{improved_reply}

Return ONLY the summary.
"""

    summary_response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": summary_prompt}
        ]
    )

    summary = summary_response["message"]["content"].strip()

    return jsonify({
        "intent": intent,
        "similar_email": "Previous email retrieval will be added next.",
        "previous_reply": "Previous reply retrieval will be added next.",
        "generated_reply": generated_reply,
        "improved_reply": improved_reply,
        "summary": summary
    })


if __name__ == "__main__":
    app.run(debug=True)