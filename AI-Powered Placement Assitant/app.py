import json
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load predefined responses from the JSON file
with open("internship_chatbot_conversations_only.json", "r") as file:
    predefined_responses = json.load(file)

# Load GPT-J model and tokenizer locally
model_name = "EleutherAI/gpt-j-6B"  # Better alternative to GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate responses using the GPT-J local model
def generate_response(prompt):
    try:
        # Encode user input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        outputs = model.generate(
            inputs["input_ids"],
            max_length=150,
            num_return_sequences=1,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check for parroting and refine response
        if prompt.lower() in response.lower():
            response = "Let me clarify that for you! I'm here to assist with a more specific response."
        
        return response
    except Exception as e:
        return f"An error occurred while generating a response: {str(e)}"

# Scraping functionality to fetch internship listings
def fetch_internships():
    url = "https://www.linkedin.com/jobs/internship-jobs?originalSubdomain=in"  # Replace with the correct URL
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract internship details
            internships = []
            listings = soup.find_all("div", class_="base-card__info")  # Update selector as per LinkedIn's structure
            for listing in listings[:5]:  # Fetch top 5 results
                title = listing.find("h3", class_="base-card__title").text.strip() if listing.find("h3", class_="base-card__title") else "N/A"
                company = listing.find("h4", class_="base-card__subtitle").text.strip() if listing.find("h4", class_="base-card__subtitle") else "N/A"
                location = listing.find("span", class_="job-search-card__location").text.strip() if listing.find("span", class_="job-search-card__location") else "N/A"

                internships.append({
                    "title": title,
                    "company": company,
                    "location": location
                })

            return internships
        else:
            return {"error": f"Failed to fetch data. Status code: {response.status_code}"}
    except Exception as e:
        return {"error": f"An error occurred during scraping: {str(e)}"}

# Chatbot route for handling queries
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "").strip()  # Handle NoneType and strip whitespace

        # Respond to greetings
        greetings = ["hello", "hi", "hey", "namaste", "greetings", "good morning", "good evening"]
        if user_message.lower() in greetings:
            return jsonify({"reply": "Hello! Welcome to the AI-Powered Placement Assistant. How can I assist you today?"})

        if not user_message:
            return jsonify({"reply": "Please provide a valid query!"})

        # Check predefined responses from the JSON file
        if user_message.lower() in predefined_responses:
            return jsonify({"reply": predefined_responses[user_message.lower()]})

        # Handle internship-related queries
        if "internships" in user_message.lower():
            data = fetch_internships()
            if isinstance(data, list):  # If scraping was successful
                reply = "Here are some internship opportunities:\n"
                for i, job in enumerate(data, 1):
                    reply += f"{i}. {job['title']} at {job['company']} (Location: {job['location']})\n"
            else:
                reply = data.get("error", "Unable to fetch internships at the moment.")
            return jsonify({"reply": reply})

        # Handle open-ended queries using the local GPT-J model
        bot_reply = generate_response(user_message)
        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"reply": f"An error occurred while processing your query: {str(e)}"})

# Route to serve the HTML template
@app.route("/")
def home():
    return render_template("index.html")

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)