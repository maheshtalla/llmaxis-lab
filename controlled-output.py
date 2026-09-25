from openai import OpenAI
from prompt_toolkit import prompt


client = OpenAI(api_key="")

import json
from openai import OpenAI

client = OpenAI()

# --- LLM CALL ---
def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", 
             "content": "You MUST return ONLY valid JSON. No explanation. No markdown."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# --- BUILD PROMPT ---
def process_text(text):
    prompt = f"""
Return ONLY valid JSON in the following format:

{{
  "summary": "",
  "risks": "",
  "recommendations": ""
}}

Text to analyze: {text}
"""
    return call_llm(prompt)


# --- SAFE JSON PARSER WITH AUTO-REPAIR ---
def safe_parse(json_text):
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Attempt auto-repair by stripping non-JSON content
        try:
            cleaned = json_text.strip()
            cleaned = cleaned.replace("```json", "").replace("```", "")
            return json.loads(cleaned)
        except:
            return None


# --- MAIN EXECUTION ---
user_input = input("What you want to know about: ")

result = process_text(user_input)

print("\nRaw Output:\n")
print(result)

data = safe_parse(result)

if data is None:
    print("\nInvalid JSON - cannot proceed safely")
else:
    print("\nValid Data:")
    print(data)
