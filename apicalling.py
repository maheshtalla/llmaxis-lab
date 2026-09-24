from openai import OpenAI
from prompt_toolkit import prompt


client = OpenAI(api_key="yourkey")

#calling LLM API
def call_openai_api(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
#            {"role": "user", "content": "Hello! Can you tell me a joke?"}
        ]
    )
    return response.choices[0].message.content

#building a function to process text and return the response
def process_text(text):
    prompt = f"""
    give me the output in 3 bullet points {text} .
    """
    response = call_openai_api(prompt)
    return response

with open("input.txt", "r") as file:
    user_input = file.read()

result = process_text(user_input)
print(result)


