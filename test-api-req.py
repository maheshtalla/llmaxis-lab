from openai import OpenAI

client = OpenAI(api_key="yourkey")

response = client.responses.create(
    model="gpt-6-astra",
    input="Write a one-sentence bedtime story about a unicorn.",
)

print(response.output_text)