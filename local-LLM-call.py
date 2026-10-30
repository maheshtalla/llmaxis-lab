
#calling the local LLM using ollama library
#Import Library
import ollama

#Invoked Local LLM

def call_llm(prompt):
    response = ollama.generate(
            model ="tinyllama",
            prompt = prompt
                    )
    return response["response"]

user_input =input("Enter your instruction :")
print("**********User Input***********")
print(user_input)

result =call_llm(user_input)
print ("\n AI Response \n")
print(result)


#calling the local LLM using REST API

'''
import requests

def call_llm(prompt):
    url ="http://localhost:11434/api/generate"

    payload = {
        "model":"llama3.2:1b",
        "prompt":prompt,
        "stream":False
    }

    response= requests.post(url,json=payload)
    return response.json()["response"]

user_input =input("Enter your instruction :")
print("**********User Input***********")
print(user_input)

result =call_llm(user_input)
print ("\n AI Response \n")
print(result)
'''