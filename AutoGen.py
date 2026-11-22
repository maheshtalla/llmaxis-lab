from autogen import UserProxyAgent, AssistantAgent
from dotenv import load_dotenv
import os


load_dotenv()
os.environ["OPENAI_API_KEY"]="api_key"
api_key =os.getenv("OPENAI_API_KEY")

config_list=[
    {
        "model": "gpt-4o-mini",
        "api_key":api_key
    }
]

assistant =AssistantAgent(
    name="assistant",
    llm_config={"config_list":config_list}
)

user_proxy=UserProxyAgent(
    name="user",
    llm_config={"config_list":config_list},
    code_execution_config=False
)

user_proxy.initiate_chat(assistant,message='Anlayse and idea to bring up an execution plan to start a EV manufacturing at Tesla after detailed compititor analysis')


