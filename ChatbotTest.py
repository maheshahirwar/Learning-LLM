from langchain_core.messages import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_ollama import ChatOllama, OllamaLLM

llm = OllamaLLM(model="llama3")

# response = llm.invoke("Explain AI in simple words")
# print(response)

# Instantiate local Llama3 model
chat = ChatOllama(
    model="llama3",
    temperature=0
)

# response = chat.invoke("Explain AI in simple words")
# response = chat.invoke(
#    [
#         SystemMessage(content="You are an international chef that specializes in making sandwiches."),
#         HumanMessage(content="I dont like tomatoes, what else can make me a sandwich? Give a 2 line recipe.")
#     ]
# )

response = chat.invoke(
    [
        SystemMessage(content="You are a dumb AI bot that is unhelpful and makes jokes at whatever the user says"),
        HumanMessage(content="how do I navigate maps?")
    ]
)
print(response.content)
