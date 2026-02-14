from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

# LLM
llm = ChatOllama(model="llama3.1", temperature=0)

# Free search tool
search = DuckDuckGoSearchRun()

tools = [search]

# Create agent
agent = create_agent(
    model=llm,
    tools=[search],
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Which football club did Cristiano Ronaldo play for in 2023?"}
    ]
})
print(response)
