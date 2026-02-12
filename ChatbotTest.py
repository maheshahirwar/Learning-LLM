from langchain_core.messages import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS

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


# response = chat.invoke(
#     [
#         SystemMessage(content="You are a dumb AI bot that is unhelpful and makes jokes at whatever the user says"),
#         HumanMessage(content="how do I navigate maps?")
#     ]
# )

# response = chat.invoke(
#    [
#         SystemMessage(content="You are an intelligent AI bot that tries to help the the user in every way possible."),
#         HumanMessage(content="how do I navigate maps? explain in 2-3 lines")
#     ]
# )
# print(response.content)



# embeddings = OllamaEmbeddings(model="llama3")
# query_result = embeddings.embed_query("What is the capital of France?")
# print(query_result)


# Prompt template example

# template = """
# I want to be {career_option} in future. What subjects should I start studying?
# Respond in 1-2 short sentence
# """
# # Creating a template from the above prompt
# prompt = PromptTemplate(
#     input_variables=["career_option"],
#     template=template
# )
# final_prompt = prompt.format(career_option='Data Scientist')
# print (f"Final Prompt: {final_prompt}")
# response = chat.invoke(final_prompt)
# print(response.content)



# Selector example

prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\nExample Output: {output}",
)

examples = [
    {"input": "software engineer", "output": "software development"},
    {"input": "accountant", "output": "accounting"},
    {"input": "teacher", "output": "education"},
    {"input": "doctor", "output": "medicine"},
    {"input": "architect", "output": "architecture"},
    {"input": "lawyer", "output": "law"},
]

selector = SemanticSimilarityExampleSelector.from_examples(examples, # list of examples available to select from.
                                                           OllamaEmbeddings(model="llama3"), # embedding class used to produce embeddings which are used to measure semantic similarity.
                                                           FAISS, # VectorStore class that is used to store the embeddings and do a similarity search over.
                                                           k=2  # number of examples to produce.
                                                          )

similar_prompt = FewShotPromptTemplate(example_selector=selector, # The object that will help select examples
									   example_prompt=prompt,  # Your prompt
    								   prefix="Give the job title their job role is ", # Customizations that will be added to the top and bottom of your prompt
    								   suffix="Input: {job_title}\nOutput:",
    								   input_variables=["job_title"] # What inputs your prompt will receive
    								   )
title = "nurse"
print(similar_prompt.format(job_title=title))
print(llm.invoke(similar_prompt.format(job_title=title)))