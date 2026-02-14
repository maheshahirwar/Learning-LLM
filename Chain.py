from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Instantiate local Llama3 model
llm = OllamaLLM(model="llama3")

# Prompt template that take user input directly
template = """Your task is to find the movie name which won Academy Awards / Best Picture in the year in India that user suggests.
% YEAR
{year}
YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["year"], template=template)

# Create a chain that combines the prompt template and the LLM
year_chain = prompt_template | llm

# Prompt template that will consume output of the previous LLM chain
template = """Given the movie name, give a short summary of the plot of that movie.
% MOVIE_NAME
{movie_name}
YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["movie_name"], template=template)


# Create a chain that combines the prompt template and the LLM
plot_chain = prompt_template | llm


# Now we can combine the two chains together to create a single chain that takes user input and produces the final output
full_chain = year_chain | plot_chain

# Now we can run the full chain with user input
user_input = "2000"
print("User input:", user_input)
final_output = full_chain.invoke(user_input)
print(final_output)
