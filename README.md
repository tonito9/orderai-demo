# OrderAI demo

This project is a simple demo of an order processing tool using AI and built with Python and Streamlit over the weekend.


## To use the tool

To use the tool, you need to have:

- Pinecone vector index with embedded catalog items;

- Pinecone API key;

- OpenAI API key;

- if you want to use LlaVa 1.6, a Replicate API key.


You also need to create a .env at the root of the project with inside:

- OPENAI_API_KEY = [your OpenAI API key]

- PINECONE_API_KEY = [your Pinecone API key]

- PINECONE_INDEX = [the name of your Pinecone index]

- REPLICATE_API_TOKEN = [your Replicate API key]


Then, you can install the requirements, open a terminal, go to the project folder and enter "streamlit run OrderAI.py".

If you just want to have a look at the code, you will find in OrderAI.py the code related to the Streamlit UI and basic data manipulation, and in utils_ai.py the utility and AI-related functions.
