from flask import Flask, render_template, request
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, readers
from langchain import OpenAI
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the LLM
llm_predictor = LLMPredictor(llm=OpenAI(
    temperature=0, model_name="text-davinci-003"))

# Define prompt_helper and settings
max_input_size = 4096
num_outputs = 256
max_chunk_overlap = 20
chunk_size_limit = 600
prompt_helper = PromptHelper(
    max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit)

# Load data to train the model
directory_path = './data'
documents = SimpleDirectoryReader(directory_path).load_data()

# Create the index from the data
index = GPTSimpleVectorIndex(
    documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
index.save_to_disk('index')

# Predict function to take query and generate the response from chat_rugby()

# Flask app
app = Flask(__name__)
if __name__ == '__main__':
    app.run(debug=True)

# Endpoint for form submission


@app.route('/predict', methods=['POST'])
def predict():
    query = request.form['query']
    response = chat_rugby(query)
    return render_template('index.html', query=query, response=response)

# Function to generate the response


def chat_rugby(query):
    index = GPTSimpleVectorIndex.load_from_disk('index')
    response = index.query(query, response_mode="compact")
    return response.response


@app.route('/', methods=['POST'])
def index():
    return render_template('./index.html')
