from gpt_index.langchain_helpers.chatgpt import ChatGPTLLMPredictor
from flask import Flask, render_template, request, jsonify
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, GPTListIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# use ChatGPT [beta]

llm_predictor = ChatGPTLLMPredictor()

#
# Before ChatGPT API was released...
#
# Define the LLM
# llm_predictor = LLMPredictor(llm=OpenAI(
#     temperature=0, model_name="text-davinci-003"))


# Define prompt_helper and settings
max_input_size = 4096
num_outputs = 1
max_chunk_overlap = 20
embedding_limit = 1000
chunk_size_limit = 135
prompt_helper = PromptHelper(
    max_input_size, num_outputs, max_chunk_overlap, embedding_limit, chunk_size_limit)

# Load data
directory_path = './data'
documents = SimpleDirectoryReader(directory_path).load_data()

# Create the index from the data
index = GPTListIndex(
    documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
index.save_to_disk('index.json')

# Flask app
app = Flask(__name__)
if __name__ == '__main__':
    app.run(debug=True)


@app.route('/')
def index():
    return render_template('./index.html')

# Endpoint for form submission


@app.route('/predict', methods=['POST'])
# Predict function to take user query and generate the response from the index
def predict():
    query = request.json['query']
    index = GPTListIndex.load_from_disk('index.json')
    response = index.query(
        'For the Bishops Stortford team ' + query, similarity_top_k=2, mode="embedding")
    return jsonify({'response': response.response})
