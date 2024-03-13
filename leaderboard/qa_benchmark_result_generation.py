from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import pandas as pd
import pickle
import collections

# Function to answer questions based on the model name, question, and context
def answer_question(model_name, question, context):
    # Setup for device. Use GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the tokenizer and model using the specified model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    
    # Tokenize input text
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].tolist()[0]
    
    # Get model outputs
    with torch.no_grad():  # Inference mode, no backpropagation for faster execution
        outputs = model(**inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    
    # Identify the tokens with the highest 'start' and 'end' scores
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    # Convert tokens to the answer string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
    return answer

# Load your QA dataset
with open('qa_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Prepare to store the results
result = collections.defaultdict(list)

# List of models to use for answering questions
model_names = ["bert-base-uncased", "distilbert-base-uncased", "deepset/roberta-base-squad2", "albert-base-v2", "google/electra-small-discriminator", "xlnet-base-cased"]

# Iterate over each model and each question-context pair
for name in model_names:
    print(f"Processing with model: {name}")
    for q, a in zip(data.question, data.answer_context):
        try:
            # Make sure to pass 'name' to use the correct model for each iteration
            prediction = answer_question(name, q, a)
            result[name].append(prediction)
        except Exception as e:
            print(f"Error processing with model {name}: {e}")
            # Optionally, append a placeholder or error message to keep track
            result[name].append("Error processing question")

for name in model_names:
    data[name.split('/')[-1] + '_result'] = result[name]
data.to_csv('qa_dataset_result.csv')