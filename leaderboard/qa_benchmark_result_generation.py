from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import pandas as pd
import pickle
import collections
from transformers import  pipeline

# Function to answer questions based on the model name, question, and context
def answer_question(tokenizer, model, question, context):

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


def entity_extract(question, context, thred = 0):

        QA_input = {
                'question': question,
                'context': context
                }
        r = question_answerer(QA_input)
        if r['score'] < thred:
            return 'I do not know'
        else:
            return r['answer']

# Load your QA dataset
# with open('qa_dataset.pkl', 'rb') as f:
#     data = pickle.load(f)
data = pd.read_csv('qa_dataset_result.csv')
# Prepare to store the results
result = collections.defaultdict(list)

# List of models to use for answering questions
#model_names = ["bert-base-uncased", "distilbert-base-uncased", "deepset/roberta-base-squad2", "albert-base-v2", "google/electra-small-discriminator", "xlnet-base-cased"]
model_names = ["deepset/deberta-v3-large-squad2", "timpal0l/mdeberta-v3-base-squad2", "distilbert/distilbert-base-cased-distilled-squad","bert-large-uncased-whole-word-masking-finetuned-squad" ]
# Iterate over each model and each question-context pair
for name in model_names:
    if False:
        print(f"Processing with model: {name}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForQuestionAnswering.from_pretrained(name).to(device)
    else:
        question_answerer = pipeline("question-answering", model=name)
    for q, a in zip(data.question, data.answer_context):
        if False:
            try:
                    # Setup for device. Use GPU if available, otherwise fallback to CPU
                            # Make sure to pass 'name' to use the correct model for each iteration
                prediction = answer_question(tokenizer, model, q, a)
                result[name].append(prediction)
            except Exception as e:
                print(f"Error processing with model {name}: {e}")
                # Optionally, append a placeholder or error message to keep track
                result[name].append("Error processing question")
        else:
            prediction = entity_extract(q, a)
            result[name].append(prediction)

for name in model_names:
    data[name.split('/')[-1] + '_result'] = result[name]
data.to_csv('qa_dataset_result.csv')