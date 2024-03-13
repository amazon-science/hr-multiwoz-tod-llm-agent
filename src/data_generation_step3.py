import gradio as gr

# select relevant course
import json
import boto3
import PyPDF2
from fpdf import FPDF
import pandas as pd
import pickle
import random
import ast

bedrock = boto3.client(
    service_name="bedrock",
    region_name="us-east-1",
    endpoint_url="https://bedrock.us-east-1.amazonaws.com",
)


def dictionary_generation(user, template):
    prompt_data = """

Human:      
            User: {user}
            Template: {template}

            
            You are User.
            Can you fill out all the question in templates based on your experience?
            The generated dictionary should only contain key name and generated answer for that key. All keys are from Template are in generated dictionary. 
            Please make the answer extremely short (try to answer it within 5 words)
            Please put the generated dictionary in <answer></answer>  XML tags. 

Assistant:
        """.format(
        user=user, template=template
    )

    body = json.dumps(
        {
            "prompt": prompt_data,
            "max_tokens_to_sample": 500,
            "temperature": 0.5,
            "stop_sequences": ["</answer>"],
        }
    )
    modelId = "anthropic.claude-v2:1"  # change this to use a different version from the model provider
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    result = response_body.get("completion")

    start = result.index("<answer>")

    return result[start + 8 :]


with open("~/workplace/M2MHR/data/user.pkl", "rb") as f:
    users = pickle.load(f)

with open("~/workplace/M2MHR/data/template.pkl", "rb") as f:
    templates = pickle.load(f)
results = {}
for i in range(1000):
    template_keys = list(templates.keys())
    template = random.choice(template_keys)
    user = random.choice(users)
    results[i] = ast.literal_eval(dictionary_generation(user, templates[template]))
    # print(results[i])
    results[i]["template"] = template
    if i % 10 == 0:
        with open(
            "~/workplace/M2MHR/data/generated_dictionary_step3.pkl",
            "wb",
        ) as f:
            pickle.dump(results, f)
