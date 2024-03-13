# select relevant course
import json
import pickle
import random

import boto3
import gradio as gr
import pandas as pd
import PyPDF2
from fpdf import FPDF

bedrock = boto3.client(
    service_name="bedrock",
    region_name="us-east-1",
    endpoint_url="https://bedrock.us-east-1.amazonaws.com",
)


with open("~/workplace/M2MHR/data/template.pkl", "rb") as f:
    templates = pickle.load(f)

with open("~/workplace/M2MHR/data/generated_dictionary_step3.pkl", "rb") as f:
    loaded_dict = pickle.load(f)

with open("~/workplace/M2MHR/data/generated_dictionary_step4.pkl", "rb") as f:
    step4_data = pickle.load(f)


# step4_data = {}
def conversation_completion(conversation):
    prompt_data = """

Human:      
            Conversation: {conversation}
            This is the conversation between HR Assistant and an employee. Can you rewrite the conversation based on the following instructions:

                1. For each Question, paraphrase the question to make it more conversational by using more modal words and empathic.
                2. For each Answer, write it a as a complete sentence. 

            Please put the updated Conversation based on Template in <answer></answer>  XML tags. 

Assistant:
        """.format(
        conversation=conversation
    )

    body = json.dumps(
        {
            "prompt": prompt_data,
            "max_tokens_to_sample": 1000,
            "temperature": 0.5,
            "stop_sequences": ["</answer>"],
        }
    )
    modelId = "anthropic.claude-instant-v1"  # change this to use a different version from the model provider
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    result = response_body.get("completion")

    start = result.index("<answer>")

    return result[start + 8 :]


start = max(step4_data.keys())
for ind in range(start, len(loaded_dict)):
    conversations = []
    for key in templates[loaded_dict[ind]["template"]]:
        if key in loaded_dict[ind]:
            conversation = []
            conversation.append(key)
            conversation.append(templates[loaded_dict[ind]["template"]][key][0])
            conversation.append(loaded_dict[ind][key])
        conversations.append(conversation)
    random.shuffle(conversations)
    # Generate indedx list for a bunch of questions
    index = 0
    index_list = [0]
    while index < len(conversations) - 2:
        rand = random.randint(1, 2)
        index += rand
        index_list.append(index)
    index_list.append(len(conversations))
    new_conversations = []
    for ind1, ind2 in zip(index_list, index_list[1:]):
        new_conversations.append(
            [
                [val[0] for val in conversations[ind1:ind2]],
                " ".join([str(val[1]) for val in conversations[ind1:ind2]]),
                ", ".join([str(val[2]) for val in conversations[ind1:ind2]]),
            ]
        )

    inputs = [[i[1], i[2]] for i in new_conversations]
    converat = "HR Assistant: Hey, How do you want me to help today? \n"
    converat += "Employee: I need help with " + loaded_dict[ind]["template"] + "\n"
    for conv in inputs:
        converat += "HR Assistant: " + conv[0] + "\n"
        converat += "Employee: " + str(conv[1]) + "\n"

    r = conversation_completion(converat)
    step4_data[ind] = [new_conversations, r]
    if ind % 10 == 0:
        with open(
            "~/workplace/M2MHR/data/generated_dictionary_step4.pkl",
            "wb",
        ) as f:
            pickle.dump(step4_data, f)
