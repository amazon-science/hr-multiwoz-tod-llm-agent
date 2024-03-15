import pandas as pd
import pickle
import numpy as np
#ask questions every time and make the 
from transformers import  pipeline


def is_perfect_match(text1, text2, max_edits=5):
    """
    Checks if two texts can be made to perfectly match with at most `max_edits` edits.
    
    Args:
        text1 (str): The first text.
        text2 (str): The second text.
        max_edits (int, optional): The maximum number of edits allowed. Defaults to 5.
        
    Returns:
        bool: True if the two texts can be made to perfectly match with the given edits, False otherwise.
    """
    edits = 0
    i, j = 0, 0
    
    while i < len(text1) and j < len(text2):
        if text1[i] != text2[j]:
            edits += 1
            if edits > max_edits:
                return False
            if text1[i] == ' ' or text2[j] == ' ':
                if text1[i] == ' ':
                    i += 1
                if text2[j] == ' ':
                    j += 1
            else:
                i += 1
                j += 1
        else:
            i += 1
            j += 1
    
    edits += abs(len(text1) - len(text2))
    return edits <= max_edits
name = "deepset/deberta-v3-large-squad2"
question_answerer = pipeline("question-answering", model=name)
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




with open('sgd_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
with open('template.pkl', 'rb') as f:
    template = pickle.load(f)


JAG_v = 0
JAG_d = 0
SA_v = 0
SA_d = 0


for indx in range(data.shape[0]):
    types, convs, sgds = data.iloc[indx].values[1], data.iloc[indx].values[4], data.iloc[indx].values[5]
    sgd = []
    awr = {}
    match = []
    match_num = 0

    for index in range(len(sgds)):
        
        
        for entity in template[types[0]]:
            context = convs[index * 2 + 1]
            #print(context)
            question = template[types[0]][entity][0]
            
            extract = entity_extract(question, context[10:], thred = 0.2)
            if extract != 'I do not know' and entity not in awr:
                awr[entity] = extract
        sgd.append(awr.copy())
        for key in sgds[index]:
            if key in awr and is_perfect_match(awr[key], sgds[index][key], max_edits=5): 
                match_num += 1
            
        match.append(match_num) 
    cumulative_sum = 0
    new_list = []
    for num in [len(key) for key in sgds ]:
        cumulative_sum += num
        new_list.append(cumulative_sum)

    SA_v += match[-1]
    SA_d += new_list[-1]
    JAG_d += len(new_list)
    JAG_v += np.sum([v1 == v2 for v1, v2 in zip(match, new_list)])

print(SA_v, SA_d, JAG_d, JAG_v)  
print("Slot accuracy " + str(SA_v / SA_d))
print("JAG accuracy" + str(JAG_v / JAG_d))