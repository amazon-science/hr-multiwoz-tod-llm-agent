import os
import pickle

import pandas as pd

file_path = os.path.expanduser("~/workplace/M2MHR/data/generated_dictionary_step6.pkl")
with open(file_path, "rb") as file:
    dialogue_dict = pickle.load(file)

file_path = os.path.expanduser("~/workplace/M2MHR/data/template.pkl")
with open(file_path, "rb") as file:
    task_dict = pickle.load(file)

#### schema guided dialogue dataset

# get dialogue ids
dialogue_id_ls = list(dialogue_dict.keys())

# get slots to domains mapping
slots2domain_map = {}
for k in task_dict.keys():
    slots2domain_map[k] = set(task_dict[k].keys())

# get slots of dialogues
slots_set_ls = []
for k, d in dialogue_dict.items():
    slots_set = set()
    for turn in d[0]:
        slots_ls = turn[0]
        for s in slots_ls:
            slots_set.add(s)
    slots_set_ls.append(slots_set)

# get domains of dialogues
domains_ls_ls = []
for lookup_set in slots_set_ls:
    found_keys_ls = []
    for key, value_set in slots2domain_map.items():
        if value_set == lookup_set:
            found_keys_ls.append(key)
    domains_ls_ls.append(found_keys_ls)

# get length of dialogues
dialogues_len_ls = []
for k, d in dialogue_dict.items():
    dialogues_len_ls.append(len(d[0]))

# get list of turn ids
turn_id_ls_ls = [list(range(x * 2)) for x in dialogues_len_ls]

# get list of speaker ids
speaker_id_ls_ls = [[0, 1] * x for x in dialogues_len_ls]

# get list of utterances
utterances_ls_ls = []
for k, d in dialogue_dict.items():
    utterances_ls = []
    for turn in d[0]:
        utterances_ls.append(turn[3])
        utterances_ls.append(turn[4])
    utterances_ls_ls.append(utterances_ls)

# get state of dialogue
states_ls_ls_dict = []
for k, d in dialogue_dict.items():
    states_ls_dict = []
    for turn in d[0]:
        keys = turn[0]
        values = turn[-1]
        states_dict = {key: value for key, value in zip(keys, values)}
        states_ls_dict.append(states_dict)
    states_ls_ls_dict.append(states_ls_dict)

# get schema guided dialogue dataset
sgd_dataset = pd.DataFrame(
    {
        "dialogue_id": dialogue_id_ls,
        "service": domains_ls_ls,
        "turn_id": turn_id_ls_ls,
        "speaker": speaker_id_ls_ls,
        "utterance": utterances_ls_ls,
        "state": states_ls_ls_dict,
    }
)

sgd_dataset.to_pickle("~/workplace/M2MHR/src/data/sgd_dataset.pkl")

#### question answering dataset

# get context
contexts_ls = []
questions_ls = []
answers_ls = []
loaded_dict_keys_ls = list(dialogue_dict.keys())
for i in range(len(loaded_dict_keys_ls)):
    d = dialogue_dict[loaded_dict_keys_ls[i]]
    for turn in d[0]:
        assert len(turn[0]) == len(turn[-1])
        for j in range(len(turn[0])):
            contexts_ls.append(turn[4])
            answers_ls.append(turn[-1][j])
            question_topic = turn[0][j]
            question_full = task_dict[domains_ls_ls[i][0]][question_topic][0]
            questions_ls.append(question_full)

# get question answering dataset
qa_dataset = pd.DataFrame(
    {
        "question": questions_ls,
        "answer": answers_ls,
        "answer_context": contexts_ls,
    }
)
qa_dataset = qa_dataset[qa_dataset.answer != ""]
qa_dataset.to_pickle("~/workplace/M2MHR/src/data/qa_dataset.pkl")
