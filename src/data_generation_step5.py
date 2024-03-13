from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
import pickle


def entity_extract(input, question_answerer):

    QA_input = {"question": input["question"], "context": input["context"]}
    r = question_answerer(QA_input)
    return r["answer"], r["score"]


with open("~/workplace/M2MHR/data/generated_dictionary_step4.pkl", "rb") as f:
    loaded_dict = pickle.load(f)

count = []
for i in range(len(loaded_dict)):
    if (len(loaded_dict[i][0]) * 2 + 2) == (
        len([i for i in loaded_dict[i][-1].split("\n") if i != ""])
    ):
        count.append(i)
model_name = "deepset/deberta-v3-large-squad2"
question_answerer = pipeline("question-answering", model=model_name)
newd = {}
for i in count:
    current_conv = [i for i in loaded_dict[i][-1].split("\n") if i != ""]
    l1 = loaded_dict[i][0]
    for ii in range(len(loaded_dict[i][0])):

        l1[ii].append(current_conv[ii * 2 + 2])
        l1[ii].append(current_conv[ii * 2 + 3])
    newd[i] = [l1, loaded_dict[i][-1]]

with open("~/workplace/M2MHR/data/template.pkl", "rb") as f:
    templates = pickle.load(f)
all_q = {}
for i in templates:
    for j in templates[i]:
        all_q[j] = templates[i][j][0]

for ind in newd:
    for list in newd[ind][0]:
        extracted_truth = []
        for key in list[0]:
            input = {}
            input["question"] = all_q[key]
            input["context"] = list[-1]

            extracted_truth.append(entity_extract(input, question_answerer))

        list.append(extracted_truth)
    if ind % 10 == 0:
        with open(
            "~/workplace/M2MHR/data/generated_dictionary_step5.pkl",
            "wb",
        ) as f:
            pickle.dump(newd, f)
