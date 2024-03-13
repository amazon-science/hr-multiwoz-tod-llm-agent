import os
import pickle
import string

# Step 6 of the data generation is to clean up beginning and trailing whitespaces,
# as well as removing trailing punctuation.


def clean_string(s):
    """
    Remove leading and trailing whitespace, and trailing punctuation from a string.

    Args:
    s (str): The input string to be cleaned.

    Returns:
    str: The cleaned string with leading and trailing whitespace and trailing punctuation removed.
    """

    # Remove "Employee:"
    s = s.replace("Employee:", "")

    # Remove leading and trailing whitespace
    s = s.strip()

    # Remove trailing punctuation
    reversed_s = s[::-1]
    index = 0
    for char in reversed_s:
        # if char not in ['.', ',']:
        if char not in string.punctuation:
            break
        index += 1
    # Slice the string to remove the punctuation and reverse it back
    cleaned_s = reversed_s[index:][::-1]

    return cleaned_s


file_path = os.path.expanduser(
    "~/workplace/M2MHR/src/data/generated_dictionary_step5.pkl"
)
with open(file_path, "rb") as file:
    loaded_dict = pickle.load(file)

for i in loaded_dict.keys():
    turns = loaded_dict[i][0]
    new_turns = []
    for turn_elements in turns:
        raw_labels = turn_elements[5]
        clean_labels = []
        for j in range(len(raw_labels)):
            raw_label = raw_labels[j][0]
            clean_label = clean_string(raw_label)
            clean_labels.append(clean_label)
        turn_elements.append(clean_labels)
        new_turns.append(turn_elements)
    loaded_dict[i][0] = new_turns

file_path = os.path.expanduser(
    "~/workplace/M2MHR/src/data/generated_dictionary_step6.pkl"
)
with open(file_path, "wb") as file:
    pickle.dump(loaded_dict, file)
