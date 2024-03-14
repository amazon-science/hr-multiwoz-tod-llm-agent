from metric import calculate_f1_score, calculate_exact_match, calculate_bleu, calculate_rouge, calculate_meteor
import pandas as pd
evaluations = pd.read_csv('qa_dataset_result.csv').fillna('')
methods = ['bert-base-uncased_result', 'distilbert-base-uncased_result',
       'roberta-base-squad2_result', 'albert-base-v2_result',
       'electra-small-discriminator_result', 'xlnet-base-cased_result',
       'deberta-v3-large-squad2_result', 'mdeberta-v3-base-squad2_result',
       'distilbert-base-cased-distilled-squad_result',
       'bert-large-uncased-whole-word-masking-finetuned-squad_result']
results = []
for method in methods:
    result = []
    result.append(calculate_f1_score(evaluations['answer'], evaluations[method]))
    result.append(calculate_exact_match(evaluations['answer'].tolist(), evaluations[method].tolist()))
    result.append(calculate_bleu(evaluations['answer'], evaluations[method]))
    
    results.append(result)
eva = pd.DataFrame(results, columns = ['F1', 'Exact Match', 'BLEU'])
eva['method'] = ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base-squad2', 'albert', 
                 'electra-small-discriminator', 'xlnet-base', 'deberta-v3-large','mdeberta-v3-base',
                  'distilbert-base',  'bert-large-uncased']
eva.to_csv('qa_evaluation.csv')


