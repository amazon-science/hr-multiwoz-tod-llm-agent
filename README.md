## My Project

TODO: Fill this README out!

Be sure to:

* Change the title in this README
* Edit your repository description on GitHub

# :trophy: Extractive QA Benchmarks
| F1    | Exact Match | BLEU    | Method                        |
|-------|-------------|---------|-------------------------------|
|0.045  | 0.000       | 0.011   | bert-base-uncased [^1]        |
|0.047  | 0.000       | 0.010   | distilbert-base-uncased [^2]  |
|0.642  | 0.000       | 0.140   | roberta-base-squad2 [^3]      |
|0.050  | 0.000       | 0.011   | albert [^4]                   |
|0.050  | 0.001       | 0.011   | electra-small-discriminator [^5] |
|0.072  | 0.000       | 0.020   | xlnet-base [^6]               |
|0.710  | 0.000       | **0.217**   | deberta-v3-large [^7]         |
|0.588  | 0.000       | 0.134   | mdeberta-v3-base [^8]         |
|0.721  | 0.519       | 0.168   | distilbert-base [^9]          |
|**0.786**  | **0.598**       | 0.174   | bert-large-fintuned-squad [^10]      |


bert-large-uncased is finetuned through whole word masking on Squad dataset. This method achieves the best performance.
 https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad

The code to benchmark your extractive QA method
```
import pickle
import pandas as pd
from leaderboard.metric import calculate_f1_score, calculate_exact_match, calculate_bleu, calculate_rouge, calculate_meteor


#loading qa_dataset
with open('qa_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
#putting your predicted answer in data
data['predicted_answer'] = ...

#benchmark
F1 = calculate_f1_score(evaluations['answer'], evaluations[method])
Exact_Match = calculate_exact_match(evaluations['answer'].tolist(), evaluations[method].tolist())
Bleu = calculate_bleu(evaluations['answer'], evaluations[method])

```

## Reference
[1] https://arxiv.org/abs/1810.04805

[2] https://arxiv.org/abs/1910.01108

[3] https://huggingface.co/deepset/roberta-base-squad2

[4] https://arxiv.org/abs/1909.11942

[5] https://openreview.net/pdf?id=r1xMH1BtvB

[6] https://proceedings.neurips.cc/paper_files/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf

[7] https://arxiv.org/abs/2111.09543

[8] https://arxiv.org/abs/2111.09543

[9] https://arxiv.org/abs/1910.01108

[10] https://arxiv.org/abs/1810.04805


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

