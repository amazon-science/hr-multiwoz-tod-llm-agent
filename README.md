## My Project

TODO: Fill this README out!

Be sure to:

* Change the title in this README
* Edit your repository description on GitHub

## Benchmark Performance on Extractive QA
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
|**0.786**  | **0.598**       | 0.174   | bert-large-uncased [^10]      |


It seems that bert-large-uncased is finetuned through whole word masking on Squad dataset. This method achieves the best performance.
 https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad


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

