### Llama 3 Evaluation Details
This document contains additional context on the settings and methodology for how we evaluated the Llama 3 8 and 70B Pre-trained and Instruct-aligned models.

#### MMLU
- We are reporting macro averages for MMLU benchmarks. The micro average numbers for MMLU are: 65.4 and 67.4 for the 7B pre-trained and instruct-aligned models, 78.9 and 82.0 for the 70B pre-trained and instruct-aligned models
- For the instruct-aligned MMLU we ask the model to generate the best choice character
#### AGI English
- We use the default few-shot and prompt settings as specified here. The score is averaged over the english subtasks.
#### CommonSenseQA
- We use the same 7-shot chain-of-thought prompt as in Wei et al. (2022).
#### Winogrande
- We use a choice based setup for evaluation where we fill in the missing blank with the two possible choices and then compute log-likelihood over the suffix. We use 5 shots for evaluation.
#### BIG-Bench Hard
- We use a 3-shot chain of thought style prompting and compute the average exact match over the subsets in this task.
#### ARC-Challenge
- We use the arc-challenge subset from the arc benchmark. We use 25 shots and use the MMLU setup for evaluation where we provide all the choices in the prompt and calculate likelihood over choice characters
#### TriviaQA-WIKI
- We evaluate on the Wiki validation set and use 5 few-shot examples.
#### SQuAD
- We are using SQuAD v2 and compute exact match in a 1-shot setting.
#### QuAC
- Same setting as Llama 2 (1-shot, f1).
#### BoolQ
- Same setting as Llama 1 and Llama 2 (0-shot, accuracy).
#### DROP
- For each validation example, we draw 3 random few-shot examples from the train split.
#### GPQA
- We report 0-shot exact match scores over the possible options using the Main subset for our models and other open-source models (Mistral, Gemma).
#### HumanEval
- Same setting as Llama 1 and Llama 2 (pass@1).
#### GSM8K
- We use the same 8-shot chain-of-thought prompt as in Wei et al. (2022) (maj@1).
#### MATH
- We use the 4-shot problem available in Lewkowycz et al. (2022) (maj@1).
