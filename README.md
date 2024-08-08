<p align="center">
  <img src="https://github.com/meta-llama/llama3/blob/main/Llama3_Repo.jpeg" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/meta-Llama"> Models on Hugging Face</a>&nbsp | <a href="https://ai.meta.com/blog/"> Blog</a>&nbsp |  <a href="https://llama.meta.com/">Website</a>&nbsp | <a href="https://llama.meta.com/get-started/">Get Started</a>&nbsp
<br>

---

## **Note of deprecation**

Thank you for developing with Llama models. As part of the Llama 3.1 release, we’ve consolidated GitHub repos and added some additional repos as we’ve expanded Llama’s functionality into being an e2e Llama Stack. Please use the following repos going forward:
- [llama-models](https://github.com/meta-llama/llama-models) - Central repo for the foundation models including basic utilities, model cards, license and use policies
- [PurpleLlama](https://github.com/meta-llama/PurpleLlama) - Key component of Llama Stack focusing on safety risks and inference time mitigations 
- [llama-toolchain](https://github.com/meta-llama/llama-toolchain) - Model development (inference/fine-tuning/safety shields/synthetic data generation) interfaces and canonical implementations
- [llama-agentic-system](https://github.com/meta-llama/llama-agentic-system) - E2E standalone Llama Stack system, along with opinionated underlying interface, that enables creation of agentic applications
- [llama-recipes](https://github.com/meta-llama/llama-recipes) - Community driven scripts and integrations

If you have any questions, please feel free to file an issue on any of the above repos and we will do our best to respond in a timely manner. 

Thank you!


# (Deprecated) Meta Llama 3

We are unlocking the power of large language models. Our latest version of Llama is now accessible to individuals, creators, researchers, and businesses of all sizes so that they can experiment, innovate, and scale their ideas responsibly.

This release includes model weights and starting code for pre-trained and instruction-tuned Llama 3 language models — including sizes of 8B to 70B parameters.

This repository is a minimal example of loading Llama 3 models and running inference. For more detailed examples, see [llama-recipes](https://github.com/facebookresearch/llama-recipes/).

## Download

To download the model weights and tokenizer, please visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and accept our License.

Once your request is approved, you will receive a signed URL over email. Then, run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Ensure you have `wget` and `md5sum` installed. Then run the script: `./download.sh`.

Remember that the links expire after 24 hours and a certain amount of downloads. You can always re-request a link if you start seeing errors such as `403: Forbidden`.

### Access to Hugging Face

We also provide downloads on [Hugging Face](https://huggingface.co/meta-llama), in both transformers and native `llama3` formats. To download the weights from Hugging Face, please follow these steps:

- Visit one of the repos, for example [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
- Read and accept the license. Once your request is approved, you'll be granted access to all the Llama 3 models. Note that requests used to take up to one hour to get processed.
- To download the original native weights to use with this repo, click on the "Files and versions" tab and download the contents of the `original` folder. You can also download them from the command line if you `pip install huggingface-hub`:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct
```

- To use with transformers, the following [pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) snippet will download and cache the weights:

  ```python
  import transformers
  import torch

  model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

  pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
  )
  ```

## Quick Start

You can follow the steps below to get up and running with Llama 3 models quickly. These steps will let you run quick inference locally. For more examples, see the [Llama recipes repository](https://github.com/facebookresearch/llama-recipes).

1. Clone and download this repository in a conda env with PyTorch / CUDA.

2. In the top-level directory run:
    ```bash
    pip install -e .
    ```
3. Visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and register to download the model/s.

4. Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

5. Once you get the email, navigate to your downloaded llama repository and run the download.sh script.
    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email.
    - Do not use the “Copy Link” option; copy the link from the email manually.

6. Once the model/s you want have been downloaded, you can run the model locally using the command below:
```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```
**Note**
- Replace  `Meta-Llama-3-8B-Instruct/` with the path to your checkpoint directory and `Meta-Llama-3-8B-Instruct/tokenizer.model` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#inference) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.
- This example runs the [example_chat_completion.py](example_chat_completion.py) found in this repository, but you can change that to a different .py file.

## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 8B     | 1  |
| 70B    | 8  |

All models support sequence length up to 8192 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware.

### Pretrained Models

These models are not finetuned for chat or Q&A. They should be prompted so that the expected answer is the natural continuation of the prompt.

See `example_text_completion.py` for some examples. To illustrate, see the command below to run it with the llama-3-8b model (`nproc_per_node` needs to be set to the `MP` value):

```
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir Meta-Llama-3-8B/ \
    --tokenizer_path Meta-Llama-3-8B/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

### Instruction-tuned Models

The fine-tuned models were trained for dialogue applications. To get the expected features and performance for them, specific formatting defined in [`ChatFormat`](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202)
needs to be followed: The prompt begins with a `<|begin_of_text|>` special token, after which one or more messages follow. Each message starts with the `<|start_header_id|>` tag, the role `system`, `user` or `assistant`, and the `<|end_header_id|>` tag. After a double newline `\n\n`, the message's contents follow. The end of each message is marked by the `<|eot_id|>` token.

You can also deploy additional classifiers to filter out inputs and outputs that are deemed unsafe. See the llama-recipes repo for [an example](https://github.com/meta-llama/llama-recipes/blob/main/recipes/inference/local_inference/inference.py) of how to add a safety checker to the inputs and outputs of your inference code.

Examples using llama-3-8b-chat:

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

Llama 3 is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
To help developers address these risks, we have created the [Responsible Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/).

## Issues

Please report any software “bug” or other problems with the models through one of the following means:
- Reporting issues with the model: [https://github.com/meta-llama/llama3/issues](https://github.com/meta-llama/llama3/issues)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md).

## License

Our model and weights are licensed for researchers and commercial entities, upholding the principles of openness. Our mission is to empower individuals and industry through this opportunity while fostering an environment of discovery and ethical AI advancements.

See the [LICENSE](LICENSE) file, as well as our accompanying [Acceptable Use Policy](USE_POLICY.md)

## Questions

For common questions, the FAQ can be found [here](https://llama.meta.com/faq), which will be updated over time as new questions arise.

## Contributing Guidelines

We welcome contributions to the LLaMA models! If you're interested in modifying, extending, or improving the models, please follow these guidelines:

### Getting Started

1. **Familiarize yourself with the codebase**:  
   Read through the README, explore the repository, and understand the model architecture and training procedures.

2. **Set up your environment**:  
   Install the required dependencies, including PyTorch, Transformers, and any other specified libraries.

3. **Choose a contribution area**:  
   Identify an area you'd like to contribute to, such as:
   - Model architecture improvements
   - Training data enhancements
   - New task or application development
   - Bug fixes or performance optimizations

   **Note**: The Meta Llama 3 repository is deprecated. Please use the consolidated GitHub repos (`llama-models`, `PurpleLlama`, `llama-toolchain`, `llama-agentic-system`, and `llama-recipes`) for contributions.

### Contribution Process

1. **Create an issue**:  
   Open a new issue on the GitHub repository to discuss your proposed contribution. This will help us understand your intentions and provide feedback.

2. **Fork the repository**:  
   Create a fork of the LLaMA repository to make changes.

3. **Make changes**:  
   Implement your changes, following the existing code style and structure.

4. **Test and validate**:  
   Thoroughly test and validate your changes to ensure they don't introduce errors or regressions.

5. **Submit a pull request**:  
   Create a pull request to merge your changes into the main repository.

6. **Review and feedback**:  
   Our team will review your contribution, provide feedback, and may request modifications.

### Code Style and Structure

- **Follow PyTorch and Transformers conventions**:  
  Adhere to the coding style and structure used in PyTorch and Transformers.

- **Use clear and descriptive naming**:  
  Choose variable, function, and module names that clearly indicate their purpose.

- **Write comprehensive docstrings**:  
  Document your code with detailed explanations of functions, classes, and modules.

### Testing and Validation

- **Write unit tests**:  
  Create unit tests to verify the correctness of your changes.

- **Run integration tests**:  
  Ensure your changes don't break existing functionality by running integration tests.

- **Validate performance**:  
  Verify that your changes don't negatively impact model performance.

### Additional Resources

- **LLaMA documentation**:  
  Refer to the LLaMA documentation for more information on the models and training procedures.

- **PyTorch and Transformers documentation**:  
  Consult the PyTorch and Transformers documentation for guidance on using these libraries.

- **Responsible Use Guide**:  
  Familiarize yourself with the Responsible Use Guide to ensure your contributions align with our ethical guidelines.
