# Llama 3

We are unlocking the power of large language models. Our latest version of Llama is now accessible to individuals, creators, researchers, and businesses of all sizes so that they can experiment, innovate, and scale their ideas responsibly. 

This release includes model weights and starting code for pre-trained and instruction tuned Llama 3 language models — including sizes of 7B to 70B parameters.

This repository is intended as a minimal example to load Llama 3 models and run inference. For more detailed examples, see [llama-recipes](https://github.com/facebookresearch/llama-recipes/).

## Download

In order to download the model weights and tokenizer, please visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and accept our License.

Once your request is approved, you will receive a signed URL over email. Then run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have `wget` and `md5sum` installed. Then run the script: `./download.sh`.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as `403: Forbidden`, you can always re-request a link.

### Access to Hugging Face

We are also providing downloads on [Hugging Face](https://huggingface.co/meta-llama).

## Quick Start

You can follow the steps below to quickly get up and running with Llama 2 models. These steps will let you run quick inference locally. For more examples, see the [Llama recipes repository](https://github.com/facebookresearch/llama-recipes). 

1. In a conda env with PyTorch / CUDA available clone and download this repository.

2. In the top-level directory run:
    ```bash
    pip install -e .
    ```
3. Visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and register to download the model/s.

4. Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

5. Once you get the email, navigate to your downloaded llama repository and run the download.sh script. 
    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email. 
    - Do not use the “Copy Link” option but rather make sure to manually copy the link from the email.

6. Once the model/s you want have been downloaded, you can run the model locally using the command below:
```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-3-8b-prerelease-instruct/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```
**Note**
- Replace  `llama-3-8b-prerelease-instruct/` with the path to your checkpoint directory and `tokenizer.model` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#inference) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.
- This example runs the [example_chat_completion.py](example_chat_completion.py) found in this repository but you can change that to a different .py file.

## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 8B     | 1  |
| 70B    | 8  |

All models support sequence length up to 4096 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware.

### Pretrained Models

These models are not finetuned for chat or Q&A. They should be prompted so that the expected answer is the natural continuation of the prompt.

See `example_text_completion.py` for some examples. To illustrate, see the command below to run it with the llama-2-7b model (`nproc_per_node` needs to be set to the `MP` value):

```
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir llama-3-8b-prerelease/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

### Instruction-tuned Models

The fine-tuned models were trained for dialogue applications. To get the expected features and performance for them, a specific formatting defined in [`chat_completion`](https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212)
needs to be followed, including the `INST` and `<<SYS>>` tags, `BOS` and `EOS` tokens, and the whitespaces and breaklines in between (we recommend calling `strip()` on inputs to avoid double-spaces).

You can also deploy additional classifiers for filtering out inputs and outputs that are deemed unsafe. See the llama-recipes repo for [an example](https://github.com/facebookresearch/llama-recipes/blob/main/examples/inference.py) of how to add a safety checker to the inputs and outputs of your inference code.

Examples using llama-2-7b-chat:

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-3-8b-prerelease-instruct/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

Llama 3 is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
In order to help developers address these risks, we have created the [Responsible Use Guide][(https://ai.meta.com/static-resource/responsible-use-guide/).

## Issues

Please report any software “bug”, or other problems with the models through one of the following means:
- Reporting issues with the model: [github.com/facebookresearch/llama](http://github.com/facebookresearch/llama3)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## Model Card
See [MODEL_CARD.md](..).

## License

Our model and weights are licensed for both researchers and commercial entities, upholding the principles of openness. Our mission is to empower individuals, and industry through this opportunity, while fostering an environment of discovery and ethical AI advancements. 

See the [LICENSE](..) file, as well as our accompanying [Acceptable Use Policy](USE_POLICY.md)

## Questions

For common questions, the FAQ can be found [here](https://llama.meta.com/faq) which will be kept up to date over time as new questions arise. 
