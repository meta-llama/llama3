# LLaMA3_Cookbook 🦙✨
![d](https://github.com/jh941213/LLaMA3_cookbook/assets/112835087/98debd51-2697-45b9-b349-175349f2a407)


LLaMA3 (Large Language Model by META AI) is a leading-edge large language model that excels in AI technology. 🌟 This repository📁 is intended to provide information necessary to kick-start various projects🚀 using LLaMA3.


## [📜Introducing Meta Llama 3: The most capable openly available LLM to date review](https://hyun941213.tistory.com/entry/Introducing-Meta-Llama-3-The-most-capable-openly-available-LLM-to-date-%EB%A6%AC%EB%B7%B0)

## Official Website and Information 🌐
- [Official Website 🏠](https://llama.meta.com/)
- [Access Request 📬](https://llama.meta.com/llama-downloads/)
- [Meta Llama Model Card 🎴](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3)
- [Kaggle Meta 🏅](https://www.kaggle.com/organizations/metaresearch/models)
- [Meta Github 🐈‍⬛](https://github.com/meta-llama/llama3)

## ⚡️ Cloud API 
### API Calls Available 🔌
| Name          | Description                                                           | Link |
|--------------|-----------------------------------------------------------------------|-----|
| Grok         | High-performance AI Chip enabling LLaMA3 inference and API calls      | [Grok 🌐](https://console.groq.com/playground) |
| AWS          | Bedrock support for LLaMA at AWS, currently only Llama2 available     | [AWS 🌐](https://aws.amazon.com/ko/bedrock/) |
| Azure        | Support for 8B/70B models on Microsoft Azure, searchable via Azure Marketplace | [Azure 🌐](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=overview)|
| GCP          | Google Cloud Vertax AI support for LLaMA3 | [GCP 🌐](https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama3?_ga=2.164398141.-384541959.1712575317)
| together.ai  | Support for Llama2, CodeLlama, Llama3 8b/70b instances                | [together.ai 🌐](https://www.together.ai/) |
| replicate    | Llama3 API support (Node.js, Python, HTTP)                            | [replicate 🌐](https://replicate.com/blog/run-llama-3-with-an-api) |
| llama AI     | Support for Llama3 8B/70B, supports other OpenLLMs                    | [llama AI 🌐](https://www.llama-api.com/) |
| aimlapi | Supports various openLLMs as APIs | [AI/ML API](https://aimlapi.com/build-with-llama-3-api)|
| Nvidia API |Multiple OpenLLM models available Nvidia devloper |[llama AI 🌐](https://build.nvidia.com/explore/discover)
| Meta AI(github) | Connect to Meta AI api | [MetaAI 🌐](https://github.com/Strvm/meta-ai-api?tab=readme-ov-file)|



## 🤖 Inference 🧠

### Inference Platforms 🖥️
| Platform Name | Description                            | Link |
|--------------|----------------------------------------|-----|
| HuggingFace  | Llama 8B model                         | [Link 🌐](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| HuggingFace  | Llama 70B model                        | [Link 🌐](https://huggingface.co/meta-llama/Meta-Llama-3-70B) |
| HuggingFace  | Llama 8B Instruct model                | [Link 🌐](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| HuggingFace  | Llama 70B Instruct model               | [Link 🌐](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) |
| HuggingFace  | Llama Guard-2-8B(policy model)         | [Link 🌐](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B) |
| HuggingFace  | Llama 3 70B - FP8([friendliAI](https://friendli.ai/))     | [Link 🌐](https://huggingface.co/FriendliAI/Meta-Llama-3-70B-fp8) |
| HuggingFace  | Llama 3 70B Instruct - FP8([friendliAI](https://friendli.ai/))      | [Link 🌐](https://huggingface.co/FriendliAI/Meta-Llama-3-70B-Instruct-fp8) |
| HuggingFace  | Llama 3 8B - FP8([friendliAI](https://friendli.ai/))        | [Link 🌐](https://huggingface.co/FriendliAI/Meta-Llama-3-8B-fp8) |
| HuggingFace  | Llama 3 8B Instruct - FP8([friendliAI](https://friendli.ai/))      | [Link 🌐](https://huggingface.co/FriendliAI/Meta-Llama-3-8B-Instruct-fp8) |
| HuggingFace | Llama 8B KO (made beomi) | [Link 🌐](https://huggingface.co/beomi/Llama-3-Open-Ko-8B-preview)|
| Ollama       | Support for various lightweight Llama3 models | [Link 🌐](https://ollama.com/library/llama3) |

### HuggingFace Models 🐥
| Name   | Description                               | Link |
|--------------|----------------------------------------|-----|
| Trelis/Meta-Llama-3-70B-Instruct-function-calling | function calling | [Link 🌐](https://huggingface.co/Trelis/Meta-Llama-3-70B-Instruct-function-calling) |
| Trelis/Meta-Llama-3-8B-Instruct-function-calling | function calling | [Link 🌐](https://huggingface.co/Trelis/Meta-Llama-3-8B-Instruct-function-calling) |
|cognitivecomputations/dolphin-2.9-llama3-8b| Uncensored fine-tuning | [Link 🌐](https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b)|
|McGill-NLP/Llama-3-8B-Web| Zero-shot internet link selection capability |[Link 🌐](https://huggingface.co/McGill-NLP/Llama-3-8B-Web)
|teddylee777/Llama-3-Open-Ko-8B-Instruct-preview-gguf|Korean quantizatied GGUF model for ollama use|[Link 🌐](https://huggingface.co/teddylee777/Llama-3-Open-Ko-8B-Instruct-preview-gguf)
|beomi/Llama-3-Open-Ko-8B-Instruct-preview| Korean model trained with the Chat vector method| [Link 🌐](https://huggingface.co/beomi/Llama-3-Open-Ko-8B-Instruct-preview)|

## 💬 Chat Interface (Related Information) 💻
| Name              | Link |
|-----------------|-----|
| HuggingChat     | [Link 🌐](https://huggingface.co/chat/) |
| Groq            | [Link 🌐](https://groq.com/) |
| together.ai     | [Link 🌐](https://www.together.ai/) |
| replicate Llama chat(local) | [Link 🌐](https://github.com/replicate/llama-chat)|
| perplexity.ai(lightweight model) | [Link 🌐](https://labs.perplexity.ai/)|
|openrouter.ai| [Link 🌐](https://openrouter.ai/playground?models=meta-llama/llama-3-70b-instruct)|
| MetaAI (Not available in Korea)|[Link 🌐](https://www.meta.ai/)|
| Morphic(multimodal offerings) | [Link 🌐](https://www.morphic.sh/)|
| Nvidia AI | [[Link 🌐](https://www.nvidia.com/ko-kr/ai/#referrer=ai-subdomain)|

## LLaMA Framework  📘
| Name       |Type| Link 
|----------|-----|-----|
| Langchain | RAG | [Link 🌐](https://www.langchain.com/) |
| llamaindex| RAG | [Link 🌐](https://www.llamaindex.ai/) |
| llama.cpp| convert | [Link 🌐](https://github.com/ggerganov/llama.cpp) |

## 🛠️ Fine-tuning 🔧
| Name      | Link |
|---------|-----|
| Meta | [Link 🌐](https://llama.meta.com/docs/how-to-guides/fine-tuning/)
| torchrune| [Link 🌐](https://github.com/pytorch/torchtune)|
| LLaMAFactory| [Link 🌐](https://github.com/hiyouga/LLaMA-Factory)|
|axolotl| [Link 🌐](https://github.com/OpenAccess-AI-Collective/axolotl)|



## LLAMA3_Cookbook 👩‍🍳
| Information                         | Link |
|----------------------------------|-----|
| Prompt Engineering Guide           | [Link 🌐](https://www.promptingguide.ai/tools)|
| Using llama3 with WEB UI | [Link 🌐](https://dev.to/timesurgelabs/how-to-run-llama-3-locally-with-ollama-and-open-webui-297d) |
| API with Ollama, LangChain and ChromaDB with Flask API and PDF upload | [Link 🌐](https://www.youtube.com/watch?v=7VAs22LC7WE) |
| Guide for tuning and inference with Llama on MacBook | [Link 🌐](https://itnext.io/step-by-step-guide-to-running-latest-llm-model-meta-llama-3-on-apple-silicon-macs-m1-m2-or-m3-b9424ada6840) | 
| Fine-tune Llama 3 with ORPO | [Link 🌐](https://huggingface.co/blog/mlabonne/orpo-llama-3)|
| Qlora_aplaca_llama3 finetune | [Link 🌐](https://colab.research.google.com/drive/1mPw6P52cERr93w3CMBiJjocdTnyPiKTX#scrollTo=6bZsfBuZDeCL)
| fully local RAG agents with LLama3 | [Link 🌐](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb) |
| RAG Chatbot LLama3(HF) | [Link 🌐](https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3) |
| llama index RAG llama3 | [Link 🌐](https://lightning.ai/lightning-ai/studios/rag-using-llama-3-by-meta-ai) |
| ollama RAG + UI(Gradio) | [Link 🌐](https://mer.vin/2024/04/llama-3-rag-using-ollama/) |
| LangGraph + Llama3 | [Link 🌐](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb)|


## LLM Dataset 🗂️
| Information                       | Link |
|----------------------------------|-----|
|HuggingFaceFW/fineweb | [Link 🌐](https://huggingface.co/datasets/HuggingFaceFW/fineweb)|
|mlabonne/orpo-dpo-mix-40k| [Link 🌐](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)|

## LLM skills 📌
| Information                       | Link |
|----------------------------------|-----|
|FSDP+QLORA finetunning | [Link 🌐](https://github.com/AnswerDotAI/fsdp_qlora)|


## MAC vs 4090 Comparison 🖥️🆚🖥️
| Category        | M3 Max                              | M1 Pro                    | RTX 4090                                    |
|---------------|-------------------------------------|---------------------------|---------------------------------------------|
| CPU Cores      | 16 cores                            | 10 cores                  | 16 cores AMD                                |
| Memory         | 128GB                               | 16GB /32GB                | 32GB                                        |
| GPU Memory    | 16 core CPU & 40 core GPU, 400GB/s memory bandwidth              | 10 core CPU(8 performance cores & 2 efficiency cores) 16 core GPU 200GB/s memory bandwidth | 24GB                                        |
| **Model 7B**   | - Performs well on all computers    | - Performs well on all computers | - Performs well on all computers, similar performance to M3 Max |
| **Model 13B**  | - Good performance                  | - Third best performance  | - Best performance                          |
| **Model 70B**  | - Runs quickly, utilizing 128GB memory | - Lacks memory at 16GB, prone to crashes and reboots | - Cannot run on GPU, very slow on CPU |
| Lightening     | Not necessary with sufficient memory| Should be considered      | Quantization compromises necessary          |
| Power Consumption | 65W                             |                           | 250-300W                                    |
| Value for Money | Excellent ($4600)                 |                           | Relatively low ($6000 for A6000 GPU)        |

## 🙌 Contributing 💖
Would you like to contribute to this repository? Feel free to leave your comments on [Issues](https://github.com/your-github/LLaMA3_Recipes/issues) or send a Pull Request. All types of contributions are welcome!

## 📩 Contact Us 💌
Need more information or wish to collaborate? Click [here](https://github.com/your-github/LLaMA3_Recipes) to send me a message. Let's share knowledge together!
