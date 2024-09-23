from typing import List, Optional
import fire

# Placeholder for Llama model imports and Dialog object
from llama import Llama, Dialog

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Runs the chat completion example with the Llama 3 model.
    Prompts are structured as dialogs between a user and assistant.
    
    The context window of Llama 3 models is 8192 tokens, so `max_seq_len` should be <= 8192.
    """
    
    # Build the Llama model instance
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Define example dialogs
    dialogs: List[List[Dialog]] = [
        [{"role": "user", "content": "What is the recipe for mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is known for its Gothic architecture and stunning stained glass windows.
""",
            },
            {"role": "user", "content": "What is so great about #1?"}
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"}
        ],
        [
            {"role": "system", "content": "Always answer with emojis"},
            {"role": "user", "content": "How to go from Beijing to NY?"}
        ]
    ]

    # Generate chat completions
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Display results
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}")
        print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
