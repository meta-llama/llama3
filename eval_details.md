# # ### def main(

    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Example to run chat completion with Llama 3 models.
    The prompts correspond to chat turns between the user and assistant, with the final one always being the user.
    """

    # Initialize the model generator
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Example dialogues for the chat completion
    dialogs: List[Dialog] = [
        [{"role": "user", "content": "What is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": "1. Eiffel Tower\n2. Louvre Museum\n3. Notre-Dame Cathedral"
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {"role": "system", "content": "Always answer with emojis"},
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    # Model setup
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Example dialogues
    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {"role": "assistant", "content": "Suggestions about Paris."},
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {"role": "system", "content": "Always answer with emojis"},
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]

    # Run the chat completion
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Print the results
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
