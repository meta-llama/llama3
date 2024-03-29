from unittest import TestCase
from llama.tokenizer import Tokenizer
from llama.generation import Message, MessageFormat
import os

class TokenizerTests(TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(os.environ["TOKENIZER_PATH"])
        self.formatter = MessageFormat(self.tokenizer)

    def test_special_tokens(self):
        self.assertEqual(
            self.tokenizer.special_tokens["<|begin_of_text|>"],
            128000,
        )

    def test_encode(self):
        self.assertEqual(
            self.tokenizer.encode(
                "This is a test sentence.",
                bos=True,
                eos=True
            ),
            [128000, 2028, 374, 264, 1296, 11914, 13, 128001],
        )

    def test_decode(self):
        self.assertEqual(
            self.tokenizer.decode(
                [128000, 2028, 374, 264, 1296, 11914, 13, 128001],
            ),
            "<|begin_of_text|>This is a test sentence.<|end_of_text|>",
        )

    def test_encode_message(self):
        message = {
            "role": "user",
            "content": "This is a test sentence.",
            "eot": True
        }
        self.assertEqual(
            self.formatter.encode_message(message),
            [
                128005,  # <|start_header_id|>
                882,  # "user"
                128006,  # <|end_of_header|>
                271,  # "\n\n"
                2028, 374, 264, 1296, 11914, 13,  # This is a test sentence.
                128009,  # <|eot_id|>
            ]
        )

    def test_decode_message(self):
        self.assertEqual(
            self.formatter.decode_message(
                [
                    128005,  # <|start_header_id|>
                    882,  # "user"
                    128006,  # <|end_of_header|>
                    271,  # "\n\n"
                    2028, 374, 264, 1296, 11914, 13,  # This is a test sentence.
                    128009,  # <|eot_id|>
                ]
            ),
            (
                [],  # no remaining tokens
                {
                    "role": "user",
                    "content": "This is a test sentence.",
                    "eot": True
                },
            )
        )

    def test_encode_dialog(self):
        dialog = [
            {
                "role": "user",
                "content": "This is a test sentence.",
                "eot": True
            }
        ]

        self.assertEqual(
            self.formatter.encode_dialog(dialog, bos=True, eos=True),
            [
                128000,  # <|begin_of_text|>
                128005,  # <|start_header_id|>
                882,  # "user"
                128006,  # <|end_of_header|>
                271,  # "\n\n"
                2028, 374, 264, 1296, 11914, 13,  # This is a test sentence.
                128009,  # <|eot_id|>
                128001,  # <|end_of_text|>
            ]
        )
