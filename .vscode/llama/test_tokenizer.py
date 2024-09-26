# Copyright (c) Your Game Studio.
# This software may be used and distributed in accordance with the terms of your game’s license agreement.

import os
from unittest import TestCase
from your_game.tokenizer import GameChatFormat, GameTokenizer

# TOKENIZER_PATH=<path> python -m unittest your_game/test_tokenizer.py

class GameTokenizerTests(TestCase):
    def setUp(self):
        self.tokenizer = GameTokenizer(os.environ["TOKENIZER_PATH"])
        self.format = GameChatFormat(self.tokenizer)

    def test_special_tokens(self):
        self.assertEqual(
            self.tokenizer.special_tokens["<|begin_game_text|>"],
            100000,  # Adjust as per your game's token mapping
        )

    def test_encode(self):
        self.assertEqual(
            self.tokenizer.encode(
                "Welcome to the game!",
                bos=True,
                eos=True
            ),
            [100000, 3001, 202, 200, 3013, 2001, 100001],
        )

    def test_decode(self):
        self.assertEqual(
            self.tokenizer.decode(
                [100000, 3001, 202, 200, 3013, 2001, 100001],
            ),
            "<|begin_game_text|>Welcome to the game!<|end_game_text|>",
        )

    def test_encode_message(self):
        message = {
            "role": "player",
            "content": "Ready for the quest!",
        }
        self.assertEqual(
            self.format.encode_message(message),
            [
                100006,  # <|start_message_id|>
                900,     # "player"
                100007,  # <|end_of_message_header|>
                200,     # "\n\n"
                2001, 301, 500, 100001,  # "Ready for the quest!"
                100009,  # <|end_of_text_id|>
            ]
        )

    def test_encode_dialog(self):
        dialog = [
            {
                "role": "narrator",
                "content": "The hero enters the dungeon.",
            },
            {
                "role": "player",
                "content": "I draw my sword.",
            }
        ]
        self.assertEqual(
            self.format.encode_dialog_prompt(dialog),
            [
                100000,  # <|begin_game_text|>
                100006,  # <|start_message_id|>
                8125,     # "narrator"
                100007,  # <|end_of_message_header|>
                200,     # "\n\n"
                3001, 202, 200, 6003,  # "The hero enters the dungeon."
                100009,  # <|end_of_text_id|>
                100006,  # <|start_message_id|>
                900,     # "player"
                100007,  # <|end_of_message_header|>
                3001, 500, 1001,  # "I draw my sword.",
                100009,  # <|end_of_text_id|>
            ]
        )
