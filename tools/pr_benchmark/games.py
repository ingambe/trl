# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local tool-calling games; `reset(seed=...)` fixes the game, so every rollout of a group plays the same one."""

import random


WORDS = """
about above actor adult after again agent agree alarm album alive allow alone along angle apple apply argue arise
award aware basic beach begin being below birth black blame blind block blood board brain brand bread break bring
broad brown build buyer cable carry catch cause chain chair chart cheap check chest chief child claim class clean
clear climb clock close cloud coach coast count court cover craft crash cream crime cross crowd dance death delay
depth dirty doubt draft drama dream dress drink drive early earth eight empty enemy enjoy enter entry equal error
event exact exist extra faith false fault field fight final first flash floor focus force frame fresh front fruit
glass grant grass great green group guard guess guest guide happy heart heavy horse hotel house human image index
""".split()

WORDLE_PROMPT = (
    "Play Wordle: find the hidden 5-letter English word in at most 6 guesses. Make each guess with the `guess` tool. "
    "Feedback marks each letter G (right letter, right place), Y (in the word, wrong place) or X (not in the word)."
)

NUMBER_PROMPT = (
    "Find the hidden whole number between 1 and 100 in at most 7 guesses. Make each guess with the `guess` tool, "
    "which answers higher, lower or correct."
)


class Wordle:
    def reset(self, seed, **kwargs):
        self.target = random.Random(seed).choice(WORDS)
        self.guesses = 0
        self.reward = 0.0

    def guess(self, word: str) -> str:
        """
        Guess the hidden word.

        Args:
            word: A 5-letter English word.

        Returns:
            Feedback for each letter.
        """
        if self.guesses == 6 or self.reward == 1.0:
            raise ValueError("The game is over.")
        word = word.strip().lower()
        if len(word) != 5 or not word.isalpha():
            raise ValueError("The guess must be a 5-letter word.")
        self.guesses += 1
        feedback = ["G" if a == b else "X" for a, b in zip(word, self.target, strict=True)]
        remaining = [b for a, b in zip(word, self.target, strict=True) if a != b]
        for i, letter in enumerate(word):
            if feedback[i] == "X" and letter in remaining:
                feedback[i] = "Y"
                remaining.remove(letter)
        # Partial credit for the best guess, full credit when solved
        score = 1.0 if word == self.target else (2 * feedback.count("G") + feedback.count("Y")) / 20
        self.reward = max(self.reward, score)
        return "".join(feedback)


class GuessNumber:
    def reset(self, seed, **kwargs):
        self.target = random.Random(seed).randint(1, 100)
        self.guesses = 0
        self.reward = 0.0

    def guess(self, number: int) -> str:
        """
        Guess the hidden number.

        Args:
            number: A whole number between 1 and 100.

        Returns:
            Whether the hidden number is higher, lower or correct.
        """
        if self.guesses == 7 or self.reward == 1.0:
            raise ValueError("The game is over.")
        self.guesses += 1
        if number == self.target:
            self.reward = 1.0
            return "correct"
        # Partial credit for the closest guess
        self.reward = max(self.reward, 0.5 * (1 - abs(number - self.target) / 100))
        return "higher" if number < self.target else "lower"


GAMES = {"wordle": (Wordle, WORDLE_PROMPT), "number": (GuessNumber, NUMBER_PROMPT)}
