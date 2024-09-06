# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk
# Year: 2024
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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
"""Module containing functions for extracting answers from input strings."""

import re

__all__ = ["extract_answers"]


class AnswerNotFoundError(Exception):
    """Exception raised when the answer is not found in the input string."""

    def __init__(self, input_string: str, pattern: str) -> None:
        """Initialize the AnswerNotFoundError.

        Args:
        ----
            input_string: The input string.
            pattern: The pattern used for searching.

        """
        self.input_string = input_string
        self.pattern = pattern
        self.message = (
            f"Answer not found in input string: {input_string} using pattern: {pattern}"
        )
        super().__init__(self.message)


def extract_answers(
    data: dict[str, list[str]],
    pattern: str,
) -> dict[str, list[str]]:
    """Extract the answers from the input string using a regular expression pattern.

    Args:
    ----
        data: The data dictionary containing the input strings.
        pattern: The regular expression pattern to search for in the answer strings.

    Returns:
    -------
        The data dictionary with the answers extracted from the input strings.

    """
    answers: list[str] = []
    for answer in data["answer"]:
        # Search for the pattern in the input string
        match = re.search(pattern, answer)

        # Check if a match is found and extract the answer
        if match:
            answers.append(match.group(1))
        else:
            raise AnswerNotFoundError(answer, pattern)

    data["answer"] = answers

    return data
