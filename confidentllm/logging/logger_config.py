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
# limitations under the License."
"""Set up logging configuration for project."""


def create_logging_config(
    log_file: str = "${hydra.runtime.output_dir}/${hydra.job.name}.log",
    log_level: str = "INFO",
    *,
    colorlog_console: bool = True,
) -> dict:
    """Create logging configuration."""
    if colorlog_console:
        color_format: str = (
            "%(light_black)s[%(asctime)s]%(reset)s "
            "%(log_color)s[%(levelname)8s]%(reset)s "
            "%(cyan)s[%(name)s]%(reset)s "
            "%(blue)s%(message)s%(reset)s "
            "%(cyan)s(%(filename)s:%(lineno)s)"
        )
        colored_formatter: dict[str, str] = {
            "()": "colorlog.ColoredFormatter",
            "format": color_format,
        }
    simple_format: str = (
        "[%(asctime)s][%(levelname)8s][%(name)s] "
        "%(message)s (%(filename)s:%(lineno)s)"
    )
    simple_formatter: dict[str, str] = {
        "class": "logging.Formatter",
        "format": simple_format,
    }

    formatters: dict[str, dict[str, str]] = {"simple": simple_formatter}
    if colorlog_console:
        formatters["colored"] = colored_formatter

    handlers: dict[str, dict[str, str]] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored" if colorlog_console else "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "simple",
            "filename": str(log_file),
        },
    }

    root: dict[str, str | list[str]] = {
        "level": log_level,
        "handlers": ["console", "file"],
    }

    log_config: dict[str, dict] = {
        "version": 1,
        "formatters": formatters,
        "handlers": handlers,
        "root": root,
    }

    return log_config
