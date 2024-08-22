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
"""Runner for question answering using the ConfidentLLM package."""

import logging
import subprocess

from datasets import Dataset
from hydra_zen import store, zen

from confidentllm import (
    data,  # noqa: F401 - Import data to make it available in the Zen Store
)
from confidentllm.models import ModelLoader
from confidentllm.scripts.setup_tools import setup_hydra_config_and_logging

__all__ = ["main"]
logger = logging.getLogger("__main__")


class ModelAndDataDownloader:
    """Class to download the model and data."""

    def __init__(
        self,
        model: ModelLoader | None,
        data: Dataset | None,  # noqa: F811
    ) -> None:
        """Initialize the downloader."""
        self.model = model
        self.data = data

    def download_model(self) -> None:
        """Download the model."""
        if self.model is not None:
            self.model.load()

    def download_data(self) -> None:
        """Download the data."""

    def sync(self) -> None:
        """Sync the model and data."""
        subprocess.run("hpc sync_hf", check=True, shell=True)  # noqa: S607

    def __call__(self) -> None:
        """Download and sync the model and data."""
        if self.model is not None:
            self.download_model()
        if self.data is not None:
            self.download_data()
        if self.model is not None or self.data is not None:
            self.sync()


@store(
    name="download_and_sync",
    hydra_defaults=[
        "_self_",
        {"model": "default"},
        {"data": "gsm8k"},
    ],
)
def run_downloader(
    data: Dataset | None,  # noqa: F811
    model: ModelLoader | None,
) -> None:
    """Run the model and data downloader."""
    downloader = ModelAndDataDownloader(model, data)
    downloader()


def main() -> None:
    """Run the download and sync script."""
    setup_hydra_config_and_logging(
        job_name="download_and_sync",
        config_keys=[
            "data.name",
            "model.pretrained_model_name_or_path",
        ],
    )

    # Generate the CLI for run_extraction
    zen(
        run_downloader,
    ).hydra_main(
        config_name="download_and_sync",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
