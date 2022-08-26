# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""SQUAD: The Stanford Question Answering Dataset."""
"""Modified version for fine tuning T5 on Question Generation """

import json

import datasets
#from datasets.tasks import QuestionAnsweringExtractive

logger = datasets.logging.get_logger(__name__)




_URL = "https://github.com/jannatulferdousruma17/bn/blob/main/"
_URLS = {
    "train": _URL + "train.json",
    # "dev": _URL + "dev-v1.1.json",
}


class SquadConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquadConfig, self).__init__(**kwargs)


class Squad(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        SquadConfig(
            name="plain_text",
            # version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "questions": datasets.Value("string"),   
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            # homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            # citation=_CITATION,
            task_templates=[
                
            ],
        )

    # def _split_generators(self, dl_manager):
    #     downloaded_files = dl_manager.download_and_extract(_URLS)

    #     return [
    #         datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
    #         datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
    #     ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                for paragraph in article["paragraphs"]:
                    source_text = f"generate questions: {paragraph['context'].strip()}"
                    questions = [qas['question'].strip() for qas in paragraph['qas']]
                    target_text = " {sep_token} ".join(questions)
                    target_text = f"{target_text} {{sep_token}}"
                    yield key, {
                          "context": source_text,
                          "questions": target_text}
                    key += 1
