# coding=utf-8
# Copyright 2022 the HuggingFace Datasets Authors.
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

import os
import pandas as pd 
import datasets
import json
from huggingface_hub import hf_hub_url

_INPUT_CSV = "flickr_annotations_30k.csv"
_INPUT_CSV_PREFIX = "flickr_annotations_30k"
_INPUT_IMAGES = "flickr30k-images"
_REPO_ID = "nlphuji/flickr30k"
_JSON_KEYS = ['raw', 'sentids']

class Dataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="TEST", version=VERSION, description="test"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                 {
                "image": datasets.Image(),
                "captions": [datasets.Value('string')],
                "sentids": [datasets.Value("string")],
                "split": datasets.Value("string"),
                "img_id": datasets.Value("string"),
                "filename": datasets.Value("string"),
                "image_path": datasets.Value("string"),
                "image_id": datasets.Value("string"),
                }
            ),
            # task_templates=[],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        data_dir = self.config.data_dir

        repo_id = _REPO_ID

        splits = [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                "examples_csv": os.path.join(data_dir, _INPUT_CSV_PREFIX+'_train_filtered.csv'),
                "images_dir": os.path.join(data_dir)
            }),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                "examples_csv": os.path.join(data_dir, _INPUT_CSV_PREFIX+'_val_filtered.csv'),
                "images_dir": os.path.join(data_dir)
            }),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                "examples_csv": os.path.join(data_dir, _INPUT_CSV_PREFIX+'_test_filtered.csv'),
                "images_dir": os.path.join(data_dir)
            }),
        ]

        return splits
        # return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=gen_dir)]

    def _generate_examples(self, examples_csv, images_dir):
        """Yields examples."""
        df = pd.read_csv(examples_csv)
        for c in _JSON_KEYS:
            df[c] = df[c].apply(json.loads)

        for r_idx, r in df.iterrows():
            r_dict = r.to_dict()
            image_path = os.path.join(images_dir, _INPUT_IMAGES, r_dict['filename'])
            r_dict['image'] = image_path
            r_dict['captions'] = r_dict.pop('raw')
            r_dict['image_id'] = r_dict['img_id']
            r_dict['image_path'] = os.path.join(images_dir, _INPUT_IMAGES, r_dict['filename'])
            yield r_idx, r_dict