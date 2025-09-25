import datasets
import re
from pathlib import Path


class Flickr8kDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="flickr8k_dataset",
            version=VERSION,
        ),
    ]

    def _info(self):
        feature_dict = {
            "image_id": datasets.Value("string"),
            "image_path": datasets.Value("string"),
            "captions": datasets.Sequence(datasets.Value("string")),
        }
        return datasets.DatasetInfo(
            description="Flickr8k dataset",
            features=datasets.Features(feature_dict),
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        if not data_dir:
            raise ValueError("'data_dir' argument is required.")

        archive_path = dl_manager.extract(
            {
                "images": Path(data_dir).joinpath("Flickr8k_Dataset.zip"),
                "texts": Path(data_dir).joinpath("Flickr8k_text.zip"),
            }
        )
        splits = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_ids": Path(archive_path["texts"]).joinpath(
                        "Flickr8k_text/Flickr_8k.trainImages.txt"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_ids": Path(archive_path["texts"]).joinpath(
                        "Flickr8k_text/Flickr_8k.testImages.txt"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "image_ids": Path(archive_path["texts"]).joinpath(
                        "Flickr8k_text/Flickr_8k.devImages.txt"
                    ),
                },
            ),
            
        ]
        for split in splits:
            split.gen_kwargs.update(
                {
                    "image_dir": Path(archive_path["images"]).joinpath(
                        "Flicker8k_Dataset"
                    ),
                    "captions_txt": Path(archive_path["texts"]).joinpath(
                        "Flickr8k_text/Flickr8k.token.txt"
                    ),
                }
            )

        return splits

    def _generate_examples(
        self,
        image_ids,
        image_dir,
        captions_txt,
    ):
        def get_image_ids_as_list(fpath):
            ids = []
            with open(fpath, "r") as f:
                for image in f.read().split("\n"):
                    if image != "":
                        ids.append(image.split(".")[0])
                return ids

        def get_captions_by_id(captions_txt, id):
            captions = []
            with open(captions_txt, "r") as f:
                for line in f.read().split("\n"):
                    if line != "":
                        m = re.match(id + r".jpg#\d+(.*)", line)
                        if m != None:
                            captions.append(m.group(1)[1:])
                return captions

        ids = get_image_ids_as_list(image_ids)

        for id in ids:
            img_path = image_dir.joinpath(id + ".jpg")
            captions = get_captions_by_id(captions_txt, id)
            example = {
                "image_id": id,
                "image_path": str(img_path),
                "captions": captions,
            }
            yield id, example