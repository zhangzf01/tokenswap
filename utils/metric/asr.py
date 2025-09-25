import evaluate
import datasets


class ASR(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="A custom metric that returns attack sucess rate.",
            citation="",
            inputs_description="Takes references and predictions as inputs.",
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence")),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=[],
            reference_urls=[],
        )

    def _compute(self, references, predictions):
        contains = [reference.lower() in prediction.lower() for reference, prediction in zip(references, predictions)]

        contains_ratio = sum(contains) / len(references)

        return {
            "asr": contains_ratio,
        }