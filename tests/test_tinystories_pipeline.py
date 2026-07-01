from pathlib import Path
import json
import tempfile
import unittest

from datasets.statistics import collect_dataset_statistics
from datasets.tinystories import (
    TinyStoriesPreprocessOptions,
    iter_tinystories_records,
    normalize_story,
    preprocess_tinystories_file,
    tiny_stories_url,
)
from datasets.token_cache import (
    collect_token_cache_statistics,
    iter_token_cache_blocks,
    load_token_cache_metadata,
    read_uint32_tokens,
    write_token_cache,
)
from scripts.download_tinystories import build_download_targets, download_file


class TinyStoriesPipelineTests(unittest.TestCase):
    def test_tinystories_url_points_to_huggingface_resolve_file(self) -> None:
        url = tiny_stories_url("gpt4", "validation")
        self.assertEqual(
            url,
            "https://huggingface.co/datasets/roneneldan/TinyStories/"
            "resolve/main/TinyStoriesV2-GPT4-valid.txt",
        )

    def test_download_targets_include_expected_output_file(self) -> None:
        targets = build_download_targets("original", "train", "datasets/raw")
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].filename, "TinyStories-train.txt")
        self.assertTrue(targets[0].output_path.endswith("TinyStories-train.txt"))

    def test_iter_tinystories_records_splits_endoftext_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "stories.txt"
            raw_path.write_text(
                "First story.\n<|endoftext|>\nSecond story.<|endoftext|>",
                encoding="utf-8",
            )

            records = list(iter_tinystories_records(raw_path, mode="delimiter"))

        self.assertEqual(records, ["First story.\n", "\nSecond story."])

    def test_preprocess_filters_duplicates_and_short_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "stories.txt"
            output_path = Path(directory) / "stories.jsonl"
            raw_path.write_text(
                "A complete little story about kindness.<|endoftext|>"
                "Too short<|endoftext|>"
                "A complete little story about kindness.<|endoftext|>"
                "Another complete little story about sharing.",
                encoding="utf-8",
            )

            report = preprocess_tinystories_file(
                raw_path,
                output_path,
                "train",
                TinyStoriesPreprocessOptions(min_characters=20, deduplicate=True),
            )
            rows = [
                json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 2)
        self.assertEqual(report["rejected_short"], 1)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(report["output_statistics"]["document_count"], 2)

    def test_preprocess_skips_existing_output_when_not_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "stories.txt"
            output_path = Path(directory) / "stories.jsonl"
            raw_path.write_text("A complete story about sharing.", encoding="utf-8")
            output_path.write_text('{"text": "existing"}\n', encoding="utf-8")

            report = preprocess_tinystories_file(
                raw_path,
                output_path,
                "train",
                TinyStoriesPreprocessOptions(min_characters=20),
                overwrite=False,
            )
            output_text = output_path.read_text(encoding="utf-8")

        self.assertTrue(report["skipped"])
        self.assertEqual(output_text, '{"text": "existing"}\n')

    def test_normalize_story_collapses_whitespace(self) -> None:
        self.assertEqual(
            normalize_story("  A   small\n\nstory.\t "),
            "A small story.",
        )

    def test_collect_dataset_statistics_reads_jsonl_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                '{"text": "one two"}\n{"text": "three"}\n',
                encoding="utf-8",
            )

            statistics = collect_dataset_statistics([path])

        self.assertEqual(statistics["document_count"], 2)
        self.assertEqual(statistics["word_count"], 3)
        self.assertEqual(statistics["text_key"], "text")

    def test_write_token_cache_uses_uint32_binary_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "data.jsonl"
            cache_path = Path(directory) / "tokens.bin"
            input_path.write_text(
                '{"text": "red blue"}\n{"text": "green"}\n',
                encoding="utf-8",
            )

            result = write_token_cache(
                [input_path],
                FakeTokenizer(),
                cache_path,
                block_size=3,
            )
            tokens = read_uint32_tokens(cache_path)
            metadata = load_token_cache_metadata(cache_path)

        self.assertEqual(tokens, [1, 3, 4, 2, 1, 5, 2])
        self.assertEqual(result.token_count, 7)
        self.assertEqual(result.sequence_count, 2)
        self.assertAlmostEqual(result.average_tokens_per_document, 3.5)
        self.assertEqual(result.average_sequence_length, 3)
        self.assertEqual(metadata["dtype"], "<u4")
        self.assertEqual(metadata["block_size"], 3)
        self.assertEqual(metadata["average_sequence_length"], 3)
        self.assertAlmostEqual(metadata["average_tokens_per_document"], 3.5)

    def test_token_cache_statistics_and_streaming_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "data.jsonl"
            cache_path = Path(directory) / "tokens.bin"
            input_path.write_text(
                '{"text": "red blue"}\n{"text": "green"}\n',
                encoding="utf-8",
            )

            write_token_cache([input_path], FakeTokenizer(), cache_path, block_size=3)
            statistics = collect_token_cache_statistics(cache_path)
            blocks = list(iter_token_cache_blocks(cache_path, use_mmap=False))

        self.assertTrue(statistics["file_size_matches_token_count"])
        self.assertTrue(statistics["memory_mapping_supported"])
        self.assertTrue(statistics["streaming_supported"])
        self.assertEqual(statistics["token_count"], 7)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0][0].tolist(), [1, 3, 4])
        self.assertEqual(blocks[0][1].tolist(), [3, 4, 2])

    def test_write_token_cache_skips_existing_cache_when_not_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "data.jsonl"
            cache_path = Path(directory) / "tokens.bin"
            input_path.write_text('{"text": "red"}\n', encoding="utf-8")

            first = write_token_cache([input_path], FakeTokenizer(), cache_path, block_size=2)
            second = write_token_cache(
                [input_path],
                FakeTokenizer(),
                cache_path,
                block_size=2,
                overwrite=False,
            )
            tokens = read_uint32_tokens(cache_path)

        self.assertEqual(first.token_count, second.token_count)
        self.assertEqual(tokens, [1, 3, 2])

    def test_download_file_skips_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            output = Path(directory) / "output.txt"
            source.write_text("new data", encoding="utf-8")
            output.write_text("old data", encoding="utf-8")

            size = download_file(source.as_uri(), output, overwrite=False)

        self.assertEqual(size, len("old data"))


class FakeTokenizer:
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids = [len(word) for word in text.split()]
        if add_bos:
            token_ids.insert(0, 1)
        if add_eos:
            token_ids.append(2)
        return token_ids


if __name__ == "__main__":
    unittest.main()
