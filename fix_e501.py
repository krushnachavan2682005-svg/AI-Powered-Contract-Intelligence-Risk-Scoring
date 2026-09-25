def fix_file(path, replacements):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


fix_file(
    "src/data/chunker.py",
    [
        (
            'f"Overlap ({self.config.overlap}) must be less than chunk_size ({self.config.chunk_size})."',
            'f"Overlap ({self.config.overlap}) must be less than "\n                f"chunk_size ({self.config.chunk_size})."',
        ),
        (
            "f\"Offset corruption during chunking for {ans['annotation_id']}\"",
            'f"Offset corruption during chunking for "\n                            f"{ans[\'annotation_id\']}"',
        ),
    ],
)

fix_file(
    "src/data/legal_preprocessor.py",
    [
        (
            "By default, all options that could alter text length and offset mappings are disabled.",
            "By default, all options that could alter text length and offset mappings\n    are disabled.",
        ),
        (
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}, \d{4}|\d{1,2}/\d{1,2}/\d{2,4}\b",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* "
            + "\n                "
            + r"\d{1,2}, \d{4}|\d{1,2}/\d{1,2}/\d{2,4}\b",
        ),
        (
            '"""Verifies that all answer spans in the original record still map correctly in the normalized text."""',
            '"""\n        Verifies that all answer spans in the original record still map correctly\n        in the normalized text.\n        """',
        ),
        (
            'f"Offset corruption detected in document {original_record.document_id}! "',
            'f"Offset corruption detected in document "\n                        f"{original_record.document_id}! "',
        ),
        (
            "f\"Expected '{answer.text}', but got '{span_text}' at [{answer.start}:{answer.end}].\"",
            "f\"Expected '{answer.text}', but got '{span_text}' \"\n                        f\"at [{answer.start}:{answer.end}].\"",
        ),
    ],
)

fix_file(
    "src/data/tokenizer.py",
    [
        (
            'f"Tokenizer {self.config.model_name} is not a fast tokenizer. Fast tokenizer is required for offset mapping."',
            'f"Tokenizer {self.config.model_name} is not a fast tokenizer. "\n                f"Fast tokenizer is required for offset mapping."',
        ),
        (
            "Tokenizes the chunk and returns offsets without truncation unless max_length is set.",
            "Tokenizes the chunk and returns offsets without truncation\n        unless max_length is set.",
        ),
        (
            "# Edge case: Answer might end exactly at the token start, which we might miss above",
            "# Edge case: Answer might end exactly at the token start, which we might\n        # miss above",
        ),
    ],
)

fix_file(
    "src/data/window_builder.py",
    [
        (
            "return f'Highlight the parts (if any) of this contract related to \"{category}\" that should be reviewed by a lawyer. Details:'",
            "return (\n            f'Highlight the parts (if any) of this contract related to '\n            f'\"{category}\" that should be reviewed by a lawyer. Details:'\n        )",
        ),
        (
            "# ChunkAnswerRecord represents a single answer. We need to create a window",
            "# ChunkAnswerRecord represents a single answer. We need to create a\n        # window",
        ),
        (
            "# for each answer. If a chunk has no answers, it will be handled differently later.",
            "# for each answer. If a chunk has no answers, it will be handled\n        # differently later.",
        ),
        (
            "# If we couldn't exactly align, fallback to boundary tokens within context",
            "# If we couldn't exactly align, fallback to boundary tokens within\n                    # context",
        ),
        (
            "# Adjust char offsets for the answer to be relative to the new window_text",
            "# Adjust char offsets for the answer to be relative to the new\n                    # window_text",
        ),
    ],
)

fix_file(
    "tests/unit/test_chunker.py",
    [
        (
            "# Chunk 1 won't contain it because it ends at 14. Wait, dynamic expansion might expand it!",
            "# Chunk 1 won't contain it because it ends at 14.\n    # Wait, dynamic expansion might expand it!",
        ),
        (
            "# Because answer starts at 8, which is < 10, dynamic expansion will expand chunk 1 to end at 14!",
            "# Because answer starts at 8, which is < 10, dynamic expansion\n    # will expand chunk 1 to end at 14!",
        ),
    ],
)

fix_file(
    "tests/unit/test_legal_preprocessor.py",
    [
        (
            "# Let's say we have a contract that gets normalized (e.g. control char stripped instead of replaced)",
            "# Let's say we have a contract that gets normalized\n    # (e.g. control char stripped instead of replaced)",
        )
    ],
)

fix_file(
    "tests/unit/test_window_builder.py",
    [
        (
            "# short max length to force overflow",
            "# short max length to force overflow  # noqa: E501",
        )
    ],
)
