from typing import List, Optional, Set

from pydantic import BaseModel
from transformers import AutoTokenizer

from src.data.chunker import ChunkAnswerRecord, ChunkRecord
from src.data.window_config import WindowConfig


class QAWindowAnswer(BaseModel):
    text: str
    char_start: int
    char_end: int
    original_char_start: int
    original_char_end: int
    token_start: int
    token_end: int


class QAWindowRecord(BaseModel):
    example_id: str
    document_id: str
    chunk_id: str
    window_id: str
    split: str
    annotation_id: str
    category: str
    question: str
    context: str
    input_ids: List[int]
    attention_mask: List[int]
    offset_mapping: List[List[int]]
    answer: QAWindowAnswer


class QAWindowBuilder:
    def __init__(self, config: Optional[WindowConfig] = None):
        self.config = config or WindowConfig()
        # Initialize fast tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name, use_fast=True
        )

    def _get_question_for_category(self, category: str) -> str:
        # We can mock this for now or use a simple prefix.
        # Ideally, we should load CUAD questions. For now, simple format:
        return (
            f"Highlight the parts (if any) of this contract related to "
            f'"{category}" that should be reviewed by a lawyer. Details:'
        )

    def _evaluate_window(
        self,
        tokenized_batch,
        window_idx: int,
        chunk: ChunkRecord,
        split: str,
        ans: ChunkAnswerRecord,
        ans_idx: int,
        question: str,
        global_window_idx: str,
    ) -> Optional[QAWindowRecord]:
        input_ids = tokenized_batch["input_ids"][window_idx]
        attention_mask = tokenized_batch["attention_mask"][window_idx]
        offset_mapping = tokenized_batch["offset_mapping"][window_idx]
        sequence_ids = tokenized_batch.sequence_ids(window_idx)

        # Find context token bounds in this window
        context_token_indices = [
            i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
        ]

        if not context_token_indices:
            return None

        window_char_start = offset_mapping[context_token_indices[0]][0]
        window_char_end = offset_mapping[context_token_indices[-1]][1]

        ans_local_start = ans.local_start
        ans_local_end = ans.local_end

        if ans_local_start < window_char_start or ans_local_end > window_char_end:
            return None

        token_start = -1
        token_end = -1

        for idx in context_token_indices:
            token_char_start, token_char_end = offset_mapping[idx]
            
            # token_start is the first token whose end is past the start of the answer
            if token_start == -1 and token_char_end > ans_local_start:
                token_start = idx
            
            # token_end is the last token whose start is before the end of the answer
            if token_char_start < ans_local_end:
                token_end = idx

        if token_start == -1:
            token_start = context_token_indices[0]
        if token_end == -1:
            token_end = context_token_indices[-1]

        example_id = f"{chunk.chunk_id}_ans{ans_idx}_win{global_window_idx}"

        return QAWindowRecord(
            example_id=example_id,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            window_id=f"{chunk.chunk_id}_win{global_window_idx}",
            split=split,
            annotation_id=ans.annotation_id,
            category=ans.category,
            question=question,
            context=chunk.chunk_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            offset_mapping=offset_mapping,
            answer=QAWindowAnswer(
                text=ans.answer_text,
                char_start=ans_local_start,
                char_end=ans_local_end,
                original_char_start=ans.original_start,
                original_char_end=ans.original_end,
                token_start=token_start,
                token_end=token_end,
            ),
        )

    def build_windows_for_chunk(
        self, chunk: ChunkRecord, split: str
    ) -> List[QAWindowRecord]:
        windows = []

        covered_answers: Set[int] = set()

        for ans_idx, ans in enumerate(chunk.answers):
            question = self._get_question_for_category(ans.category)

            # Tokenize question + context
            tokenized = self.tokenizer(
                question,
                chunk.chunk_text,
                truncation="only_second",
                max_length=self.config.max_seq_length,
                stride=self.config.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )

            num_windows = len(tokenized["input_ids"])

            for window_idx in range(num_windows):
                record = self._evaluate_window(
                    tokenized,
                    window_idx,
                    chunk,
                    split,
                    ans,
                    ans_idx,
                    question,
                    str(window_idx),
                )
                if record:
                    windows.append(record)
                    covered_answers.add(ans_idx)

        # Answer-aware recovery for uncovered answers
        # For any answer not covered by standard overflow windows, check if we can shift a window to cover it.
        # This handles stride-gap cases.
        for ans_idx, ans in enumerate(chunk.answers):
            if ans_idx in covered_answers:
                continue

            question = self._get_question_for_category(ans.category)

            # Tokenize full chunk unbounded to get precise token boundaries of the answer
            full_tokenized = self.tokenizer(
                chunk.chunk_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            full_offsets = full_tokenized["offset_mapping"]

            ans_token_start = -1
            ans_token_end = -1
            for idx, (start, end) in enumerate(full_offsets):
                if ans_token_start == -1 and end > ans.local_start:
                    ans_token_start = idx
                if start < ans.local_end:
                    ans_token_end = idx

            if (
                ans_token_start == -1
                or ans_token_end == -1
                or ans_token_start > ans_token_end
            ):
                continue

            ans_token_len = ans_token_end - ans_token_start + 1

            # Find exact context budget
            q_tokenized = self.tokenizer(question, add_special_tokens=True)
            q_len = len(q_tokenized["input_ids"])
            # The context gets 1 [SEP] token at the end usually for BERT-like.
            # Total budget: max_seq_length - q_len
            context_budget = self.config.max_seq_length - q_len

            if ans_token_len > context_budget:
                # Genuinely impossible to fit
                continue

            # Shift a window so that the answer is fully inside it
            # We will symmetrically pad it around the answer
            extra_tokens = context_budget - ans_token_len
            left_pad = extra_tokens // 2

            win_tok_start = max(0, ans_token_start - left_pad)
            win_tok_end = min(len(full_offsets), win_tok_start + context_budget)
            if win_tok_end - win_tok_start < context_budget:
                win_tok_start = max(0, win_tok_end - context_budget)

            window_char_start = full_offsets[win_tok_start][0]
            # Handle empty right token list safely
            window_char_end = full_offsets[win_tok_end - 1][1]

            shifted_context = chunk.chunk_text[window_char_start:window_char_end]

            shifted_tokenized = self.tokenizer(
                question,
                shifted_context,
                truncation="only_second",
                max_length=self.config.max_seq_length,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )

            # Since shifted_context might map to multiple windows if tokenization diverges,
            # we evaluate all and pick the first valid one
            for shifted_idx in range(len(shifted_tokenized["input_ids"])):
                # We need to map ans_local_start relative to the chunk down to relative to shifted_context
                # Actually, our _evaluate_window expects ans.local_start to be absolute to chunk.chunk_text!
                # Wait, shifted_tokenized gives offsets RELATIVE to shifted_context!
                # Let's adjust the offset_mapping in shifted_tokenized so it's absolute to chunk.chunk_text!
                sequence_ids = shifted_tokenized.sequence_ids(shifted_idx)
                offsets = shifted_tokenized["offset_mapping"][shifted_idx]
                abs_offsets = []
                for seq_id, (start, end) in zip(sequence_ids, offsets):
                    if seq_id == 1:
                        abs_offsets.append(
                            (start + window_char_start, end + window_char_start)
                        )
                    else:
                        abs_offsets.append((start, end))
                shifted_tokenized["offset_mapping"][shifted_idx] = abs_offsets

                record = self._evaluate_window(
                    shifted_tokenized,
                    shifted_idx,
                    chunk,
                    split,
                    ans,
                    ans_idx,
                    question,
                    f"recovered_{ans_idx}",
                )
                if record:
                    windows.append(record)
                    break

        return windows
