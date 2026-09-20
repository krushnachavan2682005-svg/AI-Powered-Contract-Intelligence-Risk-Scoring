from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from src.data.chunker import ChunkRecord
from src.data.window_config import WindowConfig


class QAWindowAnswer(BaseModel):
    text: str
    char_start: int
    char_end: int
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
        return f"Highlight the parts (if any) of this contract related to \"{category}\" that should be reviewed by a lawyer. Details:"

    def build_windows_for_chunk(
        self, chunk: ChunkRecord, split: str
    ) -> List[QAWindowRecord]:
        """
        Builds QA windows for a given chunk, handling tokenizer overflow.
        Only keeps positive windows where the answer span is fully contained.
        """
        windows = []
        
        # In CUAD, there could be multiple answers per annotation. 
        # ChunkAnswerRecord represents a single answer. We need to create a window 
        # for each answer. If a chunk has no answers, it will be handled differently later.
        
        # Iterate over all answers in the chunk
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
                padding=False,  # Don't pad during generation, handle in collator
            )

            # A single chunk may produce multiple tokenized windows
            num_windows = len(tokenized["input_ids"])
            
            for window_idx in range(num_windows):
                input_ids = tokenized["input_ids"][window_idx]
                attention_mask = tokenized["attention_mask"][window_idx]
                offset_mapping = tokenized["offset_mapping"][window_idx]
                sequence_ids = tokenized.sequence_ids(window_idx)
                
                # Find context token bounds in this window
                # sequence_ids is 0 for question, 1 for context, None for special tokens
                context_token_indices = [
                    i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
                ]
                
                if not context_token_indices:
                    continue
                
                # Check if the answer falls within the context tokens of this window
                window_char_start = offset_mapping[context_token_indices[0]][0]
                window_char_end = offset_mapping[context_token_indices[-1]][1]
                
                # The local char positions of the answer in the chunk
                ans_local_start = ans.local_start
                ans_local_end = ans.local_end
                
                # Is the answer fully contained in this window?
                if ans_local_start >= window_char_start and ans_local_end <= window_char_end:
                    # Find token start and end
                    token_start = -1
                    token_end = -1
                    
                    for idx in context_token_indices:
                        token_char_start, token_char_end = offset_mapping[idx]
                        if token_char_start <= ans_local_start and token_start == -1:
                            token_start = idx
                        if token_char_end >= ans_local_end and token_end == -1:
                            token_end = idx
                            
                    # If we couldn't exactly align, fallback to boundary tokens within context
                    if token_start == -1:
                        token_start = context_token_indices[0]
                    if token_end == -1:
                        token_end = context_token_indices[-1]
                        
                    # Create the window record
                    example_id = f"{chunk.chunk_id}_ans{ans_idx}_win{window_idx}"
                    
                    record = QAWindowRecord(
                        example_id=example_id,
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        window_id=f"{chunk.chunk_id}_win{window_idx}",
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
                            token_start=token_start,
                            token_end=token_end
                        )
                    )
                    windows.append(record)
                    
        return windows
