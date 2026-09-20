import random
from typing import Dict, List

from src.data.splitter_config import SplitConfig


class DocumentSplitter:
    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()

    def split_documents(self, document_ids: List[str]) -> Dict[str, List[str]]:
        """
        Splits a list of document IDs into train, validation, and test sets.
        Uses a deterministic random seed for reproducibility.
        """
        if not document_ids:
            return {"train": [], "validation": [], "test": []}

        # Deduplicate while preserving order for deterministic shuffle
        unique_docs = list(dict.fromkeys(document_ids))
        
        # Sort first to ensure deterministic ordering before shuffle
        unique_docs.sort()
        
        rng = random.Random(self.config.seed)
        rng.shuffle(unique_docs)
        
        total = len(unique_docs)
        train_end = int(total * self.config.train_percent)
        val_end = train_end + int(total * self.config.validation_percent)
        
        # Give remaining to test to handle float rounding issues
        train_docs = unique_docs[:train_end]
        val_docs = unique_docs[train_end:val_end]
        test_docs = unique_docs[val_end:]
        
        # Ensure test docs aren't empty if percentage > 0 and total > enough
        # But for large sets it will be fine.
        
        return {
            "train": train_docs,
            "validation": val_docs,
            "test": test_docs
        }
