import pytest

from src.data.splitter import DocumentSplitter
from src.data.splitter_config import SplitConfig

def test_document_splitter_deterministic():
    config = SplitConfig(seed=42)
    splitter = DocumentSplitter(config)
    docs = [f"doc_{i}" for i in range(100)]
    
    split1 = splitter.split_documents(docs)
    split2 = splitter.split_documents(docs)
    
    assert split1["train"] == split2["train"]
    assert split1["validation"] == split2["validation"]
    assert split1["test"] == split2["test"]

def test_document_splitter_distribution():
    config = SplitConfig(train_percent=0.7, validation_percent=0.15, test_percent=0.15)
    splitter = DocumentSplitter(config)
    docs = [f"doc_{i}" for i in range(100)]
    
    splits = splitter.split_documents(docs)
    
    assert len(splits["train"]) == 70
    assert len(splits["validation"]) == 15
    assert len(splits["test"]) == 15

def test_document_splitter_no_overlap():
    config = SplitConfig()
    splitter = DocumentSplitter(config)
    docs = [f"doc_{i}" for i in range(100)]
    
    splits = splitter.split_documents(docs)
    
    train_set = set(splits["train"])
    val_set = set(splits["validation"])
    test_set = set(splits["test"])
    
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == 100
