from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
question = "What is the answer?"
context = "The quick brown fox jumps over the lazy dog."

tokenized = tokenizer(
    question,
    context,
    truncation="only_second",
    max_length=512,
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding=False,
)

print(tokenized["offset_mapping"][0])
print(tokenized.sequence_ids(0))
