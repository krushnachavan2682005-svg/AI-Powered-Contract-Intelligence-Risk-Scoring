import json

with open("reports/splitting/split_report.json", "r") as f:
    stats = json.load(f)

print(f"Total original unique answer spans: {stats['total_original_answer_spans']}")
print(f"Covered answer spans: {stats['covered_answer_spans']}")
print(f"Uncovered answer spans: {stats.get('uncovered_answer_spans', 0)}")
if stats["total_original_answer_spans"] > 0:
    print(
        f"Coverage %: {stats['covered_answer_spans'] / stats['total_original_answer_spans'] * 100:.2f}%"
    )
else:
    print("Coverage %: 0.00%")
print(f"Multiple windows count: {stats.get('multiple_window_answer_count', 0)}")

max_len = 0
min_len = 999999
total = 0
with open("data/interim/windows/cuad_qa_windows.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        l = len(json.loads(line)["input_ids"])
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l

print(f"Max length in cuad_qa_windows.jsonl is: {max_len}")
print(f"Min length in cuad_qa_windows.jsonl is: {min_len}")
print(f"Total windows: {total}")
