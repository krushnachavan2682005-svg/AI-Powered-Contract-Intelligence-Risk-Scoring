def check_coverage(
    ans_local_start, ans_local_end, context_token_indices, offset_mapping, text
):
    token_start = -1
    token_end = -1

    for idx in context_token_indices:
        t_start, t_end = offset_mapping[idx]
        if token_start == -1 and t_end > ans_local_start:
            token_start = idx
        if t_start < ans_local_end:
            token_end = idx

    if token_start == -1 or token_end == -1 or token_start > token_end:
        return False

    mapped_start = offset_mapping[token_start][0]
    mapped_end = offset_mapping[token_end][1]

    if mapped_start > ans_local_start:
        if text[ans_local_start:mapped_start].strip():
            return False

    if mapped_end < ans_local_end:
        if text[mapped_end:ans_local_end].strip():
            return False

    return True


print("Test 1:", check_coverage(0, 5, [0, 1], {0: (0, 3), 1: (4, 5)}, "hello"))
print("Test 2 (trunc start):", check_coverage(0, 5, [0], {0: (3, 5)}, "hello"))
print("Test 3 (whitespace start):", check_coverage(0, 5, [0], {0: (2, 5)}, "  llo"))
