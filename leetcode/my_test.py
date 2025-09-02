def get_substring_count(s: str) -> int:
    len_s = len(s)
    count = 0

    if len_s < 2:
        return 0

    for i, char in enumerate(s):
        for j in range(i + 1, len_s):
            if s[j] == char:
                count += 1
                break
            else:
                continue
    return count
