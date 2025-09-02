def two_sum(nums: list[int], target: int) -> list[int]:
    sum_map = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in sum_map:
            return [sum_map[complement], i]
        sum_map[num] = i
    return []
