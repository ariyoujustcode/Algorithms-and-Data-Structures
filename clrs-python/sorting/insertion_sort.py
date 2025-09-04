def insertion_sort(nums: list[int], n: int) -> list[int]:
    for i in range(1, n):
        current = nums[i]
        j = i - 1

        while j >= 0 and nums[j] > current:
            nums[j + 1] = nums[j]
            j -= 1

        nums[j + 1] = current

    return nums


nums = [10, 5, 1, 20, 15]
n = len(nums)
sorted_nums = insertion_sort(nums, n)
print(sorted_nums)
