def insertion_sort(arr: list[int]):
    for i in range(1, len(arr)):
        value = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > value:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = value


arr = [10, 5, 1, 20, 15]
insertion_sort(arr)
print(arr)
