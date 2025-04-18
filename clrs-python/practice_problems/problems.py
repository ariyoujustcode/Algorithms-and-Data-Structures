# Write a function that takes two integers and returns their sum
def getSum(a: int, b: int):
    return a + b

    # Problem one result
    print(f"One: " + str(getSum(2, 10)))


# Return true if a given number is even, else return false
def checkEven(c: int):
    return c % 2 == 0


# Problem two result
print(f"Two: " + str(checkEven(4)))


# Return max of three numbers
def getMax(d: int, e: int, f: int):
    return max(d, e, f)


# Problem three result
print(f"Three: " + str(getMax(3, 6, 9)))


# Return the reverse of a string
def getReverse(g: str):
    return g[::-1]


# Problem four result
print(f"Four: " + getReverse("hola"))


# Check palindrome
def checkPal(h: str):
    return h[::-1] == h


# Problem five result
print(f"Five: " + str(checkPal("racecar")))


# Count the vowels in a string
def count_vowels(i: str):
    vowels = "aeiou"
    return sum(1 for char in i if char in vowels)


# Problem six result
print(f"Six: " + str(count_vowels("grandma")))

# Find the second largest number in a list
integer_list = [1, 1, 2]


def secondLargest(integer_list):
    new_int_list = integer_list[:]

    if len(set(new_int_list)) < 2:
        return None

    maximum = max(new_int_list)
    new_int_list.remove(maximum)
    new_maximum = max(new_int_list)
    return new_maximum


print(str(secondLargest(integer_list)))

# Find the third largest (distinct) number in a list (can be repeats)
third_list = [1, 2, 3, 4, 5, 5, 5]


def thirdLargest(third_list):
    distinct_list = set(third_list)

    if len(distinct_list) < 3:
        return None

    sorted_distinct = sorted(distinct_list, reverse=True)

    return sorted_distinct[2]


print(str(thirdLargest(third_list)))


# Find the Kth largest distinct number in a list
def getKthLargest(k_list, k):
    # Make k_list a set of distinct numbers
    list_as_set = set(k_list)

    # Error handling
    if len(list_as_set) < k:
        return None

    # Sort descending to prompt Kth largest number in set
    sorted_set = sorted(list_as_set, reverse=True)

    k_largest = sorted_set[k - 1]

    return k_largest


print(str(getKthLargest([0, 1, 2, 2, 5, 7, 9, 20, 20, 21], 3)))


# Two Sum problem (Leetcode Easy)
# Given an array of integers nums and an integer target,
# return the indices of the two numbers such that they add up to the target.
def getIndices(nums, target):
    for i in range(len(nums)):
        num = nums[i]
        complement = target - num

        if complement in nums[i + 1 :]:
            # Get the index of the complement
            j = nums.index(complement, i + 1)

            return [i, j]

    return None


print(str(getIndices([1, 2, 5, 9, 12], 11)))


# Get unique character
def getUniqueChar(s: str):  # define function to get the unique character
    char_count = {}  # create a dictionary to count the characters (fill with ints)

    # Count occurrences of each character (case-insensitive)
    for char in s.lower():  # For each character in the lowercase string,
        if (
            char in char_count
        ):  # if the character is already in the dictionary as a key value pair,
            char_count[char] += 1  # add 1 to the value
        else:  # if not
            char_count[char] = (
                1  # make the value 1, indicating the first instance of this character
            )

    # Find the first unique character (using original case)
    for i in range(len(s)):  # for each index i in s
        if char_count[s[i].lower()] == 1:  # if the character count of that index is 1,
            return i  # Return the index of the first unique character

    return -1  # Return -1 if no unique character is found


# Test the function
print(getUniqueChar("Sagitarrius"))  # Expected output: 1 (index of 'p')


# Find all duplicates in an Array/List
def getDuplicate(original_list):
    # Define a dictionary to hold numbers and values
    genDict = {}
    duplicate_list = []

    for item in original_list:
        if item in genDict:
            genDict[item] += 1
        else:
            genDict[item] = 1

    for entry in genDict:
        if genDict[entry] > 1:
            duplicate_list.append(entry)

    return duplicate_list


print(str(getDuplicate([0, 1, 1, 2, 3, 3, 5, 5, 10, 11, 11])))