# LeetCode problem solutions
# Arrays and Hashmaps
def two_sum(nums: list[int], target: int) -> list[int]:
    num_to_index = {}
    for index, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], index]
        num_to_index[num] = index
    return []


def best_time_to_buy_and_sell_stock(prices: list[int]) -> int:
    min_price = float("inf")
    max_profit = 0
    for price in prices:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
    return max_profit


def contains_duplicate(nums: list[int]) -> bool:
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


def valid_anagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    count = {}
    for char in s:
        count[char] = count.get(char, 0) + 1
    for char in t:
        if char not in count or count[char] == 0:
            return False
        count[char] -= 1
    return True


# Two Pointers
def valid_palindrome(s: str) -> bool:
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True


def move_zeroes(nums: list[int]) -> None:
    last_non_zero = 0
    for current in range(len(nums)):
        if nums[current] != 0:
            nums[last_non_zero], nums[current] = nums[current], nums[last_non_zero]
            last_non_zero += 1


def container_with_most_water(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        width = right - left
        current_height = min(height[left], height[right])
        max_area = max(max_area, width * current_height)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_area


def three_sum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return result


# Linked Lists
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two_sorted_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy
    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    current.next = l1 if l1 else l2
    return dummy.next


def linked_list_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev


# Trees
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root):
    if not root:
        return 0
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    return max(left_depth, right_depth) + 1


def invert_tree(root):
    if root:
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root


def diameter_of_binary_tree(root):
    diameter = 0

    def depth(node):
        nonlocal diameter
        if not node:
            return 0
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        diameter = max(diameter, left_depth + right_depth)
        return max(left_depth, right_depth) + 1

    depth(root)
    return diameter


def validate_binary_search_tree(root):
    def is_valid(node, low=float("-inf"), high=float("inf")):
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return is_valid(node.left, low, node.val) and is_valid(
            node.right, node.val, high
        )

    return is_valid(root)


def binary_tree_level_order_traversal(root):
    if not root:
        return []
    result, queue = [], [root]
    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result


def binary_tree_maximum_path_sum(root):
    max_sum = float("-inf")

    def max_gain(node):
        nonlocal max_sum
        if not node:
            return 0
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        current_path_sum = node.val + left_gain + right_gain
        max_sum = max(max_sum, current_path_sum)
        return node.val + max(left_gain, right_gain)

    max_gain(root)
    return max_sum


def kth_smallest_element_in_a_bst(root, k):
    stack = []
    current = root
    while True:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        k -= 1
        if k == 0:
            return current.val
        current = current.right
    return -1


# Stack
def valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else "#"
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack


# Math
def single_number(nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result


# Sliding Window
def longest_substring_without_repeating_characters(s: str) -> int:
    char_index_map = {}
    left = max_length = 0
    for right, char in enumerate(s):
        if char in char_index_map and char_index_map[char] >= left:
            left = char_index_map[char] + 1
        char_index_map[char] = right
        max_length = max(max_length, right - left + 1)
    return max_length


def min_window(s: str, t: str) -> str:
    if not t or not s:
        return ""

    dict_t = {}
    for char in t:
        dict_t[char] = dict_t.get(char, 0) + 1

    required = len(dict_t)
    l, r = 0, 0
    formed = 0
    window_counts = {}
    ans = float("inf"), None, None

    while r < len(s):
        char = s[r]
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1

        while l <= r and formed == required:
            char = s[l]

            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)

            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1

            l += 1

        r += 1
    return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]


def minimum_size_subarray_sum(target: int, nums: list[int]) -> int:
    left = 0
    current_sum = 0
    min_length = float("inf")
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1
    return 0 if min_length == float("inf") else min_length


def longest_repeating_character_replacement(s: str, k: int) -> int:
    count = {}
    left = 0
    max_count = 0
    max_length = 0

    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        max_count = max(max_count, count[s[right]])

        while (right - left + 1) - max_count > k:
            count[s[left]] -= 1
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length


# Binary Search
def modified_binary_search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged


# Dynamic Programming
def maximum_subarray(nums: list[int]) -> int:
    max_current = max_global = nums[0]
    for num in nums[1:]:
        max_current = max(num, max_current + num)
        if max_current > max_global:
            max_global = max_current
    return max_global


def climb_stairs(n: int) -> int:
    if n <= 1:
        return 1
    first, second = 1, 1
    for _ in range(2, n + 1):
        first, second = second, first + second
    return second


def house_robber(nums: list[int]) -> int:
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    prev1, prev2 = 0, 0
    for num in nums:
        temp = prev1
        prev1 = max(prev2 + num, prev1)
        prev2 = temp
    return prev1


def coin_change(coins: list[int], amount: int) -> int:
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1


# Graphs
from collections import deque


def number_of_islands(grid: list[list[str]]) -> int:
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    visited = set()
    island_count = 0

    def bfs(r, c):
        queue = deque([(r, c)])
        visited.add((r, c))
        while queue:
            row, col = queue.popleft()
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dr, dc in directions:
                rr, cc = row + dr, col + dc
                if (
                    0 <= rr < rows
                    and 0 <= cc < cols
                    and grid[rr][cc] == "1"
                    and (rr, cc) not in visited
                ):
                    visited.add((rr, cc))
                    queue.append((rr, cc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visited:
                bfs(r, c)
                island_count += 1

    return island_count


def course_schedule(num_courses: int, prerequisites: list[list[int]]) -> bool:
    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_degree = [0] * num_courses

    for dest, src in prerequisites:
        graph[src].append(dest)
        in_degree[dest] += 1

    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    visited_courses = 0

    while queue:
        course = queue.popleft()
        visited_courses += 1
        for neighbor in graph[course]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return visited_courses == num_courses


def pacific_atlantic_water_flow(heights: list[list[int]]) -> list[list[int]]:
    if not heights:
        return []

    rows, cols = len(heights), len(heights[0])
    pacific_reachable = set()
    atlantic_reachable = set()

    def dfs(r, c, reachable):
        reachable.add((r, c))
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            if (
                0 <= rr < rows
                and 0 <= cc < cols
                and (rr, cc) not in reachable
                and heights[rr][cc] >= heights[r][c]
            ):
                dfs(rr, cc, reachable)

    for r in range(rows):
        dfs(r, 0, pacific_reachable)
        dfs(r, cols - 1, atlantic_reachable)

    for c in range(cols):
        dfs(0, c, pacific_reachable)
        dfs(rows - 1, c, atlantic_reachable)

    return list(pacific_reachable & atlantic_reachable)


# Backtracking
def generate_parentheses(n: int) -> list[str]:
    result = []

    def backtrack(s="", left=0, right=0):
        if len(s) == 2 * n:
            result.append(s)
            return
        if left < n:
            backtrack(s + "(", left + 1, right)
        if right < left:
            backtrack(s + ")", left, right + 1)

    backtrack()
    return result


def permutations(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start=0):
        if start == len(nums):
            result.append(nums[:])
            return
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack()
    return result


def subsets(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start=0, path=[]):
        result.append(path)
        for i in range(start, len(nums)):
            backtrack(i + 1, path + [nums[i]])

    backtrack()
    return result


# Advanced Arrays
def trapping_rain_water(height: list[int]) -> int:
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water_trapped = 0
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            water_trapped += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water_trapped += right_max - height[right]
    return water_trapped


def kth_largest_element(nums: list[int], k: int) -> int:
    import heapq

    return heapq.nlargest(k, nums)[-1]


def top_k_frequent_elements(nums: list[int], k: int) -> list[int]:
    from collections import Counter
    import heapq

    count = Counter(nums)
    return [item for item, freq in heapq.nlargest(k, count.items(), key=lambda x: x[1])]
