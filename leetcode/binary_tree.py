from collections import deque


class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def breadth_first_search(root: TreeNode) -> list[TreeNode]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        result.append(node.value)

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return result


root = TreeNode(10, TreeNode((8), TreeNode(5), TreeNode(3)), TreeNode(1))
print(breadth_first_search(root))
