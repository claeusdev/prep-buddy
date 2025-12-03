
import type { ReferenceItem } from '@/types';

export const DATA_STRUCTURES: ReferenceItem[] = [
  {
    id: 'ds-1',
    title: 'Dynamic Array (List)',
    category: 'Data Structure',
    summary: 'A resizable array that allows O(1) random access and amortized O(1) append.',
    complexity: {
      time: 'Access: O(1) | Search: O(n) | Insert/Delete: O(n)',
      space: 'O(n)'
    },
    description: `A Dynamic Array (like Python's list or Java's ArrayList) overcomes the fixed-size limitation of static arrays.
    
How it works:
1. It starts with a fixed initial capacity.
2. When you append an element and the array is full, it allocates a new chunk of memory (usually double the size).
3. It copies all existing elements to the new memory location.
4. It frees the old memory.

While the resizing operation takes O(n) time, it happens infrequently. Mathematical analysis shows that for N appends, the total cost is proportional to N, making the average (amortized) cost of a single append O(1).`,
    implementation: `
# Python lists are dynamic arrays by default

arr = []
arr.append(1)      # Amortized O(1)
arr.append(2)
val = arr[0]       # O(1) Access
exists = 2 in arr  # O(n) Search
arr.pop()          # O(1) Remove from end
arr.insert(0, 5)   # O(n) Insert at index involves shifting
    `
  },
  {
    id: 'ds-2',
    title: 'Hash Table (Dictionary)',
    category: 'Data Structure',
    summary: 'Stores key-value pairs using a hash function for O(1) average case lookup.',
    complexity: {
      time: 'Average: O(1) | Worst: O(n)',
      space: 'O(n)'
    },
    description: `A Hash Table provides fast data retrieval by mapping keys to values using a hash function.

Key Concepts:
- Hash Function: Converts a key (string, int, tuple) into an integer index.
- Buckets: The underlying array where data is stored.
- Collisions: When two keys hash to the same index. Handled via:
  1. Chaining: Storing a linked list at each bucket.
  2. Open Addressing: Probing for the next empty slot.

In Python, dictionaries use open addressing with a randomized probe sequence. While average operations are O(1), a bad hash function or high load factor can degrade performance to O(n).`,
    implementation: `
# Python dict is a Hash Table

hash_map = {}
hash_map["key"] = "value"  # O(1) Insert
val = hash_map.get("key")  # O(1) Access
exists = "key" in hash_map # O(1) Search
del hash_map["key"]        # O(1) Delete

# Iterating
for k, v in hash_map.items():
    print(k, v)
    `
  },
  {
    id: 'ds-3',
    title: 'Linked List',
    category: 'Data Structure',
    summary: 'A linear collection of nodes where each node points to the next.',
    complexity: {
      time: 'Access: O(n) | Insert/Delete: O(1) (if known)',
      space: 'O(n)'
    },
    description: `A Linked List consists of nodes where each node contains data and a reference (pointer) to the next node in the sequence.

Types:
- Singly Linked List: Nodes point to next only.
- Doubly Linked List: Nodes point to next and previous.
- Circular Linked List: Last node points back to head.

Advantages:
- Dynamic size.
- Efficient insertion/deletion at the beginning or if you have a reference to the node (O(1)).

Disadvantages:
- No random access (must traverse from head).
- Extra memory for pointers.`,
    implementation: `
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Usage
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)

# Traversal
curr = head
while curr:
    print(curr.val)
    curr = curr.next
    `
  },
  {
    id: 'ds-4',
    title: 'Stack',
    category: 'Data Structure',
    summary: 'LIFO (Last In, First Out) data structure.',
    complexity: {
      time: 'Push: O(1) | Pop: O(1) | Peek: O(1)',
      space: 'O(n)'
    },
    description: `A Stack follows the Last-In, First-Out principle, similar to a stack of plates. The last element added is the first one removed.

Use Cases:
- Function call management (Call Stack/Recursion).
- Expression evaluation (parsing math).
- Backtracking algorithms (DFS).
- Undo/Redo features.

In Python, a standard list serves as an efficient stack using .append() and .pop().`,
    implementation: `
stack = []

stack.append(1)  # Push
stack.append(2)

top = stack[-1]  # Peek
val = stack.pop() # Pop (Returns 2)

is_empty = len(stack) == 0
    `
  },
  {
    id: 'ds-5',
    title: 'Queue',
    category: 'Data Structure',
    summary: 'FIFO (First In, First Out) data structure.',
    complexity: {
      time: 'Enqueue: O(1) | Dequeue: O(1)',
      space: 'O(n)'
    },
    description: `A Queue follows the First-In, First-Out principle, similar to a line at a store. The first element added is the first one processed.

Use Cases:
- Breadth-First Search (BFS).
- Job scheduling (printer queues, CPU scheduling).
- Buffering data streams.

Note: Using a Python list as a queue is inefficient because removing from the front (pop(0)) is O(n). Always use 'collections.deque'.`,
    implementation: `
from collections import deque

queue = deque()

queue.append(1) # Enqueue
queue.append(2)

front = queue[0] # Peek
val = queue.popleft() # Dequeue (Returns 1)

is_empty = len(queue) == 0
    `
  },
  {
    id: 'ds-6',
    title: 'Binary Search Tree (BST)',
    category: 'Data Structure',
    summary: 'A binary tree where left child < parent < right child.',
    complexity: {
      time: 'Avg: O(log n) | Worst: O(n)',
      space: 'O(n)'
    },
    description: `A BST is a rooted binary tree data structure with the key property:
For any node N, all values in its left subtree are less than N.val, and all values in its right subtree are greater than N.val.

This property allows for binary search-like efficiency for lookups, insertions, and deletions.

Problem:
If we insert sorted data (1, 2, 3, 4) into a simple BST, it becomes a linked list (skewed tree) with O(n) operations. Self-balancing trees (AVL, Red-Black) solve this.`,
    implementation: `
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

def insert(root, val):
    if not root: return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    return root
    `
  },
  {
    id: 'ds-7',
    title: 'Heap (Priority Queue)',
    category: 'Data Structure',
    summary: 'A binary tree where parent is always smaller (min-heap) or larger (max-heap) than children.',
    complexity: {
      time: 'Push: O(log n) | Pop: O(log n) | Peek: O(1)',
      space: 'O(n)'
    },
    description: `A Heap is a specialized tree-based structure that satisfies the heap property. It is usually implemented as an array (implicit tree).

Min-Heap: Parent <= Children. Root is the minimum.
Max-Heap: Parent >= Children. Root is the maximum.

Use Cases:
- Efficiently finding the min or max element.
- Priority Queues (Dijkstra's Algorithm).
- Heap Sort.
- Finding the Kth largest/smallest element.

Python's 'heapq' module implements a Min-Heap.`,
    implementation: `
import heapq

min_heap = []
heapq.heappush(min_heap, 5)
heapq.heappush(min_heap, 1)
heapq.heappush(min_heap, 3)

# 1 is at index 0
smallest = heapq.heappop(min_heap) # Returns 1, O(log n)
peek = min_heap[0] # Returns 3, O(1)

# Max Heap trick: multiply numbers by -1
val = 10
heapq.heappush(min_heap, -val)
largest = -heapq.heappop(min_heap)
    `
  },
  {
    id: 'ds-8',
    title: 'Union Find (Disjoint Set)',
    category: 'Data Structure',
    summary: 'Efficiently tracks partitioned sets and connectivity.',
    complexity: {
      time: 'Union/Find: O(α(n)) ~ O(1)',
      space: 'O(n)'
    },
    description: `Union-Find stores a collection of disjoint (non-overlapping) sets. It supports two primary operations:
1. Find(x): Determine which set 'x' belongs to.
2. Union(x, y): Merge the sets containing 'x' and 'y'.

Optimizations:
- Path Compression: During Find, make nodes point directly to the root.
- Union by Rank/Size: Attach the shorter tree to the taller tree to keep height minimal.

With both optimizations, operations are nearly O(1) (Inverse Ackermann function). Used in Kruskal's algorithm and detecting cycles in undirected graphs.`,
    implementation: `
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, p):
        if self.parent[p] != p:
            # Path compression
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            # Union by rank
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1
    `
  },
  {
    id: 'ds-9',
    title: 'Graphs (Adjacency List)',
    category: 'Data Structure',
    summary: 'Represents graph relationships using lists or dictionaries.',
    complexity: {
      time: 'Add Edge: O(1) | Check Edge: O(degree)',
      space: 'O(V + E)'
    },
    description: `Graphs consist of Vertices (Nodes) and Edges (Connections). They can be Directed/Undirected and Weighted/Unweighted.

Representation:
- Adjacency Matrix: 2D array. Good for dense graphs. Space O(V^2).
- Adjacency List: Dictionary or Array of lists. Good for sparse graphs. Space O(V+E).

The Adjacency List is the most common representation in interview problems. A hash map (dict) mapping a node to a list of its neighbors allows for flexible node labeling.`,
    implementation: `
from collections import defaultdict

# Graph construction
graph = defaultdict(list)
edges = [[0, 1], [1, 2], [2, 0], [2, 3]]

for u, v in edges:
    graph[u].append(v)
    graph[v].append(u) # If undirected

# graph looks like:
# {
#   0: [1, 2],
#   1: [0, 2],
#   2: [1, 0, 3],
#   3: [2]
# }

# Iterating neighbors
for neighbor in graph[2]:
    print(neighbor)
    `
  },
  {
    id: 'ds-10',
    title: 'Monotonic Stack',
    category: 'Data Structure',
    summary: 'A stack that maintains elements in sorted order to find Next Greater Elements.',
    complexity: {
      time: 'Construction: O(n)',
      space: 'O(n)'
    },
    description: `A Monotonic Stack is a stack where elements are always sorted (either increasing or decreasing).

It is typically used to solve "Next Greater Element" or "Previous Smaller Element" problems in O(n) time.

Logic (Decreasing Stack):
When pushing a new element 'X', if 'X' is greater than the top of the stack, pop the stack until the top is >= 'X'. The element 'X' is the "Next Greater Element" for all the popped items.

Used in: Daily Temperatures, Largest Rectangle in Histogram.`,
    implementation: `
def next_greater_elements(nums):
    # Stores indices
    stack = [] 
    result = [-1] * len(nums)
    
    for i, num in enumerate(nums):
        # While stack not empty and current > top
        while stack and nums[stack[-1]] < num:
            idx = stack.pop()
            result[idx] = num
        stack.append(i)
        
    return result

# Input: [2, 1, 2, 4, 3]
# Output: [4, 2, 4, -1, -1]
    `
  }
];

export const ALGORITHMS: ReferenceItem[] = [
  {
    id: 'algo-1',
    title: 'Binary Search',
    category: 'Algorithm',
    summary: 'Search a sorted array by repeatedly dividing the search interval in half.',
    complexity: {
      time: 'O(log n)',
      space: 'O(1)'
    },
    description: `Binary Search allows you to find a target value within a sorted collection in logarithmic time.

Process:
1. Compare target with the middle element.
2. If equal, you found it.
3. If target < mid, discard the right half.
4. If target > mid, discard the left half.
5. Repeat.

Crucial details:
- Calculating mid: 'left + (right - left) // 2' prevents integer overflow in some languages.
- Loop condition: 'left <= right' vs 'left < right' changes how you handle the boundary and return value.`,
    implementation: `
def binary_search(nums, target):
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
    `
  },
  {
    id: 'algo-2',
    title: 'Two Pointers',
    category: 'Algorithm',
    summary: 'Use two pointers to iterate through a structure to satisfy constraints.',
    complexity: {
      time: 'O(n)',
      space: 'O(1)'
    },
    description: `The Two Pointers technique involves using two references to traverse a data structure, usually an array or string.

Common patterns:
1. Converging: Start one pointer at the beginning and one at the end (e.g., checking palindrome, Two Sum sorted).
2. Parallel: Move both pointers in the same direction, perhaps at different speeds (e.g., merging sorted arrays).

This technique often reduces a nested loop O(n^2) solution to a single pass O(n) solution.`,
    implementation: `
def is_palindrome(s):
    # Clean string logic skipped for brevity
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]:
            return False
        l += 1
        r -= 1
    return True
    
# Two Sum (Sorted)
def two_sum_sorted(numbers, target):
    l, r = 0, len(numbers) - 1
    while l < r:
        cur_sum = numbers[l] + numbers[r]
        if cur_sum == target:
            return [l + 1, r + 1]
        elif cur_sum < target:
            l += 1
        else:
            r -= 1
    `
  },
  {
    id: 'algo-3',
    title: 'Sliding Window',
    category: 'Algorithm',
    summary: 'Efficiently process subarrays or substrings of a specific size or constraint.',
    complexity: {
      time: 'O(n)',
      space: 'O(1)'
    },
    description: `Sliding Window is used to perform operations on a specific window size of an array or string.

Types:
1. Fixed Size: The window size 'k' never changes. You slide it one step right, adding one element and removing one.
2. Dynamic Size: Expand 'right' pointer to include elements until a condition is met/broken, then contract 'left' pointer to restore the condition.

Used for: Longest Substring Without Repeating Characters, Maximum Sum Subarray of Size K.`,
    implementation: `
def max_sum_subarray(nums, k):
    max_sum = 0
    window_sum = 0
    left = 0
    
    for right in range(len(nums)):
        window_sum += nums[right]
        
        # Window size is k
        if right - left + 1 == k:
            max_sum = max(max_sum, window_sum)
            # Slide window
            window_sum -= nums[left]
            left += 1
            
    return max_sum
    `
  },
  {
    id: 'algo-4',
    title: 'Depth First Search (DFS)',
    category: 'Algorithm',
    summary: 'Traverse tree/graph structure by exploring as far as possible along each branch.',
    complexity: {
      time: 'O(V + E)',
      space: 'O(V)'
    },
    description: `DFS explores a path all the way to a leaf or dead end before backtracking.

Implementation:
- Recursion (uses the system call stack).
- Iterative (uses an explicit Stack).

Use Cases:
- Finding connected components.
- Path finding (though not necessarily shortest).
- Topological Sorting.
- Cycle detection.
- Solving puzzles (mazes, sudoku).`,
    implementation: `
# Recursive DFS
def dfs(graph, node, visited):
    if node in visited:
        return
    
    visited.add(node)
    # Process node
    print(node)
    
    for neighbor in graph[node]:
        dfs(graph, neighbor, visited)

# Usage
visited = set()
dfs(adj_list, start_node, visited)
    `
  },
  {
    id: 'algo-5',
    title: 'Breadth First Search (BFS)',
    category: 'Algorithm',
    summary: 'Traverse tree/graph structure level by level.',
    complexity: {
      time: 'O(V + E)',
      space: 'O(V)'
    },
    description: `BFS explores all neighbors of the current node before moving to the next level neighbors. It radiates outward from the start.

Implementation:
- Requires a Queue (FIFO).

Use Cases:
- Shortest Path in unweighted graphs (guaranteed).
- Level-order traversal of a tree.
- Finding all nodes within distance K.
- Web Crawlers.`,
    implementation: `
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    `
  },
  {
    id: 'algo-6',
    title: 'Prefix Sums',
    category: 'Algorithm',
    summary: 'Precompute cumulative sums to answer range sum queries in O(1).',
    complexity: {
      time: 'Build: O(n) | Query: O(1)',
      space: 'O(n)'
    },
    description: `Prefix Sum involves creating an auxiliary array where index 'i' stores the sum of all elements from index 0 to 'i'.

Formula:
Prefix[i] = nums[0] + ... + nums[i]
Sum(i, j) = Prefix[j] - Prefix[i-1]

This transforms a O(n) range sum query into O(1).
This concept extends to 2D matrices (for sum of rectangles) and products.`,
    implementation: `
class PrefixSum:
    def __init__(self, nums):
        # Padding with 0 for easier indexing
        self.prefix = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            self.prefix[i + 1] = self.prefix[i] + nums[i]

    def query(self, left, right):
        # Returns sum of nums[left...right] inclusive
        return self.prefix[right + 1] - self.prefix[left]
    `
  },
  {
    id: 'algo-7',
    title: 'Fast & Slow Pointers',
    category: 'Algorithm',
    summary: 'Use two pointers moving at different speeds to detect cycles or find midpoints.',
    complexity: {
      time: 'O(n)',
      space: 'O(1)'
    },
    description: `Also known as Floyd's Cycle-Finding Algorithm or the Tortoise and Hare algorithm.

How it works:
- Slow pointer moves 1 step at a time.
- Fast pointer moves 2 steps at a time.

Applications:
1. Cycle Detection: If there is a cycle, Fast will eventually lap Slow and they will meet.
2. Middle of List: When Fast reaches the end, Slow will be exactly at the middle.
3. Start of Cycle: Mathematical properties allow finding the exact node where the cycle begins after they meet.`,
    implementation: `
def has_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def find_middle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
    `
  },
  {
    id: 'algo-8',
    title: 'Topological Sort',
    category: 'Algorithm',
    summary: 'Linear ordering of vertices in a directed graph such that for every edge u->v, u comes before v.',
    complexity: {
      time: 'O(V + E)',
      space: 'O(V)'
    },
    description: `Topological Sorting is only possible in Directed Acyclic Graphs (DAGs). It basically means "scheduling tasks" where some tasks have prerequisites.

Algorithms:
1. Kahn's Algorithm (BFS):
   - Calculate in-degrees of all nodes.
   - Add nodes with 0 in-degree to a queue.
   - Process queue: remove node, reduce in-degree of neighbors.
   - If neighbor becomes 0, add to queue.
   
2. DFS Post-Order:
   - Run DFS. Add node to a list after visiting all children.
   - Reverse the list at the end.`,
    implementation: `
from collections import deque

def topological_sort(num_nodes, edges):
    indegree = [0] * num_nodes
    graph = {i: [] for i in range(num_nodes)}
    
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1
        
    queue = deque([i for i in range(num_nodes) if indegree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
                
    return result if len(result) == num_nodes else [] # Cycle check
    `
  },
  {
    id: 'algo-9',
    title: 'Quick Sort',
    category: 'Algorithm',
    summary: 'Divide-and-conquer sorting algorithm that selects a "pivot" and partitions the array.',
    complexity: {
      time: 'Avg: O(n log n) | Worst: O(n^2)',
      space: 'O(log n)'
    },
    description: `Quick Sort is one of the most efficient sorting algorithms in practice.

Steps:
1. Pivot Selection: Choose an element (first, last, random, or median).
2. Partitioning: Reorder the array so that all elements less than the pivot come before it, and all greater come after it.
3. Recursion: Recursively apply the same logic to the sub-arrays left and right of the pivot.

It is an in-place sort (low memory overhead) but is not stable (relative order of equal elements may change).`,
    implementation: `
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)
    `
  },
  {
    id: 'algo-10',
    title: 'Backtracking',
    category: 'Algorithm',
    summary: 'Algorithm for finding solutions by building candidates incrementally and abandoning invalid ones.',
    complexity: {
      time: 'Exponential',
      space: 'O(Recursion Depth)'
    },
    description: `Backtracking is a refined brute force approach. It explores the solution space via a recursion tree.

Key Concept:
- Make a choice.
- Recurse (move to next step).
- Undo the choice (Backtrack) to try other options.

Use this for Permutations, Combinations, Subsets, Sudoku, N-Queens, and Word Search problems.`,
    implementation: `
def permute(nums):
    res = []
    
    def backtrack(path, remaining):
        if not remaining:
            res.append(path[:])
            return
        
        for i in range(len(remaining)):
            # Make choice
            # Pass new path and new remaining list
            backtrack(
                path + [remaining[i]], 
                remaining[:i] + remaining[i+1:]
            )
            
    backtrack([], nums)
    return res
    `
  },
  {
    id: 'algo-11',
    title: 'Greedy',
    category: 'Algorithm',
    summary: 'Making the locally optimal choice at each stage with the hope of finding a global optimum.',
    complexity: {
      time: 'Problem Dependent (often O(n log n))',
      space: 'O(1) or O(n)'
    },
    description: `Greedy algorithms make the decision that looks best at the moment without regard for future consequences.

It works if the problem has:
1. Greedy Choice Property: A global optimum can be arrived at by selecting a local optimum.
2. Optimal Substructure: An optimal solution to the problem contains an optimal solution to subproblems.

Examples: Interval Scheduling, Huffman Coding, Dijkstra's (shortest path).
Counter-Example: 0/1 Knapsack (Requires DP), Coin Change with non-standard coins.`,
    implementation: `
# Interval Scheduling: Max number of non-overlapping intervals
def erase_overlap_intervals(intervals):
    if not intervals: return 0
    
    # Sort by end time
    intervals.sort(key=lambda x: x[1])
    
    end = intervals[0][1]
    count = 0 # Intervals to remove
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < end:
            # Overlap found, remove current
            count += 1
        else:
            # No overlap, update end
            end = intervals[i][1]
            
    return count
    `
  },
  {
    id: 'algo-12',
    title: 'Dynamic Programming',
    category: 'Algorithm',
    summary: 'Breaking a problem into simpler sub-problems and storing results to avoid re-computation.',
    complexity: {
      time: 'Problem Dependent',
      space: 'O(n) to O(n^2)'
    },
    description: `DP is basically Recursion + Caching. It is used when a problem has Overlapping Subproblems and Optimal Substructure.

Two Approaches:
1. Top-Down (Memoization): Recursive. Store the result of a function call in a hash map/array before returning. Check map before computing.
2. Bottom-Up (Tabulation): Iterative. Fill a table (dp array) starting from base cases (dp[0]) up to the target.

Common Problems: Fibonacci, Climbing Stairs, Coin Change, House Robber, LCS.`,
    implementation: `
# Top-Down (Memoization)
memo = {}
def fib(n):
    if n <= 1: return n
    if n in memo: return memo[n]
    memo[n] = fib(n-1) + fib(n-2)
    return memo[n]

# Bottom-Up (Tabulation) - Space Optimized
def fib_iterative(n):
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
    `
  },
  {
    id: 'algo-13',
    title: 'Bit Manipulation',
    category: 'Algorithm',
    summary: 'Direct manipulation of bits using bitwise operators for optimization.',
    complexity: {
      time: 'O(1)',
      space: 'O(1)'
    },
    description: `Bitwise operations operate at the binary level, making them extremely fast.

Common Operators:
- & (AND): Both bits 1.
- | (OR): Either bit 1.
- ^ (XOR): Different bits 1 (x^x = 0).
- ~ (NOT): Invert bits.
- << (Left Shift): Multiply by 2.
- >> (Right Shift): Divide by 2.

Tricks:
- Check even/odd: (x & 1) == 0
- Clear last set bit: x & (x - 1) (Used to count set bits or check power of 2)
- Get last set bit: x & -x`,
    implementation: `
def count_set_bits(n):
    count = 0
    while n > 0:
        n = n & (n - 1) # Removes the rightmost '1'
        count += 1
    return count

def single_number(nums):
    # Find the number that appears once (others appear twice)
    res = 0
    for n in nums:
        res ^= n
    return res
    `
  },
  {
    id: 'algo-14',
    title: 'Longest Common Subsequence',
    category: 'Algorithm',
    summary: 'Finds the longest subsequence present in two sequences in the same relative order.',
    complexity: {
      time: 'O(n * m)',
      space: 'O(n * m)'
    },
    description: `LCS is a classic 2D Dynamic Programming problem. A subsequence is not necessarily contiguous (e.g., "ace" is a subsequence of "abcde").

State:
dp[i][j] = Length of LCS between text1[0...i] and text2[0...j].

Transitions:
1. If chars match: dp[i][j] = 1 + dp[i-1][j-1]
2. If no match: dp[i][j] = max(dp[i-1][j], dp[i][j-1]) (Skip char from text1 or text2).`,
    implementation: `
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    return dp[m][n]
    `
  },
  {
    id: 'algo-15',
    title: 'Longest Palindromic Substring',
    category: 'Algorithm',
    summary: 'Finds the longest substring of a string that is a palindrome.',
    complexity: {
      time: 'O(n^2)',
      space: 'O(1)'
    },
    description: `A naive solution checks all substrings in O(n^3). We can optimize this by "Expanding Around Center".

A palindrome mirrors around its center.
- A center can be a character (e.g., "aba", center 'b').
- A center can be between characters (e.g., "abba", center between 'b's).

There are 2n - 1 centers in a string of length n. Expanding takes O(n), so total time is O(n^2).`,
    implementation: `
def longest_palindrome(s):
    res = ""
    
    for i in range(len(s)):
        # Odd length (center at i)
        l, r = i, i
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r - l + 1) > len(res):
                res = s[l:r+1]
            l -= 1
            r += 1
            
        # Even length (center between i and i+1)
        l, r = i, i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r - l + 1) > len(res):
                res = s[l:r+1]
            l -= 1
            r += 1
            
    return res
    `
  }
];

export const INTERVIEW_CONCEPTS: ReferenceItem[] = [
  {
    id: 'concept-1',
    title: 'Big O Notation',
    category: 'Concept',
    summary: 'Standard notation to describe the limiting behavior of a function.',
    complexity: { time: 'N/A', space: 'N/A' },
    description: `Big O notation describes the upper bound of an algorithm's complexity (worst-case scenario).

Hierarchy (Fastest to Slowest):
1. O(1) - Constant: Hash map lookup, Array access.
2. O(log n) - Logarithmic: Binary Search.
3. O(n) - Linear: Iterating a list.
4. O(n log n) - Linearithmic: Quick Sort, Merge Sort.
5. O(n^2) - Quadratic: Nested loops (Bubble Sort).
6. O(2^n) - Exponential: Recursive Fibonacci.
7. O(n!) - Factorial: Permutations.`,
    implementation: `
"""
COMPLEXITY CHEATSHEET

O(1):       return arr[0]
O(log n):   while n > 1: n = n // 2
O(n):       for i in range(n): ...
O(n log n): merge_sort(arr)
O(n^2):     for i in n: for j in n: ...
O(2^n):     def fib(n): return fib(n-1) + fib(n-2)
"""
    `
  },
  {
    id: 'concept-2',
    title: 'CAP Theorem',
    category: 'System Design',
    summary: 'A distributed system can only provide two of three: Consistency, Availability, Partition Tolerance.',
    complexity: { time: 'N/A', space: 'N/A' },
    description: `The CAP theorem states that in a distributed data store, you must choose 2 out of 3 guarantees.

1. Consistency (C): Every read receives the most recent write or an error.
2. Availability (A): Every request receives a response, without guarantee that it contains the most recent write.
3. Partition Tolerance (P): The system continues to operate despite an arbitrary number of messages being dropped (network failure).

Since network partitions (P) are inevitable in distributed systems, you must choose between C and A.
- CP (Consistency + Partition Tolerance): MongoDB, HBase. (System may return error if partition occurs).
- AP (Availability + Partition Tolerance): Cassandra, DynamoDB. (System always responds, but data might be stale - eventual consistency).`,
    implementation: `
"""
TRADE-OFFS:

CP (Consistent + Partition Tolerant)
- Good for: Banking, Payments.
- Risk: System becomes unavailable during network splits.

AP (Available + Partition Tolerant)
- Good for: Social Media Feeds, Shopping Carts.
- Risk: Users might see old data briefly (Eventual Consistency).

CA (Consistent + Available)
- Only possible if no partitions exist (Single Server).
- Not applicable for distributed systems.
"""
    `
  },
  {
    id: 'concept-3',
    title: 'ACID Properties',
    category: 'Concept',
    summary: 'Properties that guarantee database transactions are processed reliably.',
    complexity: { time: 'N/A', space: 'N/A' },
    description: `ACID ensures data integrity in Relational Databases (SQL).

1. Atomicity: "All or Nothing". If one part of a transaction fails, the entire transaction fails (rollback).
2. Consistency: Database transitions from one valid state to another (constraints enforced).
3. Isolation: Concurrent transactions do not interfere with each other.
4. Durability: Once a transaction is committed, it remains committed even in case of power loss.`,
    implementation: `
"""
Example: Bank Transfer (A to B)

BEGIN TRANSACTION;
  UPDATE accounts SET balance = balance - 100 WHERE id = 'A';
  UPDATE accounts SET balance = balance + 100 WHERE id = 'B';
COMMIT;

Atomicity: If B's update fails, A's update is rolled back.
Durability: Once COMMIT returns, the $100 move is permanent.
"""
    `
  },
  {
    id: 'concept-4',
    title: 'Load Balancing',
    category: 'System Design',
    summary: 'Distributing incoming network traffic across multiple servers.',
    complexity: { time: 'N/A', space: 'N/A' },
    description: `Load Balancers (LB) sit between the client and the server farm to improve availability and responsiveness.

Types:
- L4 (Transport Layer): Routes based on IP and Port. Simple, fast.
- L7 (Application Layer): Routes based on content (URL, Cookies, Headers). Smarter, more CPU intensive.

Algorithms:
- Round Robin: Sequential distribution.
- Least Connections: Send to server with fewest active connections.
- IP Hash: Consistent mapping of client IP to a specific server (sticky session).`,
    implementation: `
"""
Traffic -> [ Load Balancer ]
             /     |     \\
       [Web 1]  [Web 2]  [Web 3]

Health Checks:
LB periodically pings servers.
If [Web 2] fails 3 pings, LB stops sending traffic to it.
"""
    `
  },
  {
    id: 'concept-5',
    title: 'Caching Strategies',
    category: 'System Design',
    summary: 'Techniques to store temporary data to speed up retrieval.',
    complexity: { time: 'N/A', space: 'N/A' },
    description: `Caching reduces latency and DB load. Common strategies:

1. Cache-Aside (Lazy Loading):
   - App checks cache. If miss, read DB, write to cache, return to user.
   - Pros: Resilient to cache failure. Cons: Initial latency.

2. Write-Through:
   - App writes to Cache AND DB synchronously.
   - Pros: Data consistency. Cons: Higher write latency.

3. Write-Back:
   - App writes to Cache (fast). Cache writes to DB asynchronously.
   - Pros: Fast writes. Cons: Data loss risk if cache crashes before sync.`,
    implementation: `
"""
Cache-Aside Pseudocode:

def get_user(user_id):
    # 1. Check Cache
    user = cache.get(user_id)
    if user:
        return user
        
    # 2. Check DB
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # 3. Update Cache
    if user:
        cache.set(user_id, user, ttl=3600)
        
    return user
"""
    `
  },
  {
    id: 'concept-6',
    title: 'SOLID Principles',
    category: 'Concept',
    summary: 'Five design principles for maintainable Object-Oriented software.',
    complexity: { time: 'N/A', space: 'N/A' },
    description: `
S - Single Responsibility: A class should have one, and only one, reason to change.
O - Open/Closed: Open for extension, closed for modification.
L - Liskov Substitution: Subclasses must be substitutable for their base classes.
I - Interface Segregation: Many client-specific interfaces are better than one general-purpose interface.
D - Dependency Inversion: Depend upon abstractions, not concretions.`,
    implementation: `
"""
BAD (Violates SRP):
class User:
    def authenticate(self): ...
    def save_to_db(self): ...
    def send_email(self): ...

GOOD (SRP):
class UserAuth: ...
class UserRepository: ...
class EmailService: ...
"""
    `
  }
];