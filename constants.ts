
import type { Question } from '@/types';

export const QUESTIONS: Question[] = [
  // --- ARRAYS & HASHING ---
  {
    id: '1',
    title: 'Two Sum',
    difficulty: 'Easy',
    category: 'Arrays & Hashing',
    tags: ['Array', 'Hash Table'],
    companies: ['Google', 'Amazon', 'Apple', 'Meta', 'Microsoft'],
    description: `Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.`,
    examples: [
      { input: 'nums = [2,7,11,15], target = 9', output: '[0,1]', explanation: 'Because nums[0] + nums[1] == 9, we return [0, 1].' },
      { input: 'nums = [3,2,4], target = 6', output: '[1,2]' }
    ],
    constraints: [
      '2 <= nums.length <= 10^4',
      '-10^9 <= nums[i] <= 10^9',
      '-10^9 <= target <= 10^9',
      'Only one valid answer exists.'
    ],
    officialSolution: `
Approach: Hash Map (One-pass)

We can do this in one pass. While we iterate and inserting elements into the hash table, we also look back to check if current element's complement already exists in the hash table. If it exists, we have found a solution and return immediately.

Time Complexity: O(n)
Space Complexity: O(n)

\`\`\`python
def twoSum(nums, target):
    prevMap = {}  # val : index
    for i, n in enumerate(nums):
        diff = target - n
        if diff in prevMap:
            return [prevMap[diff], i]
        prevMap[n] = i
\`\`\`
    `
  },
  {
    id: '2',
    title: 'Best Time to Buy and Sell Stock',
    difficulty: 'Easy',
    category: 'Arrays & Hashing',
    tags: ['Array', 'Dynamic Programming'],
    companies: ['Google', 'Amazon', 'Meta'],
    description: `You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.`,
    examples: [
      { input: 'prices = [7,1,5,3,6,4]', output: '5', explanation: 'Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.' },
      { input: 'prices = [7,6,4,3,1]', output: '0', explanation: 'In this case, no transactions are done and the max profit = 0.' }
    ],
    constraints: [
      '1 <= prices.length <= 10^5',
      '0 <= prices[i] <= 10^4'
    ],
    officialSolution: `
Approach: One Pass

We need to find the largest peak following the smallest valley. We can maintain two variables - minprice and maxprofit. With minprice corresponding to the lowest price so far and maxprofit corresponding to the maximum profit obtained so far.

Iterate through the array:
1. Update min_price if current price is lower.
2. Else update max_profit if (current_price - min_price) is greater.

Time Complexity: O(n)
Space Complexity: O(1)

\`\`\`python
def maxProfit(prices):
    l, r = 0, 1 # Left=buy, Right=sell
    maxP = 0
    
    while r < len(prices):
        if prices[l] < prices[r]:
            profit = prices[r] - prices[l]
            maxP = max(maxP, profit)
        else:
            l = r
        r += 1
        
    return maxP
\`\`\`
    `
  },
  {
    id: '3',
    title: 'Contains Duplicate',
    difficulty: 'Easy',
    category: 'Arrays & Hashing',
    tags: ['Array', 'Hash Table', 'Sorting'],
    description: `Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.`,
    examples: [
      { input: 'nums = [1,2,3,1]', output: 'true' },
      { input: 'nums = [1,2,3,4]', output: 'false' }
    ],
    constraints: [
      '1 <= nums.length <= 10^5',
      '-10^9 <= nums[i] <= 10^9'
    ],
    officialSolution: `
Approach: Hash Set

Use a hash set to store elements as we iterate. If an element is already in the set, return true. If we finish the loop, return false.

Time Complexity: O(n)
Space Complexity: O(n)

\`\`\`python
def containsDuplicate(nums):
    hashset = set()
    for n in nums:
        if n in hashset:
            return True
        hashset.add(n)
    return False
\`\`\`
    `
  },
  {
    id: '4',
    title: 'Product of Array Except Self',
    difficulty: 'Medium',
    category: 'Arrays & Hashing',
    tags: ['Array', 'Prefix Sum'],
    companies: ['Google', 'Amazon', 'Meta', 'Microsoft'],
    description: `Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.`,
    examples: [
      { input: 'nums = [1,2,3,4]', output: '[24,12,8,6]' },
      { input: 'nums = [-1,1,0,-3,3]', output: '[0,0,9,0,0]' }
    ],
    constraints: [
      '2 <= nums.length <= 10^5',
      '-30 <= nums[i] <= 30'
    ],
    officialSolution: `
Approach: Left and Right Product Lists

1. Initialize two arrays L and R. L[i] contains product of all elements to left of i. R[i] contains product of all elements to right of i.
2. answer[i] = L[i] * R[i].

Optimization (O(1) space):
Use the output array to store L, then use a variable R to store the running right product while updating the output array in reverse.

Time Complexity: O(n)
Space Complexity: O(1) (ignoring output array)

\`\`\`python
def productExceptSelf(nums):
    res = [1] * len(nums)
    
    prefix = 1
    for i in range(len(nums)):
        res[i] = prefix
        prefix *= nums[i]
        
    postfix = 1
    for i in range(len(nums) - 1, -1, -1):
        res[i] *= postfix
        postfix *= nums[i]
        
    return res
\`\`\`
    `
  },
  {
    id: '5',
    title: 'Maximum Subarray',
    difficulty: 'Medium',
    category: 'Arrays & Hashing',
    tags: ['Array', 'Divide and Conquer', 'Dynamic Programming'],
    companies: ['Google', 'LinkedIn', 'Microsoft'],
    description: `Given an integer array nums, find the subarray with the largest sum, and return its sum.`,
    examples: [
      { input: 'nums = [-2,1,-3,4,-1,2,1,-5,4]', output: '6', explanation: 'The subarray [4,-1,2,1] has the largest sum 6.' },
      { input: 'nums = [1]', output: '1' }
    ],
    constraints: [
      '1 <= nums.length <= 10^5',
      '-10^4 <= nums[i] <= 10^4'
    ],
    officialSolution: `
Approach: Kadane's Algorithm

Iterate through the array, maintaining a current_sum.
If current_sum < 0, reset it to 0 (effectively discarding the negative prefix).
Add the current number to current_sum.
Update max_sum if current_sum > max_sum.

Time Complexity: O(n)
Space Complexity: O(1)

\`\`\`python
def maxSubArray(nums):
    maxSub = nums[0]
    curSum = 0
    
    for n in nums:
        if curSum < 0:
            curSum = 0
        curSum += n
        maxSub = max(maxSub, curSum)
        
    return maxSub
\`\`\`
    `
  },

  // --- POINTERS ---
  {
    id: '6',
    title: 'Container With Most Water',
    difficulty: 'Medium',
    category: 'Two Pointers',
    tags: ['Array', 'Two Pointers', 'Greedy'],
    companies: ['Google', 'Amazon', 'Meta'],
    description: `You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water. Return the maximum amount of water a container can store.`,
    examples: [
      { input: 'height = [1,8,6,2,5,4,8,3,7]', output: '49' },
      { input: 'height = [1,1]', output: '1' }
    ],
    constraints: [
      'n == height.length',
      '2 <= n <= 10^5',
      '0 <= height[i] <= 10^4'
    ],
    officialSolution: `
Approach: Two Pointers

Start with pointers at both ends of the array. The area is determined by the shorter line.
1. Calculate area. Update max.
2. Move the pointer pointing to the shorter line inward (hoping to find a taller line).
3. Repeat until pointers meet.

Time Complexity: O(n)
Space Complexity: O(1)

\`\`\`python
def maxArea(height):
    l, r = 0, len(height) - 1
    res = 0
    
    while l < r:
        area = (r - l) * min(height[l], height[r])
        res = max(res, area)
        
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
            
    return res
\`\`\`
    `
  },
  {
    id: '7',
    title: '3Sum',
    difficulty: 'Medium',
    category: 'Two Pointers',
    tags: ['Array', 'Two Pointers', 'Sorting'],
    companies: ['Google', 'Amazon', 'Meta'],
    description: `Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.`,
    examples: [
      { input: 'nums = [-1,0,1,2,-1,-4]', output: '[[-1,-1,2],[-1,0,1]]' },
      { input: 'nums = [0,1,1]', output: '[]' }
    ],
    constraints: [
      '3 <= nums.length <= 3000',
      '-10^5 <= nums[i] <= 10^5'
    ],
    officialSolution: `
Approach: Sorting + Two Pointers

1. Sort the array.
2. Iterate through the array with variable i.
3. For each i, use two pointers (left = i+1, right = n-1) to find pairs that sum to -nums[i].
4. Skip duplicates to avoid repeating triplets.

Time Complexity: O(n^2)
Space Complexity: O(1) (or O(n) for sorting)

\`\`\`python
def threeSum(nums):
    res = []
    nums.sort()
    
    for i, a in enumerate(nums):
        if i > 0 and a == nums[i - 1]:
            continue
            
        l, r = i + 1, len(nums) - 1
        while l < r:
            threeSum = a + nums[l] + nums[r]
            if threeSum > 0:
                r -= 1
            elif threeSum < 0:
                l += 1
            else:
                res.append([a, nums[l], nums[r]])
                l += 1
                while nums[l] == nums[l - 1] and l < r:
                    l += 1
                    
    return res
\`\`\`
    `
  },

  // --- STRING ---
  {
    id: '8',
    title: 'Longest Substring Without Repeating Characters',
    difficulty: 'Medium',
    category: 'Sliding Window',
    tags: ['Hash Table', 'String', 'Sliding Window'],
    companies: ['Google', 'Amazon', 'Meta', 'Microsoft', 'Bloomberg'],
    description: 'Given a string s, find the length of the longest substring without repeating characters.',
    examples: [
      { input: 's = "abcabcbb"', output: '3', explanation: 'The answer is "abc", with the length of 3.' },
      { input: 's = "bbbbb"', output: '1', explanation: 'The answer is "b", with the length of 1.' }
    ],
    constraints: [
      '0 <= s.length <= 5 * 10^4',
      's consists of English letters, digits, symbols and spaces.'
    ],
    officialSolution: `
Approach: Sliding Window

We use a hash set (or array for characters) to store the characters in the current window [i, j).
We slide index j to the right. If s[j] is already in the set, we slide i to the right, removing elements from the set, until s[j] is no longer in the set.
We update the max length at each step.

Time Complexity: O(n)
Space Complexity: O(min(m, n)) where m is charset size.

\`\`\`python
def lengthOfLongestSubstring(s):
    charSet = set()
    l = 0
    res = 0
    
    for r in range(len(s)):
        while s[r] in charSet:
            charSet.remove(s[l])
            l += 1
        charSet.add(s[r])
        res = max(res, r - l + 1)
        
    return res
\`\`\`
    `
  },
  {
    id: '9',
    title: 'Valid Anagram',
    difficulty: 'Easy',
    category: 'Strings',
    tags: ['Hash Table', 'String', 'Sorting'],
    description: `Given two strings s and t, return true if t is an anagram of s, and false otherwise.`,
    examples: [
      { input: 's = "anagram", t = "nagaram"', output: 'true' },
      { input: 's = "rat", t = "car"', output: 'false' }
    ],
    constraints: [
      '1 <= s.length, t.length <= 5 * 10^4',
      's and t consist of lowercase English letters.'
    ],
    officialSolution: `
Approach: Frequency Counter

1. Check if lengths are different. If so, return false.
2. Use an array of size 26 (for lowercase English letters).
3. Iterate through s, incrementing counts.
4. Iterate through t, decrementing counts.
5. If any count is non-zero, return false.

Time Complexity: O(n)
Space Complexity: O(1) (fixed size array of 26)

\`\`\`python
def isAnagram(s, t):
    if len(s) != len(t):
        return False
        
    countS, countT = {}, {}
    
    for i in range(len(s)):
        countS[s[i]] = 1 + countS.get(s[i], 0)
        countT[t[i]] = 1 + countT.get(t[i], 0)
        
    return countS == countT
\`\`\`
    `
  },
  {
    id: '10',
    title: 'Valid Parentheses',
    difficulty: 'Easy',
    category: 'Stack',
    tags: ['String', 'Stack'],
    companies: ['Google', 'Amazon', 'Meta', 'Microsoft', 'LinkedIn'],
    description: `Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.`,
    examples: [
      { input: 's = "()[]{}"', output: 'true' },
      { input: 's = "(]"', output: 'false' }
    ],
    constraints: [
      '1 <= s.length <= 10^4',
      's consists of parentheses only.'
    ],
    officialSolution: `
Approach: Stack

1. Initialize an empty stack.
2. Iterate through characters.
3. If char is an opening bracket, push to stack.
4. If char is a closing bracket, check if stack is empty or if top of stack matches the pair. Pop if match, else return false.
5. Return true if stack is empty at the end.

Time Complexity: O(n)
Space Complexity: O(n)

\`\`\`python
def isValid(s):
    stack = []
    closeToOpen = { ")" : "(", "]" : "[", "}" : "{" }
    
    for c in s:
        if c in closeToOpen:
            if stack and stack[-1] == closeToOpen[c]:
                stack.pop()
            else:
                return False
        else:
            stack.append(c)
            
    return True if not stack else False
\`\`\`
    `
  },
  {
    id: '11',
    title: 'Valid Palindrome',
    difficulty: 'Easy',
    category: 'Two Pointers',
    tags: ['Two Pointers', 'String'],
    companies: ['Meta', 'Google'],
    description: `A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Given a string s, return true if it is a palindrome, or false otherwise.`,
    examples: [
      { input: 's = "A man, a plan, a canal: Panama"', output: 'true' },
      { input: 's = "race a car"', output: 'false' }
    ],
    constraints: [
      '1 <= s.length <= 2 * 10^5',
      's consists only of printable ASCII characters.'
    ],
    officialSolution: `
Approach: Two Pointers

1. Initialize two pointers, left = 0 and right = s.length - 1.
2. While left < right:
   - Move left forward until it hits an alphanumeric char.
   - Move right backward until it hits an alphanumeric char.
   - Compare characters (case insensitive). If mismatch, return false.
   - Move both pointers inward.

Time Complexity: O(n)
Space Complexity: O(1)

\`\`\`python
def isPalindrome(s):
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not alphaNum(s[l]):
            l += 1
        while r > l and not alphaNum(s[r]):
            r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l, r = l + 1, r - 1
    return True
    
def alphaNum(c):
    return (ord('A') <= ord(c) <= ord('Z') or 
            ord('a') <= ord(c) <= ord('z') or 
            ord('0') <= ord(c) <= ord('9'))
\`\`\`
    `
  },

  // --- LINKED LIST ---
  {
    id: '12',
    title: 'Reverse Linked List',
    difficulty: 'Easy',
    category: 'Linked List',
    tags: ['Linked List', 'Recursion'],
    description: 'Given the head of a singly linked list, reverse the list, and return the reversed list.',
    examples: [
      { input: 'head = [1,2,3,4,5]', output: '[5,4,3,2,1]' },
      { input: 'head = [1,2]', output: '[2,1]' }
    ],
    constraints: [
      'Number of nodes is [0, 5000].',
      '-5000 <= Node.val <= 5000'
    ],
    officialSolution: `
Approach: Iterative

Initialize three pointers: prev as NULL, curr as head, and next as NULL.
Iterate through the linked list. In loop:
1. Store next node
2. Change next of current
3. Move prev and curr one step forward

Time Complexity: O(n)
Space Complexity: O(1)

\`\`\`python
def reverseList(head):
    prev, curr = None, head
    
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
        
    return prev
\`\`\`
    `
  },
  {
    id: '13',
    title: 'Merge Two Sorted Lists',
    difficulty: 'Easy',
    category: 'Linked List',
    tags: ['Linked List', 'Recursion'],
    companies: ['Google', 'Amazon', 'Microsoft'],
    description: `You are given the heads of two sorted linked lists list1 and list2. Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists. Return the head of the merged linked list.`,
    examples: [
      { input: 'list1 = [1,2,4], list2 = [1,3,4]', output: '[1,1,2,3,4,4]' },
      { input: 'list1 = [], list2 = []', output: '[]' }
    ],
    constraints: [
      'Number of nodes in both lists is [0, 50].',
      '-100 <= Node.val <= 100'
    ],
    officialSolution: `
Approach: Iterative with Dummy Node

1. Create a dummy node and a current pointer pointing to it.
2. While both list1 and list2 are not null:
   - If list1.val < list2.val, attach list1 to current.next and move list1.
   - Else, attach list2 to current.next and move list2.
   - Move current pointer.
3. Attach remaining non-null list to current.next.
4. Return dummy.next.

Time Complexity: O(n + m)
Space Complexity: O(1)

\`\`\`python
def mergeTwoLists(list1, list2):
    dummy = ListNode()
    tail = dummy
    
    while list1 and list2:
        if list1.val < list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
        
    if list1:
        tail.next = list1
    elif list2:
        tail.next = list2
        
    return dummy.next
\`\`\`
    `
  },
  {
    id: '14',
    title: 'Linked List Cycle',
    difficulty: 'Easy',
    category: 'Linked List',
    tags: ['Hash Table', 'Linked List', 'Two Pointers'],
    description: `Given head, the head of a linked list, determine if the linked list has a cycle in it. Return true if there is a cycle in the linked list. Otherwise, return false.`,
    examples: [
      { input: 'head = [3,2,0,-4], pos = 1', output: 'true' },
      { input: 'head = [1,2], pos = 0', output: 'true' }
    ],
    constraints: [
      'Number of nodes is [0, 10^4].',
      '-10^5 <= Node.val <= 10^5'
    ],
    officialSolution: `
Approach: Floyd's Cycle Finding Algorithm (Tortoise and Hare)

1. Initialize two pointers, slow and fast, both at head.
2. While fast and fast.next are not null:
   - Move slow by 1 step.
   - Move fast by 2 steps.
   - If slow == fast, a cycle exists. Return true.
3. If loop finishes, return false.

Time Complexity: O(n)
Space Complexity: O(1)

\`\`\`python
def hasCycle(head):
    slow, fast = head, head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
            
    return False
\`\`\`
    `
  },
  {
    id: '15',
    title: 'Remove Nth Node From End of List',
    difficulty: 'Medium',
    category: 'Linked List',
    tags: ['Linked List', 'Two Pointers'],
    companies: ['Google', 'Facebook'],
    description: `Given the head of a linked list, remove the nth node from the end of the list and return its head.`,
    examples: [
      { input: 'head = [1,2,3,4,5], n = 2', output: '[1,2,3,5]' },
      { input: 'head = [1], n = 1', output: '[]' }
    ],
    constraints: [
      'Number of nodes is [1, 30].',
      '1 <= n <= sz'
    ],
    officialSolution: `
Approach: Two Pointers

1. Create a dummy node pointing to head (handles edge case of removing first node).
2. Initialize two pointers, first and second, at dummy.
3. Move first pointer n + 1 steps forward.
4. Move both first and second pointers forward until first reaches the end.
5. second.next is now the node to remove. Set second.next = second.next.next.

Time Complexity: O(n)
Space Complexity: O(1)

\`\`\`python
def removeNthFromEnd(head, n):
    dummy = ListNode(0, head)
    left = dummy
    right = head
    
    while n > 0 and right:
        right = right.next
        n -= 1
        
    while right:
        left = left.next
        right = right.next
        
    left.next = left.next.next
    return dummy.next
\`\`\`
    `
  },

  // --- TREES ---
  {
    id: '16',
    title: 'Maximum Depth of Binary Tree',
    difficulty: 'Easy',
    category: 'Trees',
    tags: ['Tree', 'DFS', 'BFS', 'Binary Tree'],
    description: `Given the root of a binary tree, return its maximum depth.
A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.`,
    examples: [
      { input: 'root = [3,9,20,null,null,15,7]', output: '3' },
      { input: 'root = [1,null,2]', output: '2' }
    ],
    constraints: [
      'Number of nodes is [0, 10^4].',
      '-100 <= Node.val <= 100'
    ],
    officialSolution: `
Approach: Recursive DFS

1. Base case: if root is null, return 0.
2. Recursive step: return 1 + max(maxDepth(root.left), maxDepth(root.right)).

Time Complexity: O(n)
Space Complexity: O(n) (worst case skewed tree) or O(log n) (balanced)

\`\`\`python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
\`\`\`
    `
  },
  {
    id: '17',
    title: 'Invert Binary Tree',
    difficulty: 'Easy',
    category: 'Trees',
    tags: ['Tree', 'DFS', 'BFS', 'Binary Tree'],
    companies: ['Google', 'Homebrew'],
    description: `Given the root of a binary tree, invert the tree, and return its root.`,
    examples: [
      { input: 'root = [4,2,7,1,3,6,9]', output: '[4,7,2,9,6,3,1]' },
      { input: 'root = [2,1,3]', output: '[2,3,1]' }
    ],
    constraints: [
      'Number of nodes is [0, 100].',
      '-100 <= Node.val <= 100'
    ],
    officialSolution: `
Approach: Recursive

1. Base case: if root is null, return null.
2. Swap root.left and root.right.
3. Recursively invert root.left.
4. Recursively invert root.right.

Time Complexity: O(n)
Space Complexity: O(n)

\`\`\`python
def invertTree(root):
    if not root:
        return None
    
    # swap children
    root.left, root.right = root.right, root.left
    
    invertTree(root.left)
    invertTree(root.right)
    return root
\`\`\`
    `
  },
  {
    id: '18',
    title: 'Same Tree',
    difficulty: 'Easy',
    category: 'Trees',
    tags: ['Tree', 'DFS', 'BFS', 'Binary Tree'],
    description: `Given the roots of two binary trees p and q, write a function to check if they are the same or not. Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.`,
    examples: [
      { input: 'p = [1,2,3], q = [1,2,3]', output: 'true' },
      { input: 'p = [1,2], q = [1,null,2]', output: 'false' }
    ],
    constraints: [
      'Number of nodes is [0, 100].'
    ],
    officialSolution: `
Approach: Recursive

1. If both p and q are null, return true.
2. If one is null and other isn't, or values differ, return false.
3. Return isSameTree(p.left, q.left) && isSameTree(p.right, q.right).

Time Complexity: O(n)
Space Complexity: O(n)

\`\`\`python
def isSameTree(p, q):
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
        
    return (isSameTree(p.left, q.left) and 
            isSameTree(p.right, q.right))
\`\`\`
    `
  },
  {
    id: '19',
    title: 'Subtree of Another Tree',
    difficulty: 'Easy',
    category: 'Trees',
    tags: ['Tree', 'DFS', 'Binary Tree', 'String Matching', 'Hash Function'],
    description: `Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.`,
    examples: [
      { input: 'root = [3,4,5,1,2], subRoot = [4,1,2]', output: 'true' },
      { input: 'root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]', output: 'false' }
    ],
    constraints: [
      'root nodes: [1, 2000]',
      'subRoot nodes: [1, 1000]'
    ],
    officialSolution: `
Approach: DFS + Helper

1. Create a helper function isSameTree(p, q).
2. In main function:
   - If subRoot is null, return true.
   - If root is null, return false.
   - If isSameTree(root, subRoot) is true, return true.
   - Return isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot).

Time Complexity: O(n * m)
Space Complexity: O(n)

\`\`\`python
def isSubtree(root, subRoot):
    if not subRoot: return True
    if not root: return False
    
    if sameTree(root, subRoot):
        return True
    
    return (isSubtree(root.left, subRoot) or 
            isSubtree(root.right, subRoot))

def sameTree(s, t):
    if not s and not t: return True
    if s and t and s.val == t.val:
        return (sameTree(s.left, t.left) and 
                sameTree(s.right, t.right))
    return False
\`\`\`
    `
  },

  // --- HEAP ---
  {
    id: '20',
    title: 'Top K Frequent Elements',
    difficulty: 'Medium',
    category: 'Heap',
    tags: ['Array', 'Hash Table', 'Divide and Conquer', 'Sorting', 'Heap', 'Bucket Sort'],
    companies: ['Google', 'Amazon', 'Meta'],
    description: `Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.`,
    examples: [
      { input: 'nums = [1,1,1,2,2,3], k = 2', output: '[1,2]' },
      { input: 'nums = [1], k = 1', output: '[1]' }
    ],
    constraints: [
      '1 <= nums.length <= 10^5',
      'k is in range [1, number of unique elements]'
    ],
    officialSolution: `
Approach: Hash Map + Min Heap (or Bucket Sort)

1. Build a hash map of frequency counts.
2. Keep a Min Heap of size k.
3. Iterate through map entries. Push to heap. If heap size > k, pop min.
4. Remaining elements in heap are top k.

Time Complexity: O(n log k)
Space Complexity: O(n + k)

\`\`\`python
def topKFrequent(nums, k):
    count = {}
    freq = [[] for i in range(len(nums) + 1)]
    
    for n in nums:
        count[n] = 1 + count.get(n, 0)
    for n, c in count.items():
        freq[c].append(n)
        
    res = []
    for i in range(len(freq) - 1, 0, -1):
        for n in freq[i]:
            res.append(n)
            if len(res) == k:
                return res
\`\`\`
    `
  },

  // --- GRAPH ---
  {
    id: '21',
    title: 'Number of Islands',
    difficulty: 'Medium',
    category: 'Graphs',
    tags: ['Array', 'DFS', 'BFS', 'Union Find', 'Matrix'],
    companies: ['Google', 'Amazon', 'Microsoft', 'Meta'],
    description: `Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.`,
    examples: [
      { input: 'grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]', output: '1' },
      { input: 'grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]', output: '3' }
    ],
    constraints: [
      'm == grid.length, n == grid[i].length',
      '1 <= m, n <= 300'
    ],
    officialSolution: `
Approach: DFS (or BFS)

1. Iterate through every cell in grid.
2. If cell is '1', increment island count and start DFS.
3. In DFS:
   - Boundary checks (return if out of bounds or '0').
   - Mark current cell as '0' (visited).
   - Recursively call DFS on 4 directions.

Time Complexity: O(m * n)
Space Complexity: O(m * n) (recursion stack)

\`\`\`python
def numIslands(grid):
    if not grid: return 0
    
    rows, cols = len(grid), len(grid[0])
    visit = set()
    islands = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            grid[r][c] == "0" or (r, c) in visit):
            return
        visit.add((r, c))
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for dr, dc in directions:
            dfs(r + dr, c + dc)
            
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visit:
                islands += 1
                dfs(r, c)
    return islands
\`\`\`
    `
  },
  {
    id: '22',
    title: 'Clone Graph',
    difficulty: 'Medium',
    category: 'Graphs',
    tags: ['Hash Table', 'DFS', 'BFS', 'Graph'],
    companies: ['Google', 'Amazon', 'Meta'],
    description: `Given a reference of a node in a connected undirected graph. Return a deep copy (clone) of the graph.`,
    examples: [
      { input: 'adjList = [[2,4],[1,3],[2,4],[1,3]]', output: '[[2,4],[1,3],[2,4],[1,3]]' }
    ],
    constraints: [
      'Number of nodes [0, 100].',
      'Node values are unique.'
    ],
    officialSolution: `
Approach: DFS with Hash Map

1. Use a Hash Map to store (originalNode -> cloneNode) mapping to handle cycles/visited nodes.
2. Function clone(node):
   - If node in map, return map[node].
   - Create new node with node.val.
   - Add to map.
   - Iterate neighbors, add clone(neighbor) to new node's neighbors.
   - Return new node.

Time Complexity: O(V + E)
Space Complexity: O(V)

\`\`\`python
def cloneGraph(node):
    oldToNew = {}
    
    def dfs(node):
        if node in oldToNew:
            return oldToNew[node]
        
        copy = Node(node.val)
        oldToNew[node] = copy
        for nei in node.neighbors:
            copy.neighbors.append(dfs(nei))
        return copy
        
    return dfs(node) if node else None
\`\`\`
    `
  },

  // --- DYNAMIC PROGRAMMING ---
  {
    id: '23',
    title: 'Climbing Stairs',
    difficulty: 'Easy',
    category: 'Dynamic Programming',
    tags: ['Math', 'Dynamic Programming', 'Memoization'],
    description: `You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?`,
    examples: [
      { input: 'n = 2', output: '2', explanation: '1. 1 step + 1 step\n2. 2 steps' },
      { input: 'n = 3', output: '3', explanation: '1. 1+1+1\n2. 1+2\n3. 2+1' }
    ],
    constraints: [
      '1 <= n <= 45'
    ],
    officialSolution: `
Approach: Dynamic Programming (Fibonacci)

dp[i] = ways to reach step i.
dp[i] = dp[i-1] + dp[i-2].
Base cases: dp[1] = 1, dp[2] = 2.

Optimization: store only prev1 and prev2 variables.

Time Complexity: O(n)
Space Complexity: O(1)

\`\`\`python
def climbStairs(n):
    one, two = 1, 1
    
    for i in range(n - 1):
        temp = one
        one = one + two
        two = temp
        
    return one
\`\`\`
    `
  },
  {
    id: '24',
    title: 'Coin Change',
    difficulty: 'Medium',
    category: 'Dynamic Programming',
    tags: ['Array', 'Dynamic Programming', 'BFS'],
    companies: ['Amazon', 'Microsoft', 'Google'],
    description: `You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.`,
    examples: [
      { input: 'coins = [1,2,5], amount = 11', output: '3' },
      { input: 'coins = [2], amount = 3', output: '-1' }
    ],
    constraints: [
      '1 <= coins.length <= 12',
      '0 <= amount <= 10^4'
    ],
    officialSolution: `
Approach: DP (Bottom Up)

dp[i] = min coins to make amount i.
Initialize dp array with amount + 1 (infinity). dp[0] = 0.
Iterate i from 1 to amount:
  Iterate through coins c:
    if i - c >= 0: dp[i] = min(dp[i], 1 + dp[i-c])

If dp[amount] > amount, return -1, else return dp[amount].

Time Complexity: O(amount * len(coins))
Space Complexity: O(amount)

\`\`\`python
def coinChange(coins, amount):
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    
    for a in range(1, amount + 1):
        for c in coins:
            if a - c >= 0:
                dp[a] = min(dp[a], 1 + dp[a - c])
                
    return dp[amount] if dp[amount] != amount + 1 else -1
\`\`\`
    `
  },
  {
    id: '25',
    title: 'Longest Increasing Subsequence',
    difficulty: 'Medium',
    category: 'Dynamic Programming',
    tags: ['Array', 'Binary Search', 'Dynamic Programming'],
    companies: ['Google', 'Amazon', 'Microsoft'],
    description: `Given an integer array nums, return the length of the longest strictly increasing subsequence.`,
    examples: [
      { input: 'nums = [10,9,2,5,3,7,101,18]', output: '4', explanation: 'The longest increasing subsequence is [2,3,7,101], therefore the length is 4.' },
      { input: 'nums = [0,1,0,3,2,3]', output: '4' }
    ],
    constraints: [
      '1 <= nums.length <= 2500',
      '-10^4 <= nums[i] <= 10^4'
    ],
    officialSolution: `
Approach: DP

dp[i] = length of LIS ending at index i.
Initialize dp array with 1s.
Iterate i from 1 to n:
  Iterate j from 0 to i:
    if nums[i] > nums[j]: dp[i] = max(dp[i], dp[j] + 1)
Return max(dp).

Time Complexity: O(n^2) (Can be O(n log n) with Binary Search Patience Sort)
Space Complexity: O(n)

\`\`\`python
def lengthOfLIS(nums):
    LIS = [1] * len(nums)
    
    for i in range(len(nums) - 1, -1, -1):
        for j in range(i + 1, len(nums)):
            if nums[i] < nums[j]:
                LIS[i] = max(LIS[i], 1 + LIS[j])
    return max(LIS)
\`\`\`
    `
  },

  // --- DATA STRUCTURE DESIGN ---
  {
    id: '26',
    title: 'Design LFU Cache',
    difficulty: 'Hard',
    category: 'System Design',
    tags: ['Hash Table', 'Linked List', 'Design', 'Doubly Linked List'],
    companies: ['Google', 'Amazon'],
    description: `Design and implement a data structure for a Least Frequently Used (LFU) cache.
    
Implement the LFUCache class:
- LFUCache(int capacity) Initializes the object with the capacity of the data structure.
- int get(int key) Gets the value of the key if the key exists in the cache. Otherwise, returns -1.
- void put(int key, int value) Update the value of the key if present, or inserts the key if not already present. When the cache reaches its capacity, it should invalidate and remove the least frequently used key before inserting a new item. For this problem, when there is a tie (i.e., two or more keys have the same frequency), the least recently used key would be invalidated.

To determine the least frequently used key, a use counter is maintained for each key in the cache. The key with the smallest use counter is the least frequently used key.

When a key is first inserted into the cache, its use counter is set to 1 (due to the put operation). The use counter for a key in the cache is incremented either a get or put operation is called on it.

The functions get and put must each run in O(1) average time complexity.`,
    examples: [
      {
        input: '["LFUCache", "put", "put", "get", "put", "get", "get", "put", "get", "get", "get"]\n[[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [4, 4], [1], [3], [4]]',
        output: '[null, null, null, 1, null, -1, 3, null, -1, 3, 4]'
      }
    ],
    constraints: [
      '0 <= capacity <= 10^4',
      '0 <= key <= 10^5',
      '0 <= value <= 10^9',
      'At most 2 * 10^5 calls will be made to get and put.'
    ],
    officialSolution: `
Approach: Two Hash Maps + Doubly Linked List (or LinkedHashSet)

1. vals: key -> value
2. counts: key -> frequency
3. lists: frequency -> Doubly Linked List of keys (to handle LRU within same frequency)
4. minFreq: tracks the minimum frequency currently in the cache.

GET(key):
- If key not in vals, return -1.
- Get freq from counts.
- Remove key from lists[freq]. If lists[freq] empty and freq == minFreq, increment minFreq.
- Increment freq. Add key to lists[freq+1].
- Update counts[key].
- Return value.

PUT(key, value):
- If capacity 0, return.
- If key exists: update value, then call GET(key) logic to update frequency.
- If key does not exist:
  - If size == capacity:
    - Remove first element from lists[minFreq].
    - Remove from vals and counts.
  - Insert key with value.
  - counts[key] = 1.
  - minFreq = 1.
  - Add to lists[1].

Time Complexity: O(1) for get and put.
Space Complexity: O(n)
    `
  },
  {
    id: '27',
    title: 'Design Binary Search Tree',
    difficulty: 'Medium',
    category: 'Data Structures',
    tags: ['Tree', 'Design', 'Binary Search Tree', 'Binary Tree'],
    description: `Design a Binary Search Tree class that supports the following operations:
    
1. insert(val): Inserts a value into the BST.
2. search(val): Returns true if the value exists in the BST, false otherwise.
3. remove(val): Removes a value from the BST if it exists.
4. inorder(): Returns a list of values in sorted order.

Note: A valid BST is defined as follows:
- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys greater than the node's key.
- Both the left and right subtrees must also be binary search trees.`,
    examples: [
      {
        input: 'insert(5), insert(3), insert(7), search(3), remove(3), search(3)',
        output: 'null, null, null, true, null, false'
      }
    ],
    constraints: [
      'Values are unique integers.',
      'At most 1000 operations.'
    ],
    officialSolution: `
Approach: Recursive Class Design

Node structure: { val, left, right }

INSERT(val):
- If root is null, create new node.
- If val < root.val, recurse left.
- If val > root.val, recurse right.

SEARCH(val):
- If root is null, return false.
- If val == root.val, return true.
- If val < root.val, recurse left; else right.

REMOVE(val):
- Search for node.
- Case 1: Leaf node -> delete.
- Case 2: One child -> replace node with child.
- Case 3: Two children -> Find successor (min of right subtree), copy value, delete successor.

INORDER():
- Recursive: inorder(left) + [val] + inorder(right).

Time Complexity: O(h) where h is height (O(log n) balanced, O(n) worst).
Space Complexity: O(n) to store nodes.
    `
  },
  {
    id: '28',
    title: 'Design Min Heap',
    difficulty: 'Medium',
    category: 'Data Structures',
    tags: ['Array', 'Design', 'Heap', 'Priority Queue'],
    description: `Design a Min Heap (Priority Queue) class that supports:

1. push(val): Adds an integer to the heap.
2. pop(): Removes and returns the smallest element.
3. peek(): Returns the smallest element without removing it.
4. size(): Returns the number of elements.

You should implement this using an array/list internally.`,
    examples: [
      { input: 'push(5), push(2), push(8), peek(), pop(), peek()', output: 'null, null, null, 2, 2, 5' }
    ],
    constraints: [
      'Operations should maintain heap property: parent <= children.'
    ],
    officialSolution: `
Approach: Array Implementation

Array indices:
- Parent(i) = (i - 1) // 2
- Left(i) = 2*i + 1
- Right(i) = 2*i + 2

PUSH(val):
- Append val to end of array.
- Heapify Up (Bubble Up): Swap with parent while val < parent.

POP():
- Swap root (index 0) with last element.
- Remove last element (return value).
- Heapify Down (Bubble Down) from root: Swap with smaller child until heap property satisfied.

PEEK():
- Return array[0].

Time Complexity: Push O(log n), Pop O(log n), Peek O(1).
Space Complexity: O(n).
    `
  },
  {
    id: '29',
    title: 'Design Dynamic Array',
    difficulty: 'Easy',
    category: 'Data Structures',
    tags: ['Array', 'Design'],
    description: `Design a Dynamic Array (like Java ArrayList or C++ vector).

Implement the DynamicArray class:
- DynamicArray(int capacity): Initialize with capacity.
- int get(int i): Return element at index i.
- void set(int i, int n): Set element at index i to n.
- void pushback(int n): Push n to the end. Resize if necessary (double capacity).
- int popback(): Pop and return element from end.
- void resize(): Double the capacity.
- int getSize(): Return current number of elements.
- int getCapacity(): Return current capacity.`,
    examples: [
      { input: 'Array(2), pushback(1), pushback(2), get(1), pushback(3), getCapacity()', output: 'null, null, null, 2, null, 4' }
    ],
    constraints: [
      '0 <= i < size'
    ],
    officialSolution: `
Approach: Resizable Array

Maintain 'arr' (internal array), 'size' (current elements), 'capacity' (max elements).

PUSHBACK(n):
- If size == capacity, resize().
- arr[size] = n.
- size++.

RESIZE():
- Create new array with 2 * capacity.
- Copy elements from old to new.
- Update capacity.

POPBACK():
- Return arr[size-1].
- size--.

Time Complexity: O(1) amortized for pushback. O(1) for get/set. O(n) for resize.
Space Complexity: O(n).
    `
  },

  // --- NEW ADDITIONS ---

  // DP
  {
    id: '30',
    title: 'Min Cost Climbing Stairs',
    difficulty: 'Easy',
    category: 'Dynamic Programming',
    tags: ['Array', 'Dynamic Programming'],
    description: `You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index 0, or the step with index 1.

Return the minimum cost to reach the top of the floor.`,
    examples: [
      { input: 'cost = [10,15,20]', output: '15', explanation: 'Start at index 1 (cost 15) and climb 2 steps to reach top.' },
      { input: 'cost = [1,100,1,1,1,100,1,1,100,1]', output: '6' }
    ],
    constraints: ['2 <= cost.length <= 1000', '0 <= cost[i] <= 999'],
    officialSolution: `
Approach: Dynamic Programming (In-Place)

We can solve this by modifying the input array to store the minimum cost to reach each step.
To reach step i, we must have come from either i-1 or i-2.
cost[i] = cost[i] + min(cost[i-1], cost[i-2])
Or iterating backwards:
cost[i] = cost[i] + min(cost[i+1], cost[i+2])

Time Complexity: O(n)
Space Complexity: O(1)

\`\`\`python
def minCostClimbingStairs(cost):
    # Modify in place from end to start
    for i in range(len(cost) - 3, -1, -1):
        cost[i] += min(cost[i + 1], cost[i + 2])
        
    return min(cost[0], cost[1])
\`\`\`
    `
  },
  {
    id: '31',
    title: 'Unique Paths',
    difficulty: 'Medium',
    category: 'Dynamic Programming',
    tags: ['Math', 'Dynamic Programming', 'Combinatorics'],
    companies: ['Google', 'Amazon', 'Meta'],
    description: `There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.`,
    examples: [
      { input: 'm = 3, n = 7', output: '28' },
      { input: 'm = 3, n = 2', output: '3' }
    ],
    constraints: ['1 <= m, n <= 100'],
    officialSolution: `
Approach: Dynamic Programming

Let dp[i][j] be the number of unique paths to reach cell (i, j).
Since we can only move down or right, to reach (i, j), we must come from (i-1, j) or (i, j-1).
dp[i][j] = dp[i-1][j] + dp[i][j-1].
Base case: First row and first column are all 1s.

Time Complexity: O(m * n)
Space Complexity: O(n) (Space optimized to one row)

\`\`\`python
def uniquePaths(m, n):
    row = [1] * n
    
    for i in range(m - 1):
        newRow = [1] * n
        for j in range(n - 2, -1, -1):
            newRow[j] = newRow[j + 1] + row[j]
        row = newRow
        
    return row[0]
\`\`\`
    `
  },
  {
    id: '32',
    title: 'Minimum Falling Path Sum',
    difficulty: 'Medium',
    category: 'Dynamic Programming',
    tags: ['Array', 'Dynamic Programming', 'Matrix'],
    description: `Given an n x n array of integers matrix, return the minimum sum of any falling path through matrix.

A falling path starts at any element in the first row and chooses the element in the next row that is either directly below or diagonally left/right. Specifically, the next element from position (row, col) will be (row + 1, col - 1), (row + 1, col), or (row + 1, col + 1).`,
    examples: [
      { input: 'matrix = [[2,1,3],[6,5,4],[7,8,9]]', output: '13', explanation: 'Path 1->5->7' }
    ],
    constraints: ['n == matrix.length', '1 <= n <= 100'],
    officialSolution: `
Approach: Dynamic Programming

Iterate from the second to last row up to the top.
For each cell (r, c), add the minimum of the 3 possible next steps from the row below it.
matrix[r][c] += min(matrix[r+1][c-1], matrix[r+1][c], matrix[r+1][c+1]) (handling bounds).

Time Complexity: O(n^2)
Space Complexity: O(1) (Modifying input matrix)

\`\`\`python
def minFallingPathSum(matrix):
    N = len(matrix)
    
    for r in range(1, N):
        for c in range(N):
            mid = matrix[r-1][c]
            left = matrix[r-1][c-1] if c > 0 else float('inf')
            right = matrix[r-1][c+1] if c < N - 1 else float('inf')
            
            matrix[r][c] += min(mid, left, right)
            
    return min(matrix[-1])
\`\`\`
    `
  },

  // Backtracking
  {
    id: '33',
    title: 'Permutations',
    difficulty: 'Medium',
    category: 'Backtracking',
    tags: ['Array', 'Backtracking'],
    description: `Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.`,
    examples: [
      { input: 'nums = [1,2,3]', output: '[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]' },
      { input: 'nums = [0,1]', output: '[[0,1],[1,0]]' }
    ],
    constraints: ['1 <= nums.length <= 6', 'All integers of nums are unique.'],
    officialSolution: `
Approach: Backtracking

Recursively build the permutation.
Base case: if permutation length equals nums length, add to result.
Recursive step: Iterate through nums. If num not in current permutation, add it and recurse. Remove it after return (backtrack).

Time Complexity: O(n * n!)
Space Complexity: O(n)

\`\`\`python
def permute(nums):
    res = []
    
    # prev is the current permutation being built
    def backtrack(prev):
        if len(prev) == len(nums):
            res.append(prev[:])
            return
        
        for n in nums:
            if n not in prev:
                prev.append(n)
                backtrack(prev)
                prev.pop()
                
    backtrack([])
    return res
\`\`\`
    `
  },
  {
    id: '34',
    title: 'Subsets',
    difficulty: 'Medium',
    category: 'Backtracking',
    tags: ['Array', 'Backtracking', 'Bit Manipulation'],
    companies: ['Facebook', 'Amazon', 'Google'],
    description: `Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.`,
    examples: [
      { input: 'nums = [1,2,3]', output: '[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]' },
      { input: 'nums = [0]', output: '[[],[0]]' }
    ],
    constraints: ['1 <= nums.length <= 10', 'All numbers are unique.'],
    officialSolution: `
Approach: Backtracking (DFS)

For each element, we have two choices: include it in the current subset or not.
1. Include nums[i], move to i+1.
2. Don't include nums[i], move to i+1.

Time Complexity: O(n * 2^n)
Space Complexity: O(n)

\`\`\`python
def subsets(nums):
    res = []
    subset = []
    
    def dfs(i):
        if i >= len(nums):
            res.append(subset.copy())
            return
        
        # Decision to include nums[i]
        subset.append(nums[i])
        dfs(i + 1)
        
        # Decision NOT to include nums[i]
        subset.pop()
        dfs(i + 1)
        
    dfs(0)
    return res
\`\`\`
    `
  },
  {
    id: '35',
    title: 'Combination Sum',
    difficulty: 'Medium',
    category: 'Backtracking',
    tags: ['Array', 'Backtracking'],
    description: `Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.`,
    examples: [
      { input: 'candidates = [2,3,6,7], target = 7', output: '[[2,2,3],[7]]' }
    ],
    constraints: ['1 <= candidates.length <= 30', '1 <= target <= 500'],
    officialSolution: `
Approach: Backtracking

For each candidate, we can either:
1. Include it (and stay at same index to allow reusing).
2. Skip it (move to next index).

Base cases: total == target (found), total > target or index out of bounds (stop).

Time Complexity: O(2^t) where t is target
Space Complexity: O(t)

\`\`\`python
def combinationSum(candidates, target):
    res = []
    
    def dfs(i, cur, total):
        if total == target:
            res.append(cur.copy())
            return
        if i >= len(candidates) or total > target:
            return
            
        # Include candidates[i]
        cur.append(candidates[i])
        dfs(i, cur, total + candidates[i])
        
        # Skip candidates[i]
        cur.pop()
        dfs(i + 1, cur, total)
        
    dfs(0, [], 0)
    return res
\`\`\`
    `
  },
  {
    id: '36',
    title: 'Word Search',
    difficulty: 'Medium',
    category: 'Backtracking',
    tags: ['Array', 'Backtracking', 'Matrix'],
    companies: ['Google', 'Microsoft', 'Amazon'],
    description: `Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.`,
    examples: [
      { input: 'board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"', output: 'true' }
    ],
    constraints: ['m == board.length', '1 <= m, n <= 6'],
    officialSolution: `
Approach: DFS (Backtracking) on Grid

Iterate through every cell. If cell matches first char of word, start DFS.
DFS(r, c, i):
- If i == len(word), return True.
- Check bounds and char match.
- Mark visited.
- Recurse 4 directions.
- Unmark visited (backtrack).

Time Complexity: O(m * n * 4^L) where L is length of word.
Space Complexity: O(L)

\`\`\`python
def exist(board, word):
    ROWS, COLS = len(board), len(board[0])
    path = set()
    
    def dfs(r, c, i):
        if i == len(word):
            return True
        if (r < 0 or c < 0 or 
            r >= ROWS or c >= COLS or 
            word[i] != board[r][c] or 
            (r, c) in path):
            return False
        
        path.add((r, c))
        res = (dfs(r + 1, c, i + 1) or
               dfs(r - 1, c, i + 1) or
               dfs(r, c + 1, i + 1) or
               dfs(r, c - 1, i + 1))
        path.remove((r, c))
        return res
        
    for r in range(ROWS):
        for c in range(COLS):
            if dfs(r, c, 0): return True
    return False
\`\`\`
    `
  },

  // Stack & Queue
  {
    id: '37',
    title: 'Min Stack',
    difficulty: 'Medium',
    category: 'Stack',
    tags: ['Stack', 'Design'],
    description: `Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:
- MinStack() initializes the stack object.
- void push(int val) pushes the element val onto the stack.
- void pop() removes the element on the top of the stack.
- int top() gets the top element of the stack.
- int getMin() retrieves the minimum element in the stack.

You must implement a solution with O(1) time complexity for each function.`,
    examples: [
      { input: 'push(-2), push(0), push(-3), getMin(), pop(), top(), getMin()', output: '-3, 0, -2' }
    ],
    constraints: ['-2^31 <= val <= 2^31 - 1'],
    officialSolution: `
Approach: Two Stacks

Maintain two stacks:
1. 'stack' to store actual values.
2. 'minStack' to store the minimum value encountered so far.

PUSH: append to stack. append min(val, minStack.top) to minStack.
POP: pop from both.
GETMIN: return minStack.top.

Time Complexity: O(1)
Space Complexity: O(n)

\`\`\`python
class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]
\`\`\`
    `
  },
  {
    id: '38',
    title: 'Daily Temperatures',
    difficulty: 'Medium',
    category: 'Stack',
    tags: ['Array', 'Stack', 'Monotonic Stack'],
    description: `Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0.`,
    examples: [
      { input: 'temperatures = [73,74,75,71,69,72,76,73]', output: '[1,1,4,2,1,1,0,0]' }
    ],
    constraints: ['1 <= temperatures.length <= 10^5'],
    officialSolution: `
Approach: Monotonic Decreasing Stack

Store indices in the stack.
Iterate through the temperatures.
While current temp > stack top temp:
  - We found a warmer day for the day at stack.pop().
  - Calculate difference in indices and store in result.
Push current index.

Time Complexity: O(n)
Space Complexity: O(n)

\`\`\`python
def dailyTemperatures(temperatures):
    res = [0] * len(temperatures)
    stack = [] # pair: [temp, index]
    
    for i, t in enumerate(temperatures):
        while stack and t > stack[-1][0]:
            stackT, stackInd = stack.pop()
            res[stackInd] = (i - stackInd)
        stack.append([t, i])
    return res
\`\`\`
    `
  },

  // Trees
  {
    id: '39',
    title: 'Binary Tree Level Order Traversal',
    difficulty: 'Medium',
    category: 'Trees',
    tags: ['Tree', 'BFS', 'Binary Tree'],
    description: `Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).`,
    examples: [
      { input: 'root = [3,9,20,null,null,15,7]', output: '[[3],[9,20],[15,7]]' }
    ],
    constraints: ['Nodes [0, 2000]'],
    officialSolution: `
Approach: BFS (Queue)

Use a queue.
While queue is not empty:
1. Get size of queue (nodes in current level).
2. Iterate 'size' times:
   - Pop node, add to level list.
   - Push children.
3. Add level list to result.

Time Complexity: O(n)
Space Complexity: O(n)

\`\`\`python
def levelOrder(root):
    res = []
    q = collections.deque()
    if root:
        q.append(root)
        
    while q:
        val = []
        for i in range(len(q)):
            node = q.popleft()
            val.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(val)
    return res
\`\`\`
    `
  },
  {
    id: '40',
    title: 'Lowest Common Ancestor of a BST',
    difficulty: 'Medium',
    category: 'Trees',
    tags: ['Tree', 'DFS', 'BST', 'Binary Tree'],
    description: `Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

The LCA is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).`,
    examples: [
      { input: 'root = [6,2,8,0,4,7,9], p = 2, q = 8', output: '6' }
    ],
    constraints: ['Number of nodes [2, 10^5]', 'All Node.val are unique', 'p != q'],
    officialSolution: `
Approach: Iterative using BST Property

Start at root.
1. If both p and q are greater than curr, go right.
2. If both p and q are smaller than curr, go left.
3. Otherwise, split occurs (or one is ancestor of other), so curr is LCA.

Time Complexity: O(h)
Space Complexity: O(1)

\`\`\`python
def lowestCommonAncestor(root, p, q):
    cur = root
    
    while cur:
        if p.val > cur.val and q.val > cur.val:
            cur = cur.right
        elif p.val < cur.val and q.val < cur.val:
            cur = cur.left
        else:
            return cur
\`\`\`
    `
  },
  {
    id: '41',
    title: 'Validate Binary Search Tree',
    difficulty: 'Medium',
    category: 'Trees',
    tags: ['Tree', 'DFS', 'BST', 'Binary Tree'],
    companies: ['Amazon', 'Microsoft', 'Facebook', 'Bloomberg'],
    description: `Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:
- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys greater than the node's key.
- Both the left and right subtrees must also be binary search trees.`,
    examples: [
      { input: 'root = [2,1,3]', output: 'true' },
      { input: 'root = [5,1,4,null,null,3,6]', output: 'false' }
    ],
    constraints: ['Nodes [1, 10^4]'],
    officialSolution: `
Approach: DFS with Range

Each node must be within a valid range (left, right).
Root range: (-inf, +inf).
When moving left, range becomes (parent_left, parent_val).
When moving right, range becomes (parent_val, parent_right).

Time Complexity: O(n)
Space Complexity: O(n)

\`\`\`python
def isValidBST(root):
    def valid(node, left, right):
        if not node:
            return True
        if not (left < node.val < right):
            return False
        
        return (valid(node.left, left, node.val) and
                valid(node.right, node.val, right))
        
    return valid(root, float("-inf"), float("inf"))
\`\`\`
    `
  },

  // Graphs
  {
    id: '42',
    title: 'Course Schedule',
    difficulty: 'Medium',
    category: 'Graphs',
    tags: ['DFS', 'BFS', 'Graph', 'Topological Sort'],
    companies: ['Amazon', 'Google', 'Microsoft', 'Meta'],
    description: `There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

Return true if you can finish all courses. Otherwise, return false.`,
    examples: [
      { input: 'numCourses = 2, prerequisites = [[1,0]]', output: 'true' },
      { input: 'numCourses = 2, prerequisites = [[1,0],[0,1]]', output: 'false' }
    ],
    constraints: ['1 <= numCourses <= 2000', '0 <= len(prerequisites) <= 5000'],
    officialSolution: `
Approach: DFS (Cycle Detection)

Map each course to its prerequisite list (Adjacency list).
Run DFS on each course.
Keep track of 'visiting' set to detect cycles in current path.
If a node is already in 'visited' (fully processed), return True.

Time Complexity: O(V + E)
Space Complexity: O(V + E)

\`\`\`python
def canFinish(numCourses, prerequisites):
    preMap = { i:[] for i in range(numCourses) }
    for crs, pre in prerequisites:
        preMap[crs].append(pre)
        
    visitSet = set()
    
    def dfs(crs):
        if crs in visitSet:
            return False
        if preMap[crs] == []:
            return True
            
        visitSet.add(crs)
        for pre in preMap[crs]:
            if not dfs(pre): return False
        visitSet.remove(crs)
        preMap[crs] = []
        return True
        
    for crs in range(numCourses):
        if not dfs(crs): return False
    return True
\`\`\`
    `
  },
  {
    id: '43',
    title: 'Pacific Atlantic Water Flow',
    difficulty: 'Medium',
    category: 'Graphs',
    tags: ['Array', 'DFS', 'BFS', 'Matrix'],
    companies: ['Google', 'Amazon', 'Apple'],
    description: `There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans. Water flows from a cell to adjacent cells with height equal or lower.`,
    examples: [
      { input: 'heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]', output: '[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]' }
    ],
    constraints: ['1 <= m, n <= 200'],
    officialSolution: `
Approach: DFS from Oceans

Instead of checking where water flows FROM each cell, check where water can flow TO (reverse flow: upstream from ocean to higher ground).
1. DFS from all Pacific border cells. Mark reachable cells.
2. DFS from all Atlantic border cells. Mark reachable cells.
3. The intersection of both sets is the answer.

Time Complexity: O(m * n)
Space Complexity: O(m * n)

\`\`\`python
def pacificAtlantic(heights):
    ROWS, COLS = len(heights), len(heights[0])
    pac, atl = set(), set()
    
    def dfs(r, c, visit, prevHeight):
        if ((r, c) in visit or 
            r < 0 or c < 0 or r == ROWS or c == COLS or 
            heights[r][c] < prevHeight):
            return
        visit.add((r, c))
        dfs(r + 1, c, visit, heights[r][c])
        dfs(r - 1, c, visit, heights[r][c])
        dfs(r, c + 1, visit, heights[r][c])
        dfs(r, c - 1, visit, heights[r][c])
        
    for c in range(COLS):
        dfs(0, c, pac, heights[0][c])
        dfs(ROWS - 1, c, atl, heights[ROWS - 1][c])
        
    for r in range(ROWS):
        dfs(r, 0, pac, heights[r][0])
        dfs(r, COLS - 1, atl, heights[r][COLS - 1])
        
    res = []
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in pac and (r, c) in atl:
                res.append([r, c])
    return res
\`\`\`
    `
  },

  // Prefix Sum
  {
    id: '44',
    title: 'Subarray Sum Equals K',
    difficulty: 'Medium',
    category: 'Arrays & Hashing',
    tags: ['Array', 'Hash Table', 'Prefix Sum'],
    description: `Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.

A subarray is a contiguous non-empty sequence of elements within an array.`,
    examples: [
      { input: 'nums = [1,1,1], k = 2', output: '2' },
      { input: 'nums = [1,2,3], k = 3', output: '2' }
    ],
    constraints: ['1 <= nums.length <= 2 * 10^4'],
    officialSolution: `
Approach: Prefix Sum + Hash Map

Maintain a running prefix sum.
If (currentSum - k) exists in our hash map, it means there is a subarray ending here with sum k.
Add the count of (currentSum - k) occurrences to result.
Add currentSum to hash map.

Time Complexity: O(n)
Space Complexity: O(n)

\`\`\`python
def subarraySum(nums, k):
    res = 0
    curSum = 0
    prefixSums = { 0 : 1 }
    
    for n in nums:
        curSum += n
        diff = curSum - k
        
        res += prefixSums.get(diff, 0)
        prefixSums[curSum] = 1 + prefixSums.get(curSum, 0)
        
    return res
\`\`\`
    `
  },

  // Sliding Window
  {
    id: '45',
    title: 'Longest Repeating Character Replacement',
    difficulty: 'Medium',
    category: 'Sliding Window',
    tags: ['Hash Table', 'String', 'Sliding Window'],
    companies: ['Google', 'Amazon'],
    description: `You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.`,
    examples: [
      { input: 's = "ABAB", k = 2', output: '4' },
      { input: 's = "AABABBA", k = 1', output: '4' }
    ],
    constraints: ['1 <= s.length <= 10^5'],
    officialSolution: `
Approach: Sliding Window

Window: [l, r].
Valid window: length - max_freq_char <= k.
If invalid, increment l (shrink window).
Result is max valid window size found.

Optimization: We don't need to decrement max_f when shrinking, because we only care about finding a larger window than max found so far.

Time Complexity: O(n)
Space Complexity: O(26)

\`\`\`python
def characterReplacement(s, k):
    count = {}
    res = 0
    l = 0
    maxf = 0
    
    for r in range(len(s)):
        count[s[r]] = 1 + count.get(s[r], 0)
        maxf = max(maxf, count[s[r]])
        
        while (r - l + 1) - maxf > k:
            count[s[l]] -= 1
            l += 1
            
        res = max(res, r - l + 1)
    return res
\`\`\`
    `
  },
  {
    id: '46',
    title: 'Minimum Window Substring',
    difficulty: 'Hard',
    category: 'Sliding Window',
    tags: ['Hash Table', 'String', 'Sliding Window'],
    companies: ['Facebook', 'Google', 'Amazon'],
    description: `Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.`,
    examples: [
      { input: 's = "ADOBECODEBANC", t = "ABC"', output: '"BANC"' },
      { input: 's = "a", t = "a"', output: '"a"' }
    ],
    constraints: ['m == s.length, n == t.length', '1 <= m, n <= 10^5'],
    officialSolution: `
Approach: Sliding Window + Hash Map

1. Count chars in t.
2. Expand window (r) until we satisfy condition (have all chars from t).
3. Once satisfied, contract window (l) to minimize it while maintaining condition.
4. Store min window.

Time Complexity: O(n)
Space Complexity: O(n)

\`\`\`python
def minWindow(s, t):
    if t == "": return ""
    
    countT, window = {}, {}
    for c in t:
        countT[c] = 1 + countT.get(c, 0)
        
    have, need = 0, len(countT)
    res, resLen = [-1, -1], float("inf")
    l = 0
    
    for r in range(len(s)):
        c = s[r]
        window[c] = 1 + window.get(c, 0)
        
        if c in countT and window[c] == countT[c]:
            have += 1
            
        while have == need:
            # update result
            if (r - l + 1) < resLen:
                res = [l, r]
                resLen = (r - l + 1)
                
            # pop from left
            window[s[l]] -= 1
            if s[l] in countT and window[s[l]] < countT[s[l]]:
                have -= 1
            l += 1
            
    l, r = res
    return s[l:r+1] if resLen != float("inf") else ""
\`\`\`
    `
  }
];