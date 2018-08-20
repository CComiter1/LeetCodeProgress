'''
Problem 1: Two Sum
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
'''
class Problem1:
    def twoSum(self, nums, target):
        numInfo = collections.defaultdict(list)
        for i,num in enumerate(nums): numInfo[num].append(i)
        for num in numInfo:
            if target - num == num and len(numInfo[num]) > 1: 
                return numInfo[num][:2]
            elif target - num in numInfo:
                return [numInfo[num][0], numInfo[target - num][0]]
        return None

'''
Problem 2: Add Two Numbers
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
'''
class Problem2:
    def addTwoNumbers(self, l1, l2):
        num1, num2 = str(), str()
        while l1:
            num1 += str(l1.val)
            l1 = l1.next
        while l2:
            num2 += str(l2.val)
            l2 = l2.next
        theNum = str(int((num1)[::-1])+int((num2)[::-1]))
        toReturn, theNum = ListNode(theNum[-1]), theNum[:-1]
        tracker = toReturn
        while theNum: 
            tracker.next, theNum = ListNode(theNum[-1]), theNum[:-1]
            tracker = tracker.next
        return toReturn

'''
Problem 3: Longest Substring Without Repeating Characters
Given a string, find the length of the longest substring without repeating characters.
'''
class Problem3:
    def lengthOfLongestSubstring(self, s):
        res, lB, cur = int(), int(), dict()
        for i,let in enumerate(s):
            if let in cur: 
                new = cur[let] + 1
                for let in s[lB:new]: cur.pop(let)
                lB = new
            cur[let] = i
            res = max(res, i - lB + 1)
        return res

'''
Problem 4: Median of Two Sorted Arrays
There are two sorted arrays nums1 and nums2 of size m and n respectively.
Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
You may assume nums1 and nums2 cannot be both empty.
'''
class Problem4:
    def findMedianSortedArrays(self, nums1, nums2):
        len1, len2 = len(nums1), len(nums2)
        if len1 > len2:
            nums1, nums2 = nums2, nums1 
            len1, len2 = len2, len1
        if not nums1: 
            if not len2 % 2:
                return (nums2[int(math.ceil(len2/2.0) - 1)] + nums2[int(math.floor(len2/2.0))]) / 2.0 
            return nums2[len2//2] 
        lB, uB = 0, len1
        totLen = len1 + len2
        targ, arrs, lens = totLen // 2, [nums1, nums2], [len1, len2]
        while 1:
            cons = [(lB + uB) / 2]
            cons.append(targ - cons[0])
            lefts = [arr[pt - 1] if pt > 0 else -float('inf') for (arr,pt) in zip(arrs, cons)]
            rights = [arr[pt] if pt < itLen else float('inf') for (arr,pt,itLen) in zip(arrs, cons, lens)]
            if all([rights[1] >= lefts[0], rights[0] >= lefts[1]]):
                return (max(lefts) + min(rights)) / 2.0 if not totLen % 2 else min(rights)
            elif lefts[0] > rights[1]: 
                uB = cons[0]
            elif lefts[1] > rights[0]: 
                lB = cons[0] + 1

'''
Problem 5: Longest Palindromic Substring
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.
'''
class Problem5:
    def longestPalindrome(self, s):
        
        def helper(s1, s2, theLen):
            while s1 >= 0 and s2 < theLen and s[s1] == s[s2]:
                s1 -= 1
                s2 += 1
            if s2 - s1 - 1 > self.uB - self.lB: 
                self.lB, self.uB = s1+1, s2
                
        self.lB, self.uB, theLen = int(), int(), len(s)
        for i in range(theLen): helper(i, i, theLen)
        for i in range(theLen): helper(i, i+1, theLen)
        return s[self.lB: self.uB]

'''
Problem 6: ZigZag Conversion
Write the code that will take a string and make this conversion given a number of rows:
'''
class Problem6:
    def convert(self, s, numRows):
        if numRows < 2: return s
        res, rowDir, curRow = [list() for _ in range(numRows)], 1, 0
        for let in s:
            res[curRow].append(let)
            curRow += rowDir
            if not curRow % (numRows - 1): rowDir *= -1
        return ''.join([''.join(row) for row in res])

'''
Problem 7: Reverse Integer
Given a 32-bit signed integer, reverse digits of an integer.
'''
class Problem7:
    def reverse(self, x):
        if not x: 
            return 0
        isNeg = x < 0 
        x = abs(x)
        x, counter = str(x)[::-1], 0
        while counter < len(x) and x[counter] == '0': 
            counter += 1
        toReturn = int(x[counter:]) * (-1 if isNeg else 1)
        return 0 if not -(2**31) <= toReturn <= (2**31) - 1 else toReturn

'''
Problem 8: String to Integer (atoi)
Implement atoi which converts a string to an integer.
'''
class Problem8:
    def myAtoi(self, theStr):
        toReturn, counter, isNeg = 0, 0, 0
        theNums, theLen = {str(i): i for i in range(10)}, len(theStr)
        while counter < theLen and theStr[counter] == ' ': counter += 1
        if counter >= theLen: return toReturn
        if theStr[counter] in set(['+', '-']):
            if theStr[counter] == '-': isNeg = 1
            counter += 1
        while counter < theLen and theStr[counter] in theNums:
            toReturn *= 10
            toReturn += theNums[theStr[counter]]
            counter += 1
        if isNeg: toReturn *= -1
        return min(max(toReturn, -1*(2**31)), (2**31)-1)

'''
Problem 9: Palindrome Number
Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.
'''
class Problem9:
    def isPalindrome(self, x):
        return str(x) == str(x)[::-1]

'''
Problem 10: Regular Expression Matching
Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
'''
class Problem10:
    def isMatch(self, s, p):
        fS, lenP, lenS = [1], len(p), len(s)
        for j in range(1, lenP + 1): fS.append(0 if p[j - 1] != '*' else fS[j - 2])
        for i in range(1, lenS + 1):
            lS = [0]
            for j in range(1, lenP + 1):
                if p[j-1] != '*':
                    if any([p[j - 1] == '.', p[j - 1] == s[i - 1]]): lS.append(fS[j-1])
                    else: lS.append(0)
                else:
                    if any([p[j - 2] == '.', p[j - 2] == s[i - 1]]): lS.append(fS[j] or lS[-2])
                    else: lS.append(lS[-2])
            fS = lS
        return fS[-1] == 1

'''
Problem 11: Container With Most Water
Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.
'''
class Problem11:
    def maxArea(self, height):
        res, p1, p2 = 0, 0, len(height)-1
        while p1 < p2:
            res = max(res, min(height[p1], height[p2])*(p2-p1))
            if height[p1] < height[p2]: p1 += 1
            else: p2 -= 1
        return res

'''
Problem 12: Integer to Roman
Given an integer, convert it to a roman numeral. Input is guaranteed to be within the range from 1 to 3999.
'''
class Problem12:
    if not num: 
            return str()
        res, ten = list(), -1
        lets  = ['I', 'V', 'X', 'L', 'C', 'D', 'M']
        while num:
            cur = num % 10
            ten += 1
            num = (num - cur) // 10
            if not cur: continue
            if cur == 4:
                res.append(lets[(ten * 2) + 1] + lets[ten * 2])
            elif cur == 9:
                res.append(lets[(ten * 2) + 2] + lets[ten * 2])
            else:
                while cur % 5:
                    cur -= 1
                    res.append(lets[ten * 2])
                if cur:
                    res.append(lets[(ten * 2) + 1])
        return ''.join(res)[::-1]

'''
Problem 13: Roman to Integer
Given a roman numeral, convert it to an integer. Input is guaranteed to be within the range from 1 to 3999.
'''
class Problem13:
    def romanToInt(self, s):
        if not s: 
            return int()
        vals = {let : (10 ** (i // 2)) * (1 if not i % 2 else 5) for (i, let) in enumerate('IVXLCDM')}
        res = vals[s[-1]]
        for (i,let) in enumerate(s[:-1][::-1]): 
            res += (vals[let] * (1 if vals[let] >= vals[s[-(i + 1)]] else -1))
        return res

'''
Problem 14: Longest Common Prefix
Write a function to find the longest common prefix string amongst an array of strings.
'''
class Problem14:
    def longestCommonPrefix(self, strs):
        if not strs: 
            return str()
        res, comp = int(), min(strs, key=len)
        theLen = len(comp)
        while res < theLen and all([s[res]==comp[res] for s in strs]): res += 1
        return comp[:res]

'''
Problem 15: 3Sum
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.
'''
class Problem15:
    def threeSum(self, nums):
        nums.sort()
        theLen, res = len(nums), list()
        for i in range(theLen):
            if i > 0 and nums[i] == nums[i-1]: continue
            lB, uB = i + 1, theLen - 1
            while uB > lB:
                if nums[lB] + nums[uB] > -nums[i]: uB -= 1
                elif nums[lB] + nums[uB] < -nums[i]: lB += 1
                else:
                    res.append([nums[i], nums[lB], nums[uB]])
                    while lB < uB and nums[lB] == nums[lB + 1]: lB += 1
                    while lB < uB and nums[uB] == nums[uB - 1]: uB -= 1
                    lB += 1
                    uB -= 1
        return res

'''
Problem 16: 3Sum Closest
Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.
'''
class Problem16:
    def threeSumClosest(self, nums, target):
        nums.sort()
        res, theLen = float('inf'), len(nums)
        for i,num in enumerate(nums):
            lB, uB = i + 1, theLen - 1
            while uB > lB:
                consid = nums[lB] + nums[uB] + num
                if abs(consid - target) < abs(target - res): res = consid
                if consid > target: uB -= 1
                elif consid < target: lB += 1
                else: return target
        return res

'''
Problem 17: Letter Combinations
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.
'''

class Problem17:
    def letterCombinations(self, digits):
        if not digits: return list()
        res, lets = [str()], ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        for dig in digits:
            new = list()
            for let in lets[int(dig) - 2]:
                for seq in res: new.append(seq + let)
            res = new
        return res

'''
Problem 18: 4Sum
Given an array nums of n integers and an integer target, are there elements a, b, c, and d in nums such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.
'''
class Problem18:
    def fourSum(self, nums, target):
        nums.sort()
        theLen, res = len(nums), list()
        for i in range(theLen - 3):
            if i > 0 and nums[i] == nums[i - 1]: continue
            for j in range(i+1, theLen - 2):
                if j > i + 1 and nums[j] == nums[j - 1]: continue
                lB, uB = j+1, theLen - 1
                while lB < uB:
                    consid = nums[i] + nums[j] + nums[lB] + nums[uB]
                    if consid > target: uB -= 1
                    elif consid < target: lB += 1
                    else:
                        res.append([nums[i], nums[j], nums[lB], nums[uB]])
                        while lB < uB and nums[lB] == nums[lB + 1]: lB += 1
                        while lB < uB and nums[uB] == nums[uB - 1]: uB -= 1
                        lB += 1
                        uB -= 1
        return res

'''
Problem 19: Remove Nth Node From End of List
Given a linked list, remove the n-th node from the end of list and return its head.
'''
class Problem19:
    def removeNthFromEnd(self, head, n):
        lenTrack, theLen = head, int()
        while lenTrack: 
            lenTrack = lenTrack.next
            theLen += 1
        if n == theLen: return head.next
        remTrack = head
        for _ in range(theLen - n - 1): remTrack = remTrack.next
        remTrack.next = remTrack.next.next
        return head

'''
Problem 20: Remove Nth Node From End of List
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
'''
class Problem20:
    def isValid(self, s):
        stack, openToClose = [], {'(':')', '[':']', '{':'}'}
        for i in range(len(s)):
            if s[i] in openToClose: stack.append(s[i])
            elif stack and s[i] == openToClose[stack[-1]]: stack.pop()
            else: return False
        return not stack

'''
Problem 21: Merge Two Sorted Lists
Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.
'''
class Problem21:
    def mergeTwoLists(self, l1, l2):
        toReturn = ListNode(0)
        tracker = toReturn
        while l1 and l2:
            if l1.val < l2.val: tracker.next, l1 = l1, l1.next
            else: tracker.next, l2 = l2, l2.next
            tracker = tracker.next
        if l1: tracker.next = l1
        else: tracker.next = l2
        return toReturn.next

'''
Problem 22: Generate Parentheses
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
'''
class Problem22:
    def generateParenthesis(self, n):
        cur = [('', 0, 0)]
        for _ in range(2 * n):
            new = list()
            for (res, numO, numC) in cur:
                if numO < n: new.append((res + '(', numO + 1, numC))
                if numC < numO: new.append((res + ')', numO, numC + 1))
            cur = new
        return [res[0] for res in cur]

'''
Problem 23: Merge k Sorted Lists
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
'''
class Problem23:
    def mergeKLists(self, lists):
        res, bank = ListNode(-float('inf')), list()
        track = res
        for (i,lst) in enumerate(lists): 
            if lst: heappush(bank, (lst.val,i))
        while bank:
            (consid,i) = heappop(bank)
            track.next = ListNode(consid)
            track = track.next
            if lists[i].next:
                lists[i] = lists[i].next
                heappush(bank, (lists[i].val,i))
        return res.next

'''
Problem 24: Swap Nodes in Pairs
Given a linked list, swap every two adjacent nodes and return its head.
'''
class Problem24:
    def swapPairs(self, head):
        if not head or not head.next: return head
        temp = head.next
        head.next, temp.next = temp.next, head
        head = temp
        head.next.next = self.swapPairs(head.next.next) 
        return head

'''
Problem 25: Reverse Nodes in k-Group
Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.
'''
class Problem25:
    def reverseKGroup(self, head, k):
        if not head: return None
        laster = ListNode(int())
        laster.next, head, front = head, laster, laster
        while 1:
            back = front
            for _ in range(k):
                back = back.next
                if not back: return head.next
            r2, r4, ch, ref = front.next, back.next, front.next, front.next.next
            for _ in range(k - 1):
                hold = ref
                ref = hold.next
                hold.next = ch
                ch = hold
            r2.next, front.next, front = r4, ch, r2

'''
Problem 26: Remove Duplicates from Sorted Array
Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.
'''
class Problem26:
    def removeDuplicates(self, nums):
        if not nums: return 0
        i = 1
        for j in range(1, len(nums)):
            if nums[j] != nums[j-1]: 
                nums[i] = nums[j]
                i += 1
        del nums[i:]
        return i

'''
Problem 27: Remove Element
Given an array nums and a value val, remove all instances of that value in-place and return the new length.
'''
class Problem27:
    def removeElement(self, nums, val):
        c = 0
        while c < len(nums):
            if nums[c] == val: del nums[c]
            else: c += 1
        return len(nums)

'''
Problem 28: Implement strStr()
Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
'''
class Problem28:
    def strStr(self, haystack, needle):
        return haystack.find(needle)

'''
Problem 29: Divide Two Integers
Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.
Return the quotient after dividing dividend by divisor.
'''
class Problem29:
    def divide(self, dividend, divisor):
        if not dividend: return 0
        neg = not any([dividend < 0 and divisor < 0, dividend > 0 and divisor > 0])
        dividend, divisor = abs(dividend), abs(divisor)
        res, orig, amt = int(), divisor, 1
        while dividend >= divisor:
            while divisor << 1 <= dividend: 
                divisor <<= 1
                amt <<= 1
            dividend -= divisor 
            res += amt
            while dividend < divisor and divisor >> 1 >= orig: 
                divisor >>= 1
                amt >>= 1
        if not -(2**31) <= res <= (2**31) - 1: res = (2**31) - (0 if neg else 1)
        return res * (-1 if neg else 1)

'''
Problem 30: Substring with Concatenation of All Words
You are given a string, s, and a list of words, words, that are all of the same length. Find all starting indices of substring(s) in s that is a concatenation of each word in words exactly once and without any intervening characters.
'''
from collections import Counter
class Problem30:
    def findSubstring(self, s, words):
        if not words: return list()
        theLen = len(s)
        if words == [str()]: return [i for i in range(theLen + 1)]
        lenWord = len(words[0])
        lenAll = len(words) * lenWord
        words = Counter(words)
        res = list()
        for st in range(theLen - lenAll + 1):
            if s[st:st + lenWord] not in words: continue
            if words == Counter([s[i : i + lenWord] for i in range(st, st + lenAll, lenWord)]): res.append(st)
        return res

'''
Problem 31: Next Permutation
Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.
If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).
The replacement must be in-place and use only constant extra memory.
'''
from bisect import insort
class Problem31:
    def nextPermutation(self, nums):
        val, theLen = 0, len(nums)
        for i in range(1, theLen)[::-1]:
            if nums[i] > nums[i - 1]:
                j = i
                for k in range(i + 1, theLen):
                    if nums[i - 1] < nums[k] < nums[j]: j = k
                nums[i - 1], nums[j] = nums[j], nums[i - 1]
                for k in range(i, theLen):
                    last = float('inf')
                    temp = nums.pop(k)
                    while temp != last:
                        insort(nums, temp, i)
                        last, temp = temp, nums.pop(k)
                    insort(nums, temp, i)
                val = 1
                break
        if val: return
        for i in range(theLen // 2): nums[i], nums[-(i + 1)] = nums[-(i + 1)], nums[i]

'''
Problem 32: Longest Valid Parentheses
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.
'''
class Problem32:
    def longestValidParentheses(self, s):
        stk = [0]
        for let in s:
            if  let == ')' and stk[-1] % 2: 
                temp = 1 + stk.pop()
                stk[-1] += temp
            else: stk.append(let == '(')
        return max([num - (num % 2) for num in stk])

'''
Problem 33: Search in Rotated Sorted Array
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
You are given a target value to search. If found in the array return its index, otherwise return -1.
You may assume no duplicate exists in the array.
Your algorithm's runtime complexity must be in the order of O(log n).
'''
from bisect import bisect_left
class Problem33:
    def search(self, nums, target):
        if not nums: return -1
        theLen = len(nums)
        lB, uB = 0, theLen - 1
        while nums[lB] > nums[uB]:
            cons = (lB + uB) // 2
            if nums[uB] > nums[cons]: uB = cons
            else: lB = cons + 1
        res = -1
        for (l, h) in zip([0, lB], [lB, theLen]):
            if res > -1: continue
            ind = bisect_left(nums, target, l, h)
            if 0 <= ind < theLen and nums[ind] == target: res = ind
        return res

'''
Problem 34: Find First and Last Position of Element in Sorted Array
Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.
Your algorithm's runtime complexity must be in the order of O(log n).
If the target is not found in the array, return [-1, -1].
'''
from bisect import bisect_left as bleft, bisect as bright
class Problem34:
    def searchRange(self, nums, target):
        theLen, ind1, ind2 = len(nums), bleft(nums, target), bright(nums, target) - 1
        return [ind if 0 <= ind < theLen and nums[ind] == target else -1 for ind in [ind1, ind2]]

'''
Problem 35: Search Insert Position
Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
You may assume no duplicates in the array.
'''
class Problem35:
    def searchInsert(self, nums, target):
        return bisect.bisect_left(nums, target)

'''
Problem 36: Valid Sudoku
Determine if a 9x9 Sudoku board is valid
'''
class Problem36:
    def isValidSudoku(self, board):
        stuffs = [elem for elem in chain(*[[(r, num), (num, c), (num, r//3, c//3)] for (r, row) in enumerate(board) for (c, num) in enumerate(row)]) if '.' not in elem]
        return len(list(stuffs)) == len(set(stuffs))

'''
Problem 37: Sudoku Solver
Write a program to solve a Sudoku puzzle by filling the empty cells.
'''
from itertools import product
class Problem37:
    def solveSudoku(self, board):
        def dfs(i, j):
            if i == 9: return True
            elif board[i][j] != '.':
                if j == 8: return dfs(i + 1, 0)
                else: return dfs(i, j + 1)
            val = {str(i) for i in range(1, 10)}
            for mem in board[i]: val.discard(mem)
            for mem in [board[k][j] for k in range(9)]: val.discard(mem)
            for mem in [board[x][y] for (x,y) in product([z for z in range((i // 3) * 3, ((i // 3) * 3) + 3)], [z for z in range((j // 3) * 3, ((j // 3) * 3) + 3)])]: val.discard(mem)
            for pot in val: 
                board[i][j] = pot
                if j == 8 and dfs(i + 1, 0): return True
                elif j < 8 and dfs(i, j + 1): return True
                board[i][j] = '.'
            return False
        dfs(0, 0)

'''
Problem 38: Count and Say
Given an integer n, generate the nth term of the count-and-say sequence.
'''
class Problem38:
    def countAndSay(self, n):
        if n == 1: return "1"
        prev, last, new = self.countAndSay(n-1), -1, str()
        for c in range(len(prev)):
            if c == len(prev) - 1 or prev[c] != prev[c + 1]: 
                new += str(c - last) + prev[c]
                last = c
        return new

'''
Problem 39: Combination Sum
Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.
The same repeated number may be chosen from candidates unlimited number of times.
'''
class Problem39:
    def combinationSum(self, candidates, target):
        def helper(goal):
            if goal < 1 or goal in res: return
            for num in candidates:
                if num > goal: continue
                helper(goal - num)
                for part in res[goal - num]: res[goal].add(tuple(sorted(list(part) + [num])))
        res = collections.defaultdict(set)
        res[0].add(tuple())
        helper(target)
        return [list(part) for part in res[target]]

'''
Problem 40: Combination Sum
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.
Each number in candidates may only be used once in the combination.
'''
class Problem40:
    def dfs(toGo, ind, path):
            if not toGo: 
                res.append(path)
                return
            for j in range(ind + 1, len(candidates)):
                if (candidates[j] > toGo) or (j > ind + 1 and candidates[j] == candidates[j - 1]): continue
                dfs(toGo - candidates[j], j, path + [candidates[j]])
        candidates.sort()
        res = list()
        dfs(target, -1, list())
        return res

'''
Problem 41: First Missing Positive
Given an unsorted integer array, find the smallest missing positive integer.
'''
class Problem41:
    def firstMissingPositive(self, nums):
        posNums, theLen, counter = 0, len(nums), 1
        for num in nums:
            if 0 < num <= theLen: posNums = posNums | (1 << (num-1))
        while posNums & 1 == 1: 
            posNums >>= 1
            counter += 1
        return counter

'''
Problem 42: Trapping Rain Water
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
'''
class Problem42:
    def trap(self, height):
        theLen = len(height)
        if not height: return 0
        lL = [height[0]]
        for num in height[1:]: lL.append(max(lL[-1], num))
        lR = [height[-1]]
        for num in height[:-1][::-1]: lR.append(max(lR[-1], num))
        return sum([max(0, min(lL[i - 1], lR[-(i + 1)]) - height[i]) for i in range(1, theLen - 1)])

'''
Problem 43: Trapping Rain Water
Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
'''
class Problem43:
    def multiply(self, num1, num2):
        nums = [int() for _ in range(2)]
        theSubt = ord('0')
        for i,num in enumerate([num1, num2]):
            for dig in num:
                nums[i] *= 10
                nums[i] += ord(dig) - theSubt
        resf, res = list(), nums[0] * nums[1]
        while res:
            temp = res % 10
            resf.append(chr(temp + theSubt))
            res = (res - temp) // 10
        return ''.join(resf[::-1]) if resf else '0'

'''
Problem 44: Wildcard Matching
Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*'.
'''
class Problem44:
    def isMatch(self, s, p):
        old = [1]
        for pat in p: old.append(1 if old[-1] and pat == '*' else 0)
        for let in s:
            new = [0]
            for i,pat in enumerate(p):
                if pat == '?' or pat == let: new.append(old[i])
                elif pat == '*': new.append(new[-1] or old[i + 1])
                else: new.append(0)
            old = new
        return not not old[-1]

'''
Problem 45: Jump Game II
Given an array of non-negative integers, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Your goal is to reach the last index in the minimum number of jumps.
'''
class Problem45:
    def jump(self, nums):
        canGo, res, maxGo = int(), int(), int()
        for i,num in enumerate(nums[:-1]):
            canGo = max(canGo, i + num)
            if i == maxGo:
                maxGo = canGo
                res += 1
        return res

'''
Problem 46: Permutations
Given a collection of distinct integers, return all possible permutations.
'''
class Problem46:
    def permute(self, nums):
        if not nums: return [[]]
        new = nums.pop()
        res = self.permute(nums)
        fin = list()
        theLen = len(res[0])
        for part in res:
            for ins in range(theLen + 1):
                toAdd = [num for num in part]
                toAdd.insert(ins, new)
                fin.append(toAdd)
        return fin
        #trivially, return list(list(perm) for perm in itertools.permutations(nums))

'''
Problem 47: Permutations II 
Given a collection of numbers that might contain duplicates, return all possible unique permutations.
'''
class Problem47:
    def permuteUnique(self, nums):
        theLen = len(nums)
        if not theLen: return [[]]
        new = nums.pop()
        res = self.permuteUnique(nums)
        fin = set()
        for part in res:
            for ind in range(theLen):
                toAdd = [num for num in part]
                toAdd.insert(ind, new)
                fin.add(tuple(toAdd))
        return [list(part) for part in fin]
        #return [list(perm) for perm in set(itertools.permutations(nums))]

'''
Problem 48: Rotate Image
You are given an n x n 2D matrix representing an image.
Rotate the image by 90 degrees (clockwise).
'''
class Problem48:
    def rotate(self, matrix):
        for i in range(math.ceil(len(matrix)/2)):
            for j in range(i, len(matrix[i])-1-i):
                for (x, y) in [(j, -1*(i+1)), (-1*(1+i), -1*(1+j)), (-1*(1+j), i)]: 
                    matrix[i][j], matrix[x][y] = matrix[x][y], matrix[i][j]

'''
Problem 49: Group Anagrams
Given an array of strings, group anagrams together.
'''
from collections import defaultdict
class Problem49:
    def groupAnagrams(self, strs):
        res = defaultdict(list)
        for strng in strs: res[''.join(sorted(strng))].append(strng)
        return [lst for lst in res.values()]

'''
Problem 50: Pow(x, n)
Implement pow(x, n), which calculates x raised to the power n (xn).
'''
class Problem50:
    def myPow(self, x, n):
        res = float(1)
        neg = n < 0
        n = abs(n)
        while n:
            new = int(math.log(n, 2))
            multer = x
            for _ in range(new): multer *= multer
            res *= multer
            n -= (1<<new)
        return res if not neg else 1 / res

'''
Problem 51: N-Queens
The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.
'''
class Problem51:
    def solveNQueens(self, n):
        def dfs(soFar):
            nxt = len(soFar)
            if nxt == n:
                res.append(soFar)
                return
            poss = {i for i in range(n)}
            for i,num in enumerate(soFar):
                for rem in [num, num + (nxt - i), num - (nxt - i)]: poss.discard(rem)
            for num in poss: dfs(soFar + (num, ))
        res = list()
        dfs(tuple())
        return [[("." * num) + 'Q' + ("." * (n - num - 1)) for num in bp] for bp in res]

'''
Problem 52: N-Queens II
The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.
Given an integer n, return the number of distinct solutions to the n-queens puzzle.
'''
class Problem52:
    def totalNQueens(self, n):
        def dfs(soFar):
            nxt = len(soFar)
            if nxt == n:
                self.res += 1
                return
            poss = {i for i in range(n)}
            for i,num in enumerate(soFar):
                for rem in [num, num + (nxt - i), num - (nxt - i)]: poss.discard(rem)
            for num in poss: dfs(soFar + (num, ))
        self.res = int()
        dfs(tuple())
        return self.res

'''
Problem 53: Maximum Subarray
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
'''
class Problem53:
    def maxSubArray(self, nums):
        res, cum = -float('inf'), 0
        for num in nums:
            res = max(res, cum + num)
            cum = max(0, cum + num)
        return res

'''
Problem 54: Spiral Matrix
Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
'''
class Problem54:
    def spiralOrder(self, matrix):
        if not matrix or not matrix[0]: return list()
        len1, len2, res = len(matrix), len(matrix[0]), list()
        for i in range(math.ceil(min(len1, len2) / 2.0)):
            for j in range(i, len2 - i): res.append(matrix[i][j])
            for r in range(i + 1, len1 - i - 1): res.append(matrix[r][-(i + 1)])
            if len1 - i - 1 != i:
                for j in range(len2 - i - 1, i - 1, -1): res.append(matrix[-(i + 1)][j])
            if i != len2 - i - 1:
                for r in range(len1 - i - 2, i, -1): res.append(matrix[r][i])
        return res

'''
Problem 55: Jump Game
Given an array of non-negative integers, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index.
'''
class Problem55:
    def canJump(self, nums):
        canGo, maxGo = 0, 0
        for i,num in enumerate(nums):
            if i > maxGo: break
            maxGo = max(maxGo, i + num)
            if i == canGo: canGo = maxGo
        return maxGo >= len(nums) - 1

'''
Problem 56: Merge Intervals
Given a collection of intervals, merge all overlapping intervals.
'''
class Problem56:
    def merge(self, intervals):
        intervals.sort(key = lambda intr: intr.start)
        res = list()
        for intr in intervals:
            if not res or intr.start > res[-1].end: res.append(intr)
            else: res[-1].end = max(res[-1].end, intr.end)
        return res

'''
Problem 57: Merge Intervals
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
You may assume that the intervals were initially sorted according to their start times.
'''
class Problem57:
    def insert(self, ints, newInt):
        stInd = lCut([intv.start for intv in ints], newInt.start)
        endInd = lCut([intv.end for intv in ints], newInt.end)
        theLen = len(ints)
        if 0 < stInd <= theLen and ints[stInd - 1].end >= newInt.start: stInd -= 1
        if 0 <= endInd < len(ints) and ints[endInd].start > newInt.end: endInd -= 1
        if 0 <= stInd < theLen: newInt.start = min(newInt.start, ints[stInd].start)
        if 0 <= endInd < theLen: newInt.end = max(newInt.end, ints[endInd].end)
        ints = ints[:stInd] + [newInt] + ints[endInd + 1:]
        return ints

'''
Problem 58: Merge Intervals
Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.
If the last word does not exist, return 0.
'''
class Problem58:
    def lengthOfLastWord(self, s):
        cons = [word for word in s.split(' ') if len(word)]
        return 0 if not cons else len(cons[-1])

'''
Problem 59: Spiral Matrix II
Given a positive integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.
'''
class Problem59:
    res = [[0 for __ in range(n)] for _ in range(n)]
        crow, ccol, curct, chng, numCh = 0, 0, n, n, 0
        for num in range(1, (n ** 2) + 1):
            res[crow][ccol] = num
            curct -= 1
            if not curct: 
                numCh += 1
                curct = chng - 1
            elif numCh == 3 and curct == 1:
                numCh = 0
                chng -= 2
                curct = chng
            if not numCh % 2: ccol += 1 * (-1 if numCh else 1)
            else: crow += 1 * (-1 if numCh - 1 else 1)
        return res

'''
Problem 60: Spiral Matrix II
Given n and k, return the kth permutation sequence.
'''
class Problem60:
    def getPermutation(self, n, k):
        res, toPick = list(), [str(i) for i in range(1, n + 1)]
        curFact = math.factorial(n)
        k -= 1
        for i in range(n): 
            curFact //= (n - i) 
            res.append(toPick.pop(k // curFact))
            k %= curFact
        return ''.join(res)

'''
Problem 61: Rotate List
Given a linked list, rotate the list to the right by k places, where k is non-negative.
'''
class Problem61:
    def rotateRight(self, head, k):
        if not (k and head): return head
        vals = []
        while head:
            vals.append(head.val)
            head = head.next
        k %= len(vals)
        vals = vals[-k:] + vals[:-k]
        res = ListNode(0)
        track = res
        for val in vals:
            track.next = ListNode(val)
            track = track.next
        return res.next

'''
Problem 62: Unique Paths
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
How many possible unique paths are there?
'''
class Problem62:
    def uniquePaths(self, m, n):
        def nCk(n, k):
            return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
        return nCk(m + n - 2, n - 1)

'''
Problem 63: Unique Paths II
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
Now consider if some obstacles are added to the grids. How many unique paths would there be?
'''
class Problem63:
    def uniquePathsWithObstacles(self, grid):
        if not grid or not grid[0] or grid[-1][-1] == 1: return 0
        len1, len2 = len(grid), len(grid[0])
        grid[-1][-1] = 1.0
        for r in range(len1 - 1, -1, -1):
            for c in range(len2 - 1, -1, -1):
                if isinstance(grid[r][c], int) and grid[r][c] == 1:
                    grid[r][c] = 0
                    continue
                if r + 1 < len1: grid[r][c] += grid[r + 1][c]
                if c + 1 < len2: grid[r][c] += grid[r][c + 1]
        return int(grid[0][0])

'''
Problem 64: Minimum Path Sum
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.
'''
class Problem64:
    def minPathSum(self, grid):
        len1, len2 = len(grid), len(grid[0])
        for i in range(len1):
            for j in range(len2):
                adder = float('inf')
                if i - 1 >= 0 : adder = min(adder, grid[i - 1][j])
                if j - 1 >= 0 : adder = min(adder, grid[i][j - 1])
                grid[i][j] += 0 if adder == float('inf') else adder
        return grid[-1][-1]

'''
Problem 65: Valid Number
Validate if a given string is numeric.
'''
class Problem65:
    def isNumber(self, s):
        theLen, c = len(s), 0
        while c < theLen and s[c] == ' ': c += 1
        if c == theLen: return False
        d = theLen - 1
        while d >= 0 and s[d] == ' ': d -= 1
        s = s[c : d + 1]
        if ' ' in s: return False
        ind = s.find('e')
        end, theLen = ind, len(s)
        if ind > -1:
            if ind < theLen - 1 and (s[end + 1] in ['-', '+']): end += 1
            if ind == 0 or end == theLen - 1: return False
            if 'e' in s[:ind] or 'e' in s[end + 1:]: return False
            if '.' in s[end + 1:]: return False
            if '-' in s[1:ind] or '-' in s[end + 1:]: return False
            if set(s[end + 1:]) - {str(i) for i in range(10)}.union({'.', '-'}): return False
            s = s[:ind]
        neg = s.find('-')
        if neg > 0 or s.count('-') > 1: return False
        elif not neg: s = s[1:]
        pos = s.find('+')
        if pos > 0: return False
        elif not pos: s = s[1:]
        if s.count('.') > 1: return False
        dec = s.find('.')
        if dec > -1: s = s[:dec] + s[dec + 1:]
        if set(s) - {str(i) for i in range(10)}.union({'.', '-'}): return False
        return not not s

'''
Problem 66: Plus One
Given a non-empty array of digits representing a non-negative integer, plus one to the integer.
The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.
You may assume the integer does not contain any leading zero, except the number 0 itself.
'''
class Problem66:
    def plusOne(self, digits):
        return [int(let) for let in str(int("".join([str(num) for num in digits]))+1)]

'''
Problem 67: Add Binary
Given two binary strings, return their sum (also a binary string).
The input strings are both non-empty and contains only characters 1 or 0.
'''
class Problem67:
    def addBinary(self, a, b):
        return bin(int(a, 2)+int(b, 2))[2:]

'''
Problem 68: Text Justification
Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.
Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
For the last line of text, it should be left justified and no extra space is inserted between words.
'''
class Problem68:
    def fullJustify(self, words, maxWidth):
        res, cur, curLen, theLen = list(), list(), int(), len(words)
        for word in words:
            if len(word) + curLen + len(cur) <= maxWidth: 
                cur.append(word)
                curLen += len(word)
                continue
            fin, toAdd = maxWidth - curLen, None
            if len(cur) > 1:
                toGo, c = fin % (len(cur) - 1), 0
                for _ in range(toGo):
                    cur[c] += ' '
                    c += 1
                    if c == len(cur) - 1: c = 0
                fin //= (len(cur) - 1)
                toAdd = (' '*fin).join(cur)
            else: toAdd = cur[0] + (' '*(maxWidth - len(cur[0])))
            res.append(toAdd)
            cur, curLen = [word], len(word)
        if cur: res.append(' '.join(cur))
        else: res[-1] = ' '.join([word for word in res[-1].split(' ') if word])
        res[-1] += ' '*(maxWidth - len(res[-1]))
        return res

'''
Problem 69: Sqrt(x)
Implement int sqrt(int x).
Compute and return the square root of x, where x is guaranteed to be a non-negative integer.
Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.
'''
class Problem69:
    def mySqrt(self, x):
        lB, uB = 0, x+1
        while uB - lB > 1:
            con = (lB + uB) // 2
            if con ** 2 <= x: lB = con
            else: uB = con
        return lB

'''
Problem 70: Climbing Stairs
Implement int sqrt(int x).
Compute and return the square root of x, where x is guaranteed to be a non-negative integer.
Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.
'''
class Problem70:
    def climbStairs(self, n):
        a , b = 1 , 1
        for _ in range(2 , n + 1): a , b = b , a + b
        return b

'''
Problem 71: Simplify Path
Given an absolute path for a file (Unix-style), simplify it.
'''
class Problem71:
    def simplifyPath(self, path):
        path, stk = path.split("/"), []
        for part in path:
            if part == '..' and stk: stk.pop()
            elif part not in {'.', '..', '/', ''}: stk.append(part)
        return "/"+"/".join(stk)

'''
Problem 72: Edit Distance
Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.
You have the following 3 operations permitted on a word:
Insert a character
Delete a character
Replace a character
'''
class Problem72:
    def minDistance(self, word1, word2):
        len1, len2 = len(word1), len(word2)
        dP = [[None for __ in range(len2 + 1)] for _ in range(len1 + 1)]
        for i in range(len1 + 1): dP[i][len2] = len1 - i
        for j in range(len2 + 1): dP[len1][j] = len2 - j
        for i in range(len1 - 1, -1, -1):
            for j in range(len2 - 1, -1, -1):
                if word1[i] == word2[j]: dP[i][j] = dP[i + 1][j + 1]
                else: dP[i][j] = 1 + min(dP[i + 1][j + 1], dP[i][j + 1], dP[i + 1][j])
        return dP[0][0]

'''
Problem 73: Set Matrix Zeroes
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.
'''
class Problem73:
    def setZeroes(self, matrix):
        if not matrix or not matrix[0]: return
        len1, len2 = len(matrix), len(matrix[0])
        for i in range(len1):
            for j in range(len2):
                if not (isinstance(matrix[i][j], int) and not matrix[i][j]): continue
                for r in range(len1): matrix[r][j] = int() if ((not matrix[r][j]) and isinstance(matrix[r][j], int)) else float()
                for c in range(len2): matrix[i][c] = int() if ((not matrix[i][c]) and isinstance(matrix[i][c], int)) else float()
        for i in range(len1):
            for j in range(len2):
                matrix[i][j] = int(matrix[i][j])

'''
Problem 74: Search a 2D Matrix
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.
'''
class Problem74:
    def searchMatrix(self, matrix, target):
        if not matrix or not matrix[0]: return False
        len1, len2, res = len(matrix), len(matrix[0]), 0
        cr, cc = 0, len2 - 1
        for _ in range(len1 + len2 - 1):
            if cr >= len1 or cc < 0: break
            if matrix[cr][cc] == target:
                res = 1
                break
            elif matrix[cr][cc] > target: cc -= 1
            else: cr += 1
        return res == 1

'''
Problem 75: Sort Colors
Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white and blue.
Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
Note: You are not suppose to use the library's sort function for this problem
A rather straight forward solution is a two-pass algorithm using counting sort.
First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array with total number of 0's, then 1's and followed by 2's.
Could you come up with a one-pass algorithm using only constant space?
'''
class Problem75:
    def sortColors(self, nums):
        f1, f2 = 0, 0
        for (i, num) in enumerate(nums):
            nums[i] = 2
            if num < 2:
                nums[f2] = 1
                f2 += 1
            if not num:
                nums[f1] = 0
                f1 += 1
