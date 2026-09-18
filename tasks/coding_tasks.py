"""
A small set of coding tasks with objective, automatic ground truth (unit tests).
Each task is a dict with:
  - id: unique task identifier
  - problem: the natural-language problem statement given to Agent A
  - test_code: Python code that defines test(fn) -> bool, used to grade Agent B's output
  - entry_point: the function name Agent B's code must define

Start small (10 tasks) and grow this list once the pipeline works end to end.
Feel free to swap this out for a HumanEval subset later -- keep this file's
schema the same if you do, so the rest of the pipeline doesn't need to change.
"""

TASKS = [
    {
        "id": "task_001",
        "source": "hand_easy",
        "problem": "Write a function that returns the sum of all even numbers in a list of integers.",
        "entry_point": "sum_even",
        "test_code": """
def test(fn):
    assert fn([1, 2, 3, 4, 5, 6]) == 12
    assert fn([]) == 0
    assert fn([1, 3, 5]) == 0
    assert fn([2, 4, 6]) == 12
    return True
""",
    },
    {
        "id": "task_002",
        "source": "hand_easy",
        "problem": "Write a function that checks if a given string is a palindrome, ignoring case and spaces.",
        "entry_point": "is_palindrome",
        "test_code": """
def test(fn):
    assert fn("racecar") == True
    assert fn("A man a plan a canal Panama") == True
    assert fn("hello") == False
    assert fn("") == True
    return True
""",
    },
    {
        "id": "task_003",
        "source": "hand_easy",
        "problem": "Write a function that returns the second-largest unique number in a list of integers. Return None if there isn't one.",
        "entry_point": "second_largest",
        "test_code": """
def test(fn):
    assert fn([1, 2, 3, 4, 5]) == 4
    assert fn([5, 5, 5]) is None
    assert fn([10]) is None
    assert fn([3, 1]) == 1
    return True
""",
    },
    {
        "id": "task_004",
        "source": "hand_easy",
        "problem": "Write a function that takes a list of (name, score) tuples and returns the name with the highest score. Break ties alphabetically.",
        "entry_point": "top_scorer",
        "test_code": """
def test(fn):
    assert fn([("alice", 90), ("bob", 85)]) == "alice"
    assert fn([("bob", 90), ("alice", 90)]) == "alice"
    assert fn([("zoe", 1)]) == "zoe"
    return True
""",
    },
    {
        "id": "task_005",
        "source": "hand_easy",
        "problem": "Write a function that flattens a nested list of arbitrary depth into a single flat list.",
        "entry_point": "flatten",
        "test_code": """
def test(fn):
    assert fn([1, [2, 3, [4, 5]], 6]) == [1, 2, 3, 4, 5, 6]
    assert fn([]) == []
    assert fn([1, [2, [3, [4]]]]) == [1, 2, 3, 4]
    return True
""",
    },
    {
        "id": "task_006",
        "source": "hand_easy",
        "problem": "Write a function that returns True if a given integer is prime, False otherwise.",
        "entry_point": "is_prime",
        "test_code": """
def test(fn):
    assert fn(2) == True
    assert fn(1) == False
    assert fn(17) == True
    assert fn(18) == False
    assert fn(0) == False
    return True
""",
    },
    {
        "id": "task_007",
        "source": "hand_easy",
        "problem": "Write a function that merges two sorted lists of integers into one sorted list.",
        "entry_point": "merge_sorted",
        "test_code": """
def test(fn):
    assert fn([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert fn([], [1, 2]) == [1, 2]
    assert fn([1, 2], []) == [1, 2]
    return True
""",
    },
    {
        "id": "task_008",
        "source": "hand_easy",
        "problem": "Write a function that counts the frequency of each word in a string (case-insensitive) and returns a dict.",
        "entry_point": "word_freq",
        "test_code": """
def test(fn):
    result = fn("the cat sat on the mat")
    assert result["the"] == 2
    assert result["cat"] == 1
    assert result["mat"] == 1
    return True
""",
    },
    {
        "id": "task_009",
        "source": "hand_easy",
        "problem": "Write a function that returns the longest common prefix among a list of strings. Return empty string if none.",
        "entry_point": "longest_common_prefix",
        "test_code": """
def test(fn):
    assert fn(["flower", "flow", "flight"]) == "fl"
    assert fn(["dog", "racecar", "car"]) == ""
    assert fn(["single"]) == "single"
    return True
""",
    },
    {
        "id": "task_010",
        "source": "hand_easy",
        "problem": "Write a function that returns True if a binary tree (given as nested tuples (value, left, right) or None) is height-balanced.",
        "entry_point": "is_balanced",
        "test_code": """
def test(fn):
    assert fn(None) == True
    assert fn((1, None, None)) == True
    assert fn((1, (2, (3, None, None), None), None)) == False
    assert fn((1, (2, None, None), (3, None, None))) == True
    return True
""",
    },
]
"""
Additional tasks (task_011 - task_030) to append to TASKS in tasks/coding_tasks.py.
Same schema as the original 10: id, problem, entry_point, test_code.

To use: copy the ADDITIONAL_TASKS list below and extend your existing TASKS
list with it, e.g. at the bottom of coding_tasks.py:

    from additional_tasks import ADDITIONAL_TASKS
    TASKS.extend(ADDITIONAL_TASKS)

or just paste the dicts directly into the TASKS list.
"""

ADDITIONAL_TASKS = [
    {
        "id": "task_011",
        "source": "hand_easy",
        "problem": "Write a function that returns the number of vowels (a, e, i, o, u, case-insensitive) in a string.",
        "entry_point": "count_vowels",
        "test_code": """
def test(fn):
    assert fn("hello world") == 3
    assert fn("") == 0
    assert fn("AEIOU") == 5
    assert fn("xyz") == 0
    return True
""",
    },
    {
        "id": "task_012",
        "source": "hand_easy",
        "problem": "Write a function that removes all duplicate elements from a list while preserving the original order of first occurrence.",
        "entry_point": "dedupe",
        "test_code": """
def test(fn):
    assert fn([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]
    assert fn([]) == []
    assert fn([1, 1, 1]) == [1]
    return True
""",
    },
    {
        "id": "task_013",
        "source": "hand_easy",
        "problem": "Write a function that returns True if two strings are anagrams of each other (ignoring case and spaces), False otherwise.",
        "entry_point": "is_anagram",
        "test_code": """
def test(fn):
    assert fn("listen", "silent") == True
    assert fn("hello", "world") == False
    assert fn("Dormitory", "Dirty Room") == True
    return True
""",
    },
    {
        "id": "task_014",
        "source": "hand_easy",
        "problem": "Write a function that computes the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1) using recursion or iteration.",
        "entry_point": "fibonacci",
        "test_code": """
def test(fn):
    assert fn(0) == 0
    assert fn(1) == 1
    assert fn(10) == 55
    assert fn(2) == 1
    return True
""",
    },
    {
        "id": "task_015",
        "source": "hand_easy",
        "problem": "Write a function that takes a dictionary and returns a new dictionary with keys and values swapped. Assume all values are unique and hashable.",
        "entry_point": "invert_dict",
        "test_code": """
def test(fn):
    assert fn({"a": 1, "b": 2}) == {1: "a", 2: "b"}
    assert fn({}) == {}
    return True
""",
    },
    {
        "id": "task_016",
        "source": "hand_easy",
        "problem": "Write a function that returns the maximum depth (number of nesting levels) of a nested list.",
        "entry_point": "max_depth",
        "test_code": """
def test(fn):
    assert fn([1, 2, 3]) == 1
    assert fn([1, [2, 3]]) == 2
    assert fn([1, [2, [3, [4]]]]) == 4
    assert fn([]) == 1
    return True
""",
    },
    {
        "id": "task_017",
        "source": "hand_easy",
        "problem": "Write a function that checks if a given integer is a perfect square.",
        "entry_point": "is_perfect_square",
        "test_code": """
def test(fn):
    assert fn(16) == True
    assert fn(15) == False
    assert fn(0) == True
    assert fn(1) == True
    return True
""",
    },
    {
        "id": "task_018",
        "source": "hand_easy",
        "problem": "Write a function that groups a list of strings by their length, returning a dict mapping length -> list of strings of that length.",
        "entry_point": "group_by_length",
        "test_code": """
def test(fn):
    result = fn(["a", "bb", "cc", "ddd"])
    assert result[1] == ["a"]
    assert result[2] == ["bb", "cc"]
    assert result[3] == ["ddd"]
    return True
""",
    },
    {
        "id": "task_019",
        "source": "hand_easy",
        "problem": "Write a function that returns the running/cumulative sum of a list of numbers as a new list.",
        "entry_point": "running_sum",
        "test_code": """
def test(fn):
    assert fn([1, 2, 3, 4]) == [1, 3, 6, 10]
    assert fn([]) == []
    assert fn([5]) == [5]
    return True
""",
    },
    {
        "id": "task_020",
        "source": "hand_easy",
        "problem": "Write a function that rotates a list to the right by k positions.",
        "entry_point": "rotate_list",
        "test_code": """
def test(fn):
    assert fn([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]
    assert fn([1, 2, 3], 0) == [1, 2, 3]
    assert fn([1, 2, 3], 3) == [1, 2, 3]
    return True
""",
    },
    {
        "id": "task_021",
        "source": "hand_easy",
        "problem": "Write a function that returns True if a list of integers is sorted in non-decreasing order, False otherwise.",
        "entry_point": "is_sorted",
        "test_code": """
def test(fn):
    assert fn([1, 2, 2, 3]) == True
    assert fn([3, 1, 2]) == False
    assert fn([]) == True
    assert fn([5]) == True
    return True
""",
    },
    {
        "id": "task_022",
        "source": "hand_easy",
        "problem": "Write a function that converts a Roman numeral string (using I, V, X, L, C, D, M) to an integer.",
        "entry_point": "roman_to_int",
        "test_code": """
def test(fn):
    assert fn("III") == 3
    assert fn("IV") == 4
    assert fn("IX") == 9
    assert fn("LVIII") == 58
    assert fn("MCMXCIV") == 1994
    return True
""",
    },
    {
        "id": "task_023",
        "source": "hand_easy",
        "problem": "Write a function that finds the intersection (common elements) of two lists, returning a list with no duplicates.",
        "entry_point": "intersect",
        "test_code": """
def test(fn):
    assert sorted(fn([1, 2, 2, 3], [2, 3, 4])) == [2, 3]
    assert fn([1, 2], [3, 4]) == []
    return True
""",
    },
    {
        "id": "task_024",
        "source": "hand_easy",
        "problem": "Write a function that returns the most frequent element in a list. If there's a tie, return the one that appears first in the list.",
        "entry_point": "most_frequent",
        "test_code": """
def test(fn):
    assert fn([1, 2, 2, 3]) == 2
    assert fn([1, 1, 2, 2]) == 1
    assert fn([5]) == 5
    return True
""",
    },
    {
        "id": "task_025",
        "source": "hand_easy",
        "problem": "Write a function that validates whether a string of brackets (composed of '()[]{}') is balanced/well-formed.",
        "entry_point": "is_balanced_brackets",
        "test_code": """
def test(fn):
    assert fn("()[]{}") == True
    assert fn("(]") == False
    assert fn("([)]") == False
    assert fn("{[]}") == True
    assert fn("") == True
    return True
""",
    },
    {
        "id": "task_026",
        "source": "hand_easy",
        "problem": "Write a function that computes the greatest common divisor (GCD) of two positive integers.",
        "entry_point": "gcd",
        "test_code": """
def test(fn):
    assert fn(12, 8) == 4
    assert fn(17, 5) == 1
    assert fn(100, 25) == 25
    return True
""",
    },
    {
        "id": "task_027",
        "source": "hand_easy",
        "problem": "Write a function that takes a list of integers and returns a list of all pairs (as tuples) that sum to a given target value. Avoid duplicate pairs.",
        "entry_point": "pairs_with_sum",
        "test_code": """
def test(fn):
    result = fn([1, 2, 3, 4], 5)
    assert set(result) == {(1, 4), (2, 3)}
    assert fn([1, 2], 10) == []
    return True
""",
    },
    {
        "id": "task_028",
        "source": "hand_easy",
        "problem": "Write a function that capitalizes the first letter of each word in a sentence while leaving the rest of each word unchanged.",
        "entry_point": "title_case_preserve",
        "test_code": """
def test(fn):
    assert fn("hello world") == "Hello World"
    assert fn("mcDONALD is here") == "McDONALD Is Here"
    assert fn("") == ""
    return True
""",
    },
    {
        "id": "task_029",
        "source": "hand_easy",
        "problem": "Write a function that returns the transpose of a 2D matrix (list of lists).",
        "entry_point": "transpose",
        "test_code": """
def test(fn):
    assert fn([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert fn([[1]]) == [[1]]
    return True
""",
    },
    {
        "id": "task_030",
        "source": "hand_easy",
        "problem": "Write a function that returns True if a given year is a leap year, following standard Gregorian calendar rules.",
        "entry_point": "is_leap_year",
        "test_code": """
def test(fn):
    assert fn(2000) == True
    assert fn(1900) == False
    assert fn(2024) == True
    assert fn(2023) == False
    return True
""",
    },
]
"""
INFORMATION-DENSE tasks (task_031 - task_045) -- designed so that Agent B
CANNOT succeed without the specific details Agent A's handoff carries.
Unlike the earlier batch (which tested general robustness), these are built
so that corruption should meaningfully break things more often: multi-part
constraints, specific required behaviors on edge cases, and problems where
a plausible-but-wrong guess is easy to make without the exact spec.

Why this matters for your dataset: if corruption break-rate is much higher
here than on the "easy" batch, that's a citable finding on its own --
task information-density affects how much handoff quality matters.

Append to TASKS the same way as additional_tasks.py.
"""

HARD_TASKS = [
    {
        "id": "task_031",
        "source": "hand_hard",
        "problem": (
            "Write a function `validate_password(pw)` that returns True only if ALL of these hold: "
            "length is at least 8 AND at most 20, contains at least one uppercase letter, at least one "
            "lowercase letter, at least one digit, at least one symbol from '!@#$%^&*', and contains NO "
            "whitespace anywhere. Otherwise return False."
        ),
        "entry_point": "validate_password",
        "test_code": """
def test(fn):
    assert fn("Abcdef1!") == True
    assert fn("abcdef1!") == False   # no uppercase
    assert fn("ABCDEF1!") == False   # no lowercase
    assert fn("Abcdefg!") == False   # no digit
    assert fn("Abcdefg1") == False   # no symbol
    assert fn("Ab1!") == False       # too short
    assert fn("Ab1!" + "x"*20) == False  # too long
    assert fn("Abc def1!") == False  # whitespace
    return True
""",
    },
    {
        "id": "task_032",
        "source": "hand_hard",
        "problem": (
            "Write a function `parse_log_line(line)` that parses a log line in the EXACT format "
            "'[LEVEL] timestamp - message' where LEVEL is one of INFO/WARN/ERROR, timestamp is a string "
            "like '2024-01-15', and message is the rest. Return a dict with keys 'level', 'timestamp', "
            "'message'. If the line doesn't match this format (wrong brackets, missing ' - ' separator, "
            "or LEVEL not one of the three), return None."
        ),
        "entry_point": "parse_log_line",
        "test_code": """
def test(fn):
    r = fn("[INFO] 2024-01-15 - server started")
    assert r == {"level": "INFO", "timestamp": "2024-01-15", "message": "server started"}
    r2 = fn("[ERROR] 2024-02-01 - connection failed")
    assert r2["level"] == "ERROR" and r2["message"] == "connection failed"
    assert fn("INFO 2024-01-15 - server started") is None   # missing brackets
    assert fn("[DEBUG] 2024-01-15 - test") is None           # invalid level
    assert fn("[INFO] 2024-01-15 server started") is None    # missing separator
    return True
""",
    },
    {
        "id": "task_033",
        "source": "hand_hard",
        "problem": (
            "Write a function `merge_intervals(intervals)` that takes a list of [start, end] intervals "
            "(inclusive on both ends) and merges all overlapping or ADJACENT intervals (e.g. [1,3] and "
            "[3,5] are adjacent and must merge into [1,5], and so must [1,3] and [4,6] since 3 and 4 are "
            "consecutive integers). Return the merged list sorted by start."
        ),
        "entry_point": "merge_intervals",
        "test_code": """
def test(fn):
    assert fn([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]
    assert fn([[1, 3], [3, 5]]) == [[1, 5]]
    assert fn([[1, 3], [4, 6]]) == [[1, 6]]   # adjacent, not overlapping -- must still merge
    assert fn([[1, 2], [5, 6]]) == [[1, 2], [5, 6]]  # gap of 2, stays separate
    return True
""",
    },
    {
        "id": "task_034",
        "source": "hand_hard",
        "problem": (
            "Write a function `custom_round(x, mode)` that rounds a float x to the nearest integer, "
            "where `mode` controls tie-breaking when x is exactly halfway (e.g. 2.5): "
            "mode='up' always rounds .5 up (2.5 -> 3, -2.5 -> -2), "
            "mode='down' always rounds .5 down (2.5 -> 2, -2.5 -> -3), "
            "mode='even' rounds .5 to the nearest even integer (2.5 -> 2, 3.5 -> 4). "
            "Non-halfway values round normally regardless of mode."
        ),
        "entry_point": "custom_round",
        "test_code": """
def test(fn):
    assert fn(2.5, "up") == 3
    assert fn(-2.5, "up") == -2
    assert fn(2.5, "down") == 2
    assert fn(-2.5, "down") == -3
    assert fn(2.5, "even") == 2
    assert fn(3.5, "even") == 4
    assert fn(2.3, "up") == 2
    assert fn(2.7, "down") == 3
    return True
""",
    },
    {
        "id": "task_035",
        "source": "hand_hard",
        "problem": (
            "Write a function `allocate_seats(requests, capacity)` where `requests` is a list of "
            "(name, party_size) tuples in priority order. Seat parties in order ONLY if the remaining "
            "capacity can fit the whole party (no splitting parties). Return a list of names that got "
            "seated, in the order they were seated. Skip (do not seat) any party that doesn't fit, but "
            "keep checking later parties against remaining capacity."
        ),
        "entry_point": "allocate_seats",
        "test_code": """
def test(fn):
    assert fn([("A", 4), ("B", 3), ("C", 2)], 6) == ["A", "C"]  # B skipped (3 > 2 remaining), C fits
    assert fn([("A", 10)], 5) == []
    assert fn([("A", 2), ("B", 2), ("C", 2)], 6) == ["A", "B", "C"]
    return True
""",
    },
    {
        "id": "task_036",
        "source": "hand_hard",
        "problem": (
            "Write a function `caesar_cipher(text, shift, mode)` implementing a Caesar cipher. "
            "mode='encode' shifts each letter forward by `shift` positions in the alphabet (wrapping "
            "z->a), mode='decode' shifts backward. Preserve case. Non-letter characters (spaces, "
            "punctuation, digits) must remain UNCHANGED and not consume a shift position."
        ),
        "entry_point": "caesar_cipher",
        "test_code": """
def test(fn):
    assert fn("abc", 1, "encode") == "bcd"
    assert fn("xyz", 3, "encode") == "abc"
    assert fn("bcd", 1, "decode") == "abc"
    assert fn("Hello, World!", 3, "encode") == "Khoor, Zruog!"
    assert fn("abz", 1, "encode") == "bca"
    return True
""",
    },
    {
        "id": "task_037",
        "source": "hand_hard",
        "problem": (
            "Write a function `dedupe_within_window(items, window)` that builds an output list by "
            "scanning items left to right: for each item, check the last `window` items ALREADY ADDED "
            "to the output (not the original input) -- if the current item's value already appears "
            "among those, skip it; otherwise append it to the output. Items outside that recent window "
            "of output can repeat freely."
        ),
        "entry_point": "dedupe_within_window",
        "test_code": """
def test(fn):
    assert fn([1, 2, 1, 3], 2) == [1, 2, 3]       # second 1 matches within last-2-output window -> dropped
    assert fn([1, 2, 3, 1], 2) == [1, 2, 3, 1]     # last-2-output window at that point is [2,3] -> kept
    assert fn([1, 1, 1], 1) == [1]                 # every repeat matches the single last output item -> all dropped
    return True
""",
    },
    {
        "id": "task_038",
        "source": "hand_hard",
        "problem": (
            "Write a function `budget_split(total, ratios)` that splits an integer `total` amount among "
            "parties according to a list of integer `ratios` (e.g. [2, 3, 5] means 2:3:5). Splits must "
            "be integers and must sum EXACTLY to total -- distribute any remainder (due to integer "
            "division) one unit at a time to the parties with the LARGEST ratio first, in order of ratio "
            "descending, ties broken by original index."
        ),
        "entry_point": "budget_split",
        "test_code": """
def test(fn):
    assert fn(10, [1, 1]) == [5, 5]
    assert fn(10, [1, 2]) == [3, 7]     # 10*1/3=3.33->3, 10*2/3=6.67->6, remainder 1 goes to larger ratio (index 1)
    assert sum(fn(100, [1, 1, 1])) == 100
    assert fn(7, [1, 1, 1]) == [3, 2, 2] # 7//3=2 each, remainder 1 -> first by ratio desc/index tie -> index 0
    return True
""",
    },
    {
        "id": "task_039",
        "source": "hand_hard",
        "problem": (
            "Write a function `find_missing_ranges(nums, lo, hi)` that, given a SORTED list of unique "
            "integers `nums` and bounds `lo`/`hi` (inclusive), returns a list of [start, end] ranges "
            "(inclusive) representing all integers in [lo, hi] that are MISSING from nums. Single missing "
            "numbers should still be represented as [x, x]."
        ),
        "entry_point": "find_missing_ranges",
        "test_code": """
def test(fn):
    assert fn([3, 5], 1, 6) == [[1, 2], [4, 4], [6, 6]]
    assert fn([], 1, 3) == [[1, 3]]
    assert fn([1, 2, 3], 1, 3) == []
    return True
""",
    },
    {
        "id": "task_040",
        "source": "hand_hard",
        "problem": (
            "Write a function `safe_divide_chain(numbers)` that takes a list and divides them left to "
            "right (numbers[0] / numbers[1] / numbers[2] / ...) using FLOAT division. If at any point a "
            "division by zero would occur, return the string 'undefined' instead of raising an error. "
            "If the list has fewer than 2 elements, return the single element (or None for empty list)."
        ),
        "entry_point": "safe_divide_chain",
        "test_code": """
def test(fn):
    assert fn([100, 5, 2]) == 10.0
    assert fn([10, 0, 5]) == "undefined"
    assert fn([5]) == 5
    assert fn([]) is None
    assert fn([10, 2]) == 5.0
    return True
""",
    },
    {
        "id": "task_041",
        "source": "hand_hard",
        "problem": (
            "Write a function `normalize_whitespace(text)` that collapses any run of whitespace "
            "(spaces, tabs, newlines) into a SINGLE space, strips leading/trailing whitespace, but "
            "PRESERVES exactly one blank line (represented as '\\n\\n' in output) wherever the original "
            "text had two or more consecutive newlines, treating that as a paragraph break separate from "
            "normal whitespace collapsing."
        ),
        "entry_point": "normalize_whitespace",
        "test_code": """
def test(fn):
    assert fn("hello   world") == "hello world"
    assert fn("  hello  ") == "hello"
    assert fn("para one\\n\\npara two") == "para one\\n\\npara two"
    assert fn("a\\n\\n\\n\\nb") == "a\\n\\nb"
    assert fn("a\\tb\\tc") == "a b c"
    return True
""",
    },
    {
        "id": "task_042",
        "source": "hand_hard",
        "problem": (
            "Write a function `stack_based_eval(tokens)` that evaluates a Reverse Polish Notation "
            "expression given as a list of string tokens (numbers and operators +, -, *, /). Division "
            "must be INTEGER division truncating toward zero (e.g. -7 // 2 in RPN should give -3, not "
            "Python's default -4). Return the final integer result."
        ),
        "entry_point": "stack_based_eval",
        "test_code": """
def test(fn):
    assert fn(["2", "1", "+", "3", "*"]) == 9
    assert fn(["4", "13", "5", "/", "+"]) == 6
    assert fn(["-7", "2", "/"]) == -3   # truncate toward zero, not floor
    return True
""",
    },
    {
        "id": "task_043",
        "source": "hand_hard",
        "problem": (
            "Write a function `schedule_meetings(meetings)` where `meetings` is a list of (start, end) "
            "tuples in 24hr integer hours. Return True if NONE of the meetings overlap (touching "
            "endpoints like (9,10) and (10,11) do NOT count as overlapping), False if any two overlap."
        ),
        "entry_point": "schedule_meetings",
        "test_code": """
def test(fn):
    assert fn([(9, 10), (10, 11), (11, 12)]) == True
    assert fn([(9, 11), (10, 12)]) == False
    assert fn([(9, 10)]) == True
    assert fn([]) == True
    return True
""",
    },
    {
        "id": "task_044",
        "source": "hand_hard",
        "problem": (
            "Write a function `parse_version(v)` that parses a version string like '1.2.3' into a tuple "
            "of 3 integers (major, minor, patch). Missing parts default to 0 (e.g. '1.2' -> (1,2,0), "
            "'1' -> (1,0,0)). Then write a SECOND function `compare_versions(v1, v2)` that returns -1 if "
            "v1 < v2, 0 if equal, 1 if v1 > v2, comparing major first, then minor, then patch, using "
            "parse_version internally."
        ),
        "entry_point": "compare_versions",
        "test_code": """
def test(fn):
    assert fn("1.2.3", "1.2.4") == -1
    assert fn("1.2", "1.2.0") == 0
    assert fn("2.0", "1.9.9") == 1
    assert fn("1", "1.0.1") == -1
    return True
""",
    },
    {
        "id": "task_045",
        "source": "hand_hard",
        "problem": (
            "Write a function `token_bucket_allow(events, capacity, refill_rate)` simulating a rate "
            "limiter. `events` is a sorted list of integer timestamps (seconds) at which requests "
            "arrive. Start with `capacity` tokens. Each event consumes 1 token if available (allowed), "
            "else it's rejected. Tokens refill at `refill_rate` tokens per second (capped at `capacity`), "
            "calculated based on elapsed time since the last event. Return a list of booleans, one per "
            "event, indicating whether it was allowed."
        ),
        "entry_point": "token_bucket_allow",
        "test_code": """
def test(fn):
    # capacity 2, refill 1/sec: events at t=0,0,0 -> only 2 allowed (starts full)
    assert fn([0, 0, 0], 2, 1) == [True, True, False]
    # capacity 1, refill 1/sec: t=0 allowed, t=1 refilled -> allowed
    assert fn([0, 1], 1, 1) == [True, True]
    # capacity 1: t=0 allowed, t=0 (same time, no refill) -> rejected
    assert fn([0, 0], 1, 1) == [True, False]
    return True
""",
    },
]
HARD_TASKS_2 = [
    {
        "id": "task_046",
        "source": "hand_hard",
        "problem": (
            "Write a function `chunk_list(items, size)` that splits a list into consecutive chunks of "
            "exactly `size` elements each. The LAST chunk, if it has fewer than `size` elements, should "
            "be DROPPED entirely (not included as a partial chunk). If `size` is 0 or negative, return "
            "an empty list."
        ),
        "entry_point": "chunk_list",
        "test_code": """
def test(fn):
    assert fn([1,2,3,4,5,6], 2) == [[1,2],[3,4],[5,6]]
    assert fn([1,2,3,4,5], 2) == [[1,2],[3,4]]   # trailing [5] dropped
    assert fn([1,2], 5) == []                     # nothing full-sized
    assert fn([1,2,3], 0) == []
    return True
""",
    },
    {
        "id": "task_047",
        "source": "hand_hard",
        "problem": (
            "Write a function `resolve_aliases(mapping)` where `mapping` is a dict of string -> string "
            "that may contain CHAINS of aliases (e.g. {'a': 'b', 'b': 'c'} means a resolves to c). Return "
            "a new dict where every key maps directly to its FINAL value (the one that isn't itself a key "
            "in the mapping). If a cycle is detected (e.g. a->b->a), map that key to None instead of "
            "looping forever."
        ),
        "entry_point": "resolve_aliases",
        "test_code": """
def test(fn):
    assert fn({"a": "b", "b": "c"}) == {"a": "c", "b": "c"}
    assert fn({"a": "b", "b": "a"}) == {"a": None, "b": None}
    assert fn({"x": "y"}) == {"x": "y"}
    assert fn({"a": "b", "b": "c", "c": "d"}) == {"a": "d", "b": "d", "c": "d"}
    return True
""",
    },
    {
        "id": "task_048",
        "source": "hand_hard",
        "problem": (
            "Write a function `weighted_median(values, weights)` that computes the weighted median of a "
            "list of numbers with corresponding weights. The weighted median is the smallest value v such "
            "that the sum of weights for all elements <= v is at least half the total weight. Assume "
            "values and weights are parallel lists of the same length, all weights positive."
        ),
        "entry_point": "weighted_median",
        "test_code": """
def test(fn):
    assert fn([1, 2, 3], [1, 1, 1]) == 2          # equal weights -> normal median
    assert fn([1, 2, 3], [1, 1, 5]) == 3           # heavy weight on 3 pulls median there
    assert fn([5], [10]) == 5
    return True
""",
    },
    {
        "id": "task_049",
        "source": "hand_hard",
        "problem": (
            "Write a function `strip_comments(code, marker)` that removes everything from `marker` "
            "(e.g. '#') to the end of each line, EXCEPT when the marker appears inside a single-quoted "
            "string on that line (e.g. the line `x = '#not a comment'` must be left untouched). Assume no "
            "escaped quotes and markers never appear inside double-quoted strings in the test cases."
        ),
        "entry_point": "strip_comments",
        "test_code": """
def test(fn):
    assert fn("x = 1 # set x", "#") == "x = 1 "
    assert fn("y = '#keep this'", "#") == "y = '#keep this'"
    assert fn("z = 2  # comment\\nw = '#kept'", "#") == "z = 2  \\nw = '#kept'"
    return True
""",
    },
    {
        "id": "task_050",
        "source": "hand_hard",
        "problem": (
            "Write a function `next_business_day(day_of_week)` that takes an integer 0-6 (0=Monday, "
            "6=Sunday) representing today, and returns the integer for the NEXT business day (Mon-Fri "
            "only). Friday(4) and Saturday(5) both roll to Monday(0); Sunday(6) rolls to Monday(0) too."
        ),
        "entry_point": "next_business_day",
        "test_code": """
def test(fn):
    assert fn(0) == 1   # Mon -> Tue
    assert fn(3) == 4   # Thu -> Fri
    assert fn(4) == 0   # Fri -> Mon
    assert fn(5) == 0   # Sat -> Mon
    assert fn(6) == 0   # Sun -> Mon
    return True
""",
    },
    {
        "id": "task_051",
        "source": "hand_hard",
        "problem": (
            "Write a function `sparse_matrix_multiply(a, b)` that multiplies two matrices represented as "
            "dicts mapping (row, col) -> nonzero value (missing entries are 0). `a` is m x n, `b` is n x "
            "p. Return the result in the SAME sparse dict format, with ZERO entries in the result "
            "explicitly EXCLUDED (not stored as 0)."
        ),
        "entry_point": "sparse_matrix_multiply",
        "test_code": """
def test(fn):
    a = {(0,0): 1, (0,1): 2}
    b = {(0,0): 3, (1,0): 4}
    result = fn(a, b)
    assert result == {(0,0): 11}   # 1*3 + 2*4 = 11
    assert fn({}, {(0,0): 5}) == {}
    return True
""",
    },
    {
        "id": "task_052",
        "source": "hand_hard",
        "problem": (
            "Write a function `trim_outliers(values, k)` that removes elements more than `k` standard "
            "deviations from the MEAN of the ORIGINAL list (compute mean/stdev once, up front, before "
            "removing anything -- do not recompute after each removal). Use population standard "
            "deviation. Return the filtered list preserving original order."
        ),
        "entry_point": "trim_outliers",
        "test_code": """
def test(fn):
    data = [1, 2, 3, 4, 5, 100]
    result = fn(data, 1.5)
    assert 100 not in result
    assert fn([5, 5, 5, 5], 1) == [5, 5, 5, 5]   # zero stdev, nothing removed
    return True
""",
    },
    {
        "id": "task_053",
        "source": "hand_hard",
        "problem": (
            "Write a function `deep_merge(d1, d2)` that merges two nested dictionaries. If a key exists "
            "in both and BOTH values are dicts, merge them recursively. If a key exists in both and at "
            "least one value is NOT a dict, the value from `d2` OVERWRITES d1's value entirely (no list "
            "concatenation or other merging)."
        ),
        "entry_point": "deep_merge",
        "test_code": """
def test(fn):
    d1 = {"a": 1, "b": {"x": 1, "y": 2}}
    d2 = {"a": 2, "b": {"y": 3, "z": 4}}
    assert fn(d1, d2) == {"a": 2, "b": {"x": 1, "y": 3, "z": 4}}
    assert fn({"a": {"x": 1}}, {"a": 5}) == {"a": 5}
    return True
""",
    },
    {
        "id": "task_054",
        "source": "hand_hard",
        "problem": (
            "Write a function `longest_increasing_run(nums)` that returns the length of the longest "
            "STRICTLY increasing contiguous (not subsequence) run in a list of integers. An empty list "
            "returns 0."
        ),
        "entry_point": "longest_increasing_run",
        "test_code": """
def test(fn):
    assert fn([1, 2, 3, 2, 4, 5, 6, 1]) == 4   # longest contiguous run is [2,4,5,6]
    assert fn([5, 4, 3]) == 1
    assert fn([]) == 0
    assert fn([1, 1, 2]) == 2   # equal values break the streak (strict)
    return True
""",
    },
    {
        "id": "task_055",
        "source": "hand_hard",
        "problem": (
            "Write a function `format_currency(amount, currency)` that formats a float as a currency "
            "string. For currency='USD', format as '$X,XXX.XX' with comma thousands separators and "
            "exactly 2 decimals. For currency='JPY', format as 'Y-X,XXX' with comma separators and NO "
            "decimal places (JPY has no subunits), rounding HALF UP on exact .5 values (e.g. 1234.5 "
            "rounds to 1235, not banker's rounding to even). Negative amounts get a leading '-' before "
            "the symbol."
        ),
        "entry_point": "format_currency",
        "test_code": """
def test(fn):
    assert fn(1234.5, "USD") == "$1,234.50"
    assert fn(1234.5, "JPY") == "Y-1,235"   # rounds, no decimals despite name confusion with '-'
    assert fn(-50, "USD") == "-$50.00"
    assert fn(1000000, "USD") == "$1,000,000.00"
    return True
""",
    },
    {
        "id": "task_056",
        "source": "hand_hard",
        "problem": (
            "Write a function `topo_sort_courses(num_courses, prereqs)` where `prereqs` is a list of "
            "[course, prereq] pairs meaning `prereq` must come before `course`. Return a valid ordering "
            "(list of course indices 0..num_courses-1) satisfying all prerequisites. If it's impossible "
            "(a cycle exists), return an empty list."
        ),
        "entry_point": "topo_sort_courses",
        "test_code": """
def test(fn):
    result = fn(4, [[1,0],[2,0],[3,1],[3,2]])
    assert result.index(0) < result.index(1)
    assert result.index(0) < result.index(2)
    assert result.index(1) < result.index(3)
    assert result.index(2) < result.index(3)
    assert fn(2, [[0,1],[1,0]]) == []   # cycle
    return True
""",
    },
    {
        "id": "task_057",
        "source": "hand_hard",
        "problem": (
            "Write a function `redact_pii(text)` that replaces any substring matching a US phone number "
            "pattern (XXX-XXX-XXXX where X is a digit) with 'XXX-XXX-XXXX', and any substring matching an "
            "email pattern (word characters, @, word characters, dot, word characters) with "
            "'[REDACTED_EMAIL]'. Everything else in the text must remain UNCHANGED, including surrounding "
            "punctuation and spacing."
        ),
        "entry_point": "redact_pii",
        "test_code": """
def test(fn):
    assert fn("Call 555-123-4567 now") == "Call XXX-XXX-XXXX now"
    assert fn("Email me at a@b.com please") == "Email me at [REDACTED_EMAIL] please"
    assert fn("no sensitive data here") == "no sensitive data here"
    assert fn("Phone: 555-123-4567, Email: x@y.com") == "Phone: XXX-XXX-XXXX, Email: [REDACTED_EMAIL]"
    return True
""",
    },
    {
        "id": "task_058",
        "source": "hand_hard",
        "problem": (
            "Write a function `min_coins(amount, coins)` that returns the MINIMUM number of coins needed "
            "to make exactly `amount` using unlimited quantities of the given coin denominations. If it's "
            "impossible to make the exact amount, return -1."
        ),
        "entry_point": "min_coins",
        "test_code": """
def test(fn):
    assert fn(11, [1, 2, 5]) == 3    # 5+5+1
    assert fn(3, [2]) == -1          # impossible
    assert fn(0, [1, 2]) == 0
    assert fn(6, [1, 3, 4]) == 2     # 3+3
    return True
""",
    },
    {
        "id": "task_059",
        "source": "hand_hard",
        "problem": (
            "Write a function `sliding_window_max(nums, k)` that returns a list containing the maximum "
            "value in every contiguous window of size `k` as it slides from left to right across `nums`. "
            "If k > len(nums), return a single-element list with the max of the whole array."
        ),
        "entry_point": "sliding_window_max",
        "test_code": """
def test(fn):
    assert fn([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]
    assert fn([1,2,3], 5) == [3]
    assert fn([4], 1) == [4]
    return True
""",
    },
    {
        "id": "task_060",
        "source": "hand_hard",
        "problem": (
            "Write a function `diff_lists(old, new)` that compares two lists and returns a dict with keys "
            "'added' (items in new but not old), 'removed' (items in old but not new), and 'unchanged' "
            "(items in both), where counts matter for duplicates -- e.g. if old has two 'x' and new has "
            "one 'x', that's 1 removed 'x', not 0. Each value is a list (order doesn't matter within each)."
        ),
        "entry_point": "diff_lists",
        "test_code": """
def test(fn):
    result = fn(["a", "a", "b"], ["a", "c"])
    assert sorted(result["removed"]) == ["a", "b"]
    assert sorted(result["added"]) == ["c"]
    assert sorted(result["unchanged"]) == ["a"]
    return True
""",
    },
]
TASKS=TASKS + ADDITIONAL_TASKS + HARD_TASKS+HARD_TASKS_2