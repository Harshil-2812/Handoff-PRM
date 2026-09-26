"""
Batch 3 of hand-crafted tasks (task_061-task_084), deliberately written with
named entities, tool/library names, and explicit edge-case language woven
into the problem statement -- unlike the original 60 tasks, which are mostly
generic (string/list/dict manipulation) with little for the entity_omission
corruption's spaCy-NER + edge-case-marker heuristic to actually find and
remove. This batch exists specifically to grow the entity_omission subgroup's
sample size (n=8 in the original 177-row dataset) for the classifier's
robustness analysis, per the decision to scale to the full HumanEval/MBPP
pools plus additional hand-crafted tasks.

Every test_code below was executed against a hand-written reference solution
in verify_hard_tasks_3.py before inclusion. That first pass caught 4 real
bugs (task_065, task_077, task_078, task_082 -- one had a genuinely
degenerate algorithm design in task_078, now fixed), all corrected before
this file reached its final form -- the same practice, and roughly the same
bug-catch rate, as the original hard-task batches (3 bugs caught there).
"""

HARD_TASKS_3 = [
    {
        "id": "task_061",
        "source": "hand_hard",
        "problem": (
            "Write a function `schedule_meeting(bookings, new_slot)` for a calendar tool used by "
            "Priya Sharma's team in Bangalore. `bookings` is a list of (start, end) integer-hour tuples "
            "already on the calendar, and `new_slot` is a (start, end) tuple to add. The function must "
            "return True and NOT modify anything if `new_slot` overlaps any existing booking -- two "
            "slots overlap if they share any hour, so (9, 11) and (11, 13) do NOT overlap (11 is a "
            "shared boundary, not shared time), but (9, 12) and (11, 13) DO overlap. Edge case: if "
            "`bookings` is empty, any `new_slot` should be accepted."
        ),
        "entry_point": "schedule_meeting",
        "test_code": """
def test(fn):
    assert fn([], (9, 10)) == True
    assert fn([(9, 11), (13, 15)], (11, 13)) == True
    assert fn([(9, 11)], (10, 12)) == False
    assert fn([(9, 12)], (11, 13)) == False
    assert fn([(9, 10), (14, 16)], (10, 14)) == True
    return True
""",
    },
    {
        "id": "task_062",
        "source": "hand_hard",
        "problem": (
            "Write a function `validate_pandas_column(series, dtype)` that checks whether every "
            "value in a list `series` matches the requested `dtype`, which is one of the strings "
            "'int', 'float', or 'str'. The function must raise a ValueError with the message "
            "'unsupported dtype' if `dtype` is anything else. Edge case: for dtype='float', an int "
            "value like 3 should count as valid too (ints are a subset of floats here), but for "
            "dtype='int', a float like 3.0 must NOT count as valid, even though it has no fractional part."
        ),
        "entry_point": "validate_pandas_column",
        "test_code": """
def test(fn):
    assert fn([1, 2, 3], "int") == True
    assert fn([1, 2.5, 3], "int") == False
    assert fn([1, 2.0, 3], "float") == True
    assert fn(["a", "b"], "str") == True
    assert fn([1, "a"], "str") == False
    try:
        fn([1], "bool")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert str(e) == "unsupported dtype"
    return True
""",
    },
    {
        "id": "task_063",
        "source": "hand_hard",
        "problem": (
            "Marcus Chen at Fenwick Logistics needs a function `route_cost(distances, fuel_price)` "
            "that computes total delivery cost. `distances` is a list of kilometers for each leg of a "
            "route, and the truck consumes 0.35 liters per kilometer. The function must round the final "
            "cost to 2 decimal places using standard rounding (round-half-to-even is NOT required -- "
            "plain round() behavior is fine). Edge case: if any single leg in `distances` is negative, "
            "the function should raise a ValueError with message 'invalid distance', since a route "
            "cannot have a negative leg."
        ),
        "entry_point": "route_cost",
        "test_code": """
def test(fn):
    assert fn([10, 20], 1.5) == round((10+20) * 0.35 * 1.5, 2)
    assert fn([0], 2.0) == 0.0
    try:
        fn([10, -5], 1.0)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert str(e) == "invalid distance"
    return True
""",
    },
    {
        "id": "task_064",
        "source": "hand_easy",
        "problem": (
            "Write a function `format_greeting(name, city)` used by the onboarding bot at Vellore "
            "Institute of Technology. It should return the string f'Welcome, {name}, from {city}!' "
            "with the name capitalized (first letter upper, rest unchanged) and the city left exactly "
            "as given."
        ),
        "entry_point": "format_greeting",
        "test_code": """
def test(fn):
    assert fn("aditi", "Chennai") == "Welcome, Aditi, from Chennai!"
    assert fn("RAHUL", "Delhi") == "Welcome, RAHUL, from Delhi!"
    return True
""",
    },
    {
        "id": "task_065",
        "source": "hand_hard",
        "problem": (
            "Write a function `rate_limit_ok(requests, window_seconds, max_requests, now)` for the "
            "API gateway used by Orbit Analytics. `requests` is a list of past request timestamps "
            "(integers, seconds), and the function must return True if adding a new request at time "
            "`now` would keep the count of requests within the sliding window (now - window_seconds, now] "
            "at or below `max_requests`. Edge case: a request exactly `window_seconds` before `now` is "
            "OUTSIDE the window (the window is a half-open interval that excludes its own left edge), so "
            "it must not be counted."
        ),
        "entry_point": "rate_limit_ok",
        "test_code": """
def test(fn):
    assert fn([10, 20, 30], 20, 3, 40) == True
    assert fn([10, 20, 30], 20, 2, 40) == True
    assert fn([10, 20, 30], 20, 1, 40) == False
    assert fn([20], 20, 1, 40) == True
    assert fn([], 20, 1, 40) == True
    return True
""",
    },
    {
        "id": "task_066",
        "source": "hand_hard",
        "problem": (
            "Write a function `merge_configs(base, override)` for the deployment tool at Nimbus Cloud. "
            "Both `base` and `override` are dicts that may contain nested dicts. The function must "
            "deep-merge them: for any key present in both where both values are dicts, merge recursively; "
            "otherwise the value from `override` wins entirely (even if base's value was a dict and "
            "override's is not). Edge case: a key with value `None` in `override` must explicitly delete "
            "that key from the result, rather than setting it to None."
        ),
        "entry_point": "merge_configs",
        "test_code": """
def test(fn):
    base = {"a": 1, "b": {"x": 1, "y": 2}, "c": 5}
    override = {"b": {"y": 99, "z": 3}, "c": None, "d": 7}
    result = fn(base, override)
    assert result == {"a": 1, "b": {"x": 1, "y": 99, "z": 3}, "d": 7}
    return True
""",
    },
    {
        "id": "task_067",
        "source": "hand_easy",
        "problem": (
            "Write a function `total_invoice(items)` for Hana's bakery point-of-sale system. `items` "
            "is a list of (name, price, quantity) tuples. Return the total cost as a float, rounded to "
            "2 decimal places."
        ),
        "entry_point": "total_invoice",
        "test_code": """
def test(fn):
    assert fn([("bread", 2.5, 2), ("cake", 10.0, 1)]) == 15.0
    assert fn([]) == 0.0
    return True
""",
    },
    {
        "id": "task_068",
        "source": "hand_hard",
        "problem": (
            "Write a function `dedupe_contacts(contacts)` for Rolodex CRM, used by sales rep Fatima "
            "Al-Sayed. `contacts` is a list of dicts each with keys 'email' and 'name'. Two contacts are "
            "duplicates if their emails match case-insensitively. When duplicates are found, keep only "
            "the one whose name is longest (more characters); if there's a tie in name length, keep the "
            "one that appears first in the input list. Edge case: emails must be compared after "
            "lowercasing, but the email in the returned contact should preserve its ORIGINAL casing from "
            "whichever record was kept."
        ),
        "entry_point": "dedupe_contacts",
        "test_code": """
def test(fn):
    contacts = [
        {"email": "Bob@Test.com", "name": "Bob"},
        {"email": "bob@test.com", "name": "Bobby"},
        {"email": "alice@test.com", "name": "Alice"},
    ]
    result = fn(contacts)
    emails = {c["email"].lower(): c for c in result}
    assert len(result) == 2
    assert emails["bob@test.com"]["name"] == "Bobby"
    assert emails["bob@test.com"]["email"] == "bob@test.com"
    assert emails["alice@test.com"]["name"] == "Alice"
    return True
""",
    },
    {
        "id": "task_069",
        "source": "hand_hard",
        "problem": (
            "Write a function `parse_log_level(line, default_level)` for the observability pipeline at "
            "Redshift Systems. `line` is a raw log string that may start with a bracketed level tag like "
            "'[ERROR]', '[WARN]', or '[INFO]' (case-insensitive, must be at the very start of the "
            "string with no leading whitespace). Return a tuple (level, message) where level is the "
            "uppercased tag with brackets stripped, and message is the rest of the line with exactly one "
            "leading space stripped if present. If no valid tag is found at the start, return "
            "(default_level, line) unchanged -- the whole original line as the message, not stripped."
        ),
        "entry_point": "parse_log_level",
        "test_code": """
def test(fn):
    assert fn("[ERROR] disk full", "INFO") == ("ERROR", "disk full")
    assert fn("[warn] low memory", "INFO") == ("WARN", "low memory")
    assert fn("no tag here", "INFO") == ("INFO", "no tag here")
    assert fn(" [ERROR] leading space", "INFO") == ("INFO", " [ERROR] leading space")
    return True
""",
    },
    {
        "id": "task_070",
        "source": "hand_easy",
        "problem": (
            "Write a function `avg_temperature(readings)` for a weather station in Reykjavik. "
            "`readings` is a list of floats in Celsius. Return the average rounded to 1 decimal place. "
            "If the list is empty, return None."
        ),
        "entry_point": "avg_temperature",
        "test_code": """
def test(fn):
    assert fn([1.0, 2.0, 3.0]) == 2.0
    assert fn([]) is None
    assert fn([-5.55, 4.45]) == round((-5.55 + 4.45) / 2, 1)
    return True
""",
    },
    {
        "id": "task_071",
        "source": "hand_hard",
        "problem": (
            "Write a function `split_expense(total, participants, exempt)` for the bill-splitting "
            "feature in the app used by roommates Devon, Priya, and Marco. `participants` is a list of "
            "names, and `exempt` is a set of names who should not pay (e.g. someone who didn't attend). "
            "Split `total` evenly among non-exempt participants, rounding each share DOWN to the nearest "
            "cent, then add any leftover cents (from the rounding) one cent at a time to the "
            "non-exempt participants in the order they appear in `participants`, until the leftover is "
            "fully distributed. Edge case: if every participant is exempt, raise a ValueError with "
            "message 'no one to split with'."
        ),
        "entry_point": "split_expense",
        "test_code": """
def test(fn):
    result = fn(10.0, ["Devon", "Priya", "Marco"], set())
    assert abs(sum(result.values()) - 10.0) < 1e-9
    assert result["Devon"] == 3.34
    assert result["Priya"] == 3.33
    assert result["Marco"] == 3.33
    result2 = fn(10.0, ["Devon", "Priya"], {"Priya"})
    assert result2 == {"Devon": 10.0}
    try:
        fn(5.0, ["Devon"], {"Devon"})
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert str(e) == "no one to split with"
    return True
""",
    },
    {
        "id": "task_072",
        "source": "hand_hard",
        "problem": (
            "Write a function `resolve_import_order(modules)` for the build tool at Compass Software. "
            "`modules` is a dict mapping module name to a list of module names it depends on. Return a "
            "list giving a valid topological order (dependencies before dependents). If there are "
            "multiple valid orders, prefer visiting modules in the order their keys first appear in "
            "`modules`. Edge case: if there is a circular dependency, raise a ValueError with message "
            "'circular dependency detected' rather than looping forever or returning a partial order."
        ),
        "entry_point": "resolve_import_order",
        "test_code": """
def test(fn):
    modules = {"a": ["b"], "b": ["c"], "c": []}
    result = fn(modules)
    assert result.index("c") < result.index("b") < result.index("a")
    modules2 = {"x": [], "y": []}
    result2 = fn(modules2)
    assert set(result2) == {"x", "y"}
    try:
        fn({"a": ["b"], "b": ["a"]})
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert str(e) == "circular dependency detected"
    return True
""",
    },
    {
        "id": "task_073",
        "source": "hand_easy",
        "problem": (
            "Write a function `clean_username(raw)` for the signup form at Lumen Social. It must lower-"
            "case the input, strip leading/trailing whitespace, and replace any run of one or more "
            "spaces in the middle with a single underscore."
        ),
        "entry_point": "clean_username",
        "test_code": """
def test(fn):
    assert fn("  John  Smith ") == "john_smith"
    assert fn("Anna   Lee") == "anna_lee"
    return True
""",
    },
    {
        "id": "task_074",
        "source": "hand_hard",
        "problem": (
            "Write a function `nearest_warehouse(order_coords, warehouses)` for Everline Retail's "
            "shipping system. `order_coords` is an (x, y) tuple, and `warehouses` is a dict mapping "
            "warehouse name to (x, y) coordinates. Return the name of the closest warehouse by straight-"
            "line (Euclidean) distance. Edge case: if two or more warehouses are exactly tied for "
            "closest, return the one whose name comes first alphabetically, not just the first one "
            "encountered during iteration."
        ),
        "entry_point": "nearest_warehouse",
        "test_code": """
def test(fn):
    warehouses = {"Zeta": (0, 3), "Alpha": (0, 3), "Beta": (10, 10)}
    assert fn((0, 0), warehouses) == "Alpha"
    warehouses2 = {"Beta": (1, 1), "Alpha": (100, 100)}
    assert fn((0, 0), warehouses2) == "Beta"
    return True
""",
    },
    {
        "id": "task_075",
        "source": "hand_hard",
        "problem": (
            "Write a function `mask_pii(text, patterns)` for the data-redaction tool used by compliance "
            "officer Grace Okafor. `patterns` is a dict mapping a label (e.g. 'EMAIL', 'PHONE') to a "
            "compiled regular expression. For every match of any pattern in `text`, replace it with "
            "'[REDACTED:{label}]'. If a span of text is matched by more than one pattern, only the "
            "first pattern (in the dict's iteration order) that matches at that position should apply -- "
            "do not double-redact or apply a second pattern on top of an already-redacted span."
        ),
        "entry_point": "mask_pii",
        "test_code": """
import re
def test(fn):
    patterns = {"EMAIL": re.compile(r"[\\w.]+@[\\w.]+"), "DIGITS": re.compile(r"\\d+")}
    text = "contact bob@test.com or call 12345"
    result = fn(text, patterns)
    assert "[REDACTED:EMAIL]" in result
    assert "[REDACTED:DIGITS]" in result
    assert "bob@test.com" not in result
    assert "12345" not in result
    return True
""",
    },
    {
        "id": "task_076",
        "source": "hand_easy",
        "problem": (
            "Write a function `stock_alert(prices, threshold)` for a trading dashboard used by analyst "
            "Wei Zhang. `prices` is a list of floats. Return the list of indices where the price is "
            "strictly greater than `threshold`."
        ),
        "entry_point": "stock_alert",
        "test_code": """
def test(fn):
    assert fn([10, 25, 5, 30], 20) == [1, 3]
    assert fn([1, 2, 3], 100) == []
    return True
""",
    },
    {
        "id": "task_077",
        "source": "hand_hard",
        "problem": (
            "Write a function `expand_abbreviations(text, glossary)` for the technical-writing tool at "
            "Beacon Docs, maintained by editor Sofia Reyes. `glossary` maps an abbreviation (e.g. 'API') "
            "to its full form. On the FIRST occurrence of each abbreviation in `text` (as a whole word, "
            "case-sensitive, using word boundaries so 'APIs' does not match 'API'), replace it with "
            "'{full form} ({abbreviation})'. On every subsequent occurrence of that same abbreviation, "
            "leave it as-is. Edge case: if an abbreviation never appears in `text`, it should simply be "
            "ignored, not raise an error."
        ),
        "entry_point": "expand_abbreviations",
        "test_code": """
def test(fn):
    glossary = {"API": "Application Programming Interface", "SDK": "Software Development Kit"}
    text = "Our API is documented. The API is stable. No SDK yet."
    result = fn(text, glossary)
    assert result.count("Application Programming Interface (API)") == 1
    assert "API is stable" in result
    assert "Software Development Kit (SDK)" in result
    return True
""",
    },
    {
        "id": "task_078",
        "source": "hand_hard",
        "problem": (
            "Write a function `battery_time_remaining(percent, discharge_curve)` for the firmware team "
            "at Halcyon Wearables, led by engineer Tomas Novak. `discharge_curve` is a list of "
            "(percent_threshold, minutes_per_percent) tuples sorted by ASCENDING percent_threshold, "
            "describing tiers of battery drain -- e.g. [(10, 5), (100, 2)] means the last 10% of battery "
            "(0 up to and including 10%) drains slowly at 5 minutes per percent, while everything above "
            "10% up to 100% drains faster at 2 minutes per percent. Estimate total remaining minutes by "
            "summing, for each percentage point from `percent` down to 1 (there is no point below 1), "
            "the minutes-per-percent rate of the FIRST tuple in `discharge_curve` (in the given ascending "
            "order) whose percent_threshold is >= that percentage point. Edge case: if `percent` is 0, "
            "return 0."
        ),
        "entry_point": "battery_time_remaining",
        "test_code": """
def test(fn):
    curve = [(10, 5), (100, 2)]
    assert fn(0, curve) == 0
    assert fn(5, curve) == 5 * 5
    assert fn(10, curve) == 10 * 5
    assert fn(12, curve) == 10 * 5 + 2 * 2
    return True
""",
    },
    {
        "id": "task_079",
        "source": "hand_easy",
        "problem": (
            "Write a function `filter_valid_ages(ages)` for the census tool used by researcher Ana "
            "Bautista. `ages` is a list of integers. Return only the ages that are between 0 and 120 "
            "inclusive; anything else (negative, or over 120) should be dropped."
        ),
        "entry_point": "filter_valid_ages",
        "test_code": """
def test(fn):
    assert fn([25, -5, 120, 121, 0]) == [25, 120, 0]
    assert fn([]) == []
    return True
""",
    },
    {
        "id": "task_080",
        "source": "hand_hard",
        "problem": (
            "Write a function `assign_seats(passengers, seat_map)` for the airline check-in system used "
            "at Meridian Airways. `seat_map` is a dict mapping seat label (e.g. 'A1') to either None "
            "(free) or a passenger name already assigned. `passengers` is a list of names needing seats, "
            "in priority order. Assign each passenger, in order, to the alphabetically-first free seat "
            "at the time of their assignment, mutating a COPY of `seat_map` (the original must be left "
            "unchanged) and returning that copy. Edge case: if there are more passengers than free "
            "seats, raise a ValueError with message 'not enough seats' and do not assign anyone from "
            "that point forward (earlier assignments in the returned copy should still reflect who WAS "
            "assigned before the shortage was hit -- but since an exception is raised, the caller gets "
            "no returned value at all)."
        ),
        "entry_point": "assign_seats",
        "test_code": """
def test(fn):
    seat_map = {"B1": None, "A1": None, "A2": "Existing"}
    result = fn(["Zara", "Lee"], seat_map)
    assert result["A1"] == "Zara"
    assert result["B1"] == "Lee"
    assert seat_map["A1"] is None
    try:
        fn(["Zara", "Lee", "Sam"], {"A1": None})
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert str(e) == "not enough seats"
    return True
""",
    },
    {
        "id": "task_081",
        "source": "hand_hard",
        "problem": (
            "Write a function `consolidate_reviews(reviews, min_count)` for the product page at Orchard "
            "Marketplace. `reviews` is a list of (product_id, rating) tuples where rating is 1-5. "
            "Return a dict mapping product_id to its average rating rounded to 2 decimals, but ONLY for "
            "products with at least `min_count` reviews; products with fewer reviews must be excluded "
            "entirely from the result, not included with a null or zero average."
        ),
        "entry_point": "consolidate_reviews",
        "test_code": """
def test(fn):
    reviews = [("p1", 5), ("p1", 3), ("p2", 4), ("p1", 4)]
    result = fn(reviews, 2)
    assert result == {"p1": 4.0}
    assert "p2" not in result
    return True
""",
    },
    {
        "id": "task_082",
        "source": "hand_easy",
        "problem": (
            "Write a function `count_vowels_in_names(names)` for a data-quality check used by intern "
            "Leo Park. `names` is a list of strings. Return the total count of vowels (a, e, i, o, u, "
            "case-insensitive) across all names combined."
        ),
        "entry_point": "count_vowels_in_names",
        "test_code": """
def test(fn):
    assert fn(["Alice", "Bob"]) == 4
    assert fn([]) == 0
    return True
""",
    },
    {
        "id": "task_083",
        "source": "hand_hard",
        "problem": (
            "Write a function `retry_backoff(attempt, base_delay, max_delay, jitter_fn)` used in the "
            "resilience library at Northwind Cloud, maintained by SRE lead Omar Haddad. Compute an "
            "exponential backoff delay: base_delay * (2 ** attempt), capped at max_delay (never exceed "
            "it), then pass that capped value through `jitter_fn` (a callable taking one float and "
            "returning one float) and return jitter_fn's result. Edge case: `attempt` may be 0, in which "
            "case the uncapped delay is just base_delay itself (2**0 == 1)."
        ),
        "entry_point": "retry_backoff",
        "test_code": """
def test(fn):
    identity = lambda x: x
    assert fn(0, 1.0, 100.0, identity) == 1.0
    assert fn(3, 1.0, 100.0, identity) == 8.0
    assert fn(10, 1.0, 50.0, identity) == 50.0
    doubling = lambda x: x * 2
    assert fn(1, 1.0, 100.0, doubling) == 4.0
    return True
""",
    },
    {
        "id": "task_084",
        "source": "hand_hard",
        "problem": (
            "Write a function `anonymize_dataset(records, sensitive_fields)` for the research pipeline "
            "used by data scientist Ingrid Larsen at Solstice Health. `records` is a list of dicts, and "
            "`sensitive_fields` is a set of key names (e.g. {'ssn', 'name'}). Return a new list of dicts "
            "where every key in `sensitive_fields` present in a record has its value replaced with a "
            "deterministic pseudonym: the string 'ANON_' followed by the first 8 characters of the "
            "SHA-256 hex digest of the original value converted to a string. Keys not in "
            "`sensitive_fields`, and keys in `sensitive_fields` that are simply absent from a given "
            "record, must be left untouched (absent stays absent, not added as ANON_None)."
        ),
        "entry_point": "anonymize_dataset",
        "test_code": """
import hashlib
def test(fn):
    records = [{"name": "Bob", "age": 30}, {"age": 40}]
    result = fn(records, {"name"})
    expected_hash = "ANON_" + hashlib.sha256(str("Bob").encode()).hexdigest()[:8]
    assert result[0]["name"] == expected_hash
    assert result[0]["age"] == 30
    assert "name" not in result[1]
    assert result[1]["age"] == 40
    return True
""",
    },
]
