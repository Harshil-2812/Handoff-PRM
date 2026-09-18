"""Verifies each task_061-084 test_code against a hand-written reference solution."""
import re
import hashlib

def schedule_meeting(bookings, new_slot):
    ns, ne = new_slot
    for s, e in bookings:
        if ns < e and s < ne:
            return False
    return True

def validate_pandas_column(series, dtype):
    if dtype not in ("int", "float", "str"):
        raise ValueError("unsupported dtype")
    if dtype == "int":
        return all(type(v) is int for v in series)
    if dtype == "float":
        return all(isinstance(v, (int, float)) for v in series)
    return all(isinstance(v, str) for v in series)

def route_cost(distances, fuel_price):
    if any(d < 0 for d in distances):
        raise ValueError("invalid distance")
    total_km = sum(distances)
    liters = total_km * 0.35
    return round(liters * fuel_price, 2)

def format_greeting(name, city):
    return f"Welcome, {name[:1].upper() + name[1:]}, from {city}!"

def rate_limit_ok(requests, window_seconds, max_requests, now):
    count = sum(1 for r in requests if now - window_seconds < r <= now)
    return (count + 1) <= max_requests

def merge_configs(base, override):
    result = dict(base)
    for k, v in override.items():
        if v is None:
            result.pop(k, None)
        elif isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result

def total_invoice(items):
    return round(sum(p * q for _, p, q in items), 2)

def dedupe_contacts(contacts):
    best = {}
    for c in contacts:
        key = c["email"].lower()
        if key not in best or len(c["name"]) > len(best[key]["name"]):
            best[key] = c
    return list(best.values())

def parse_log_level(line, default_level):
    m = re.match(r"^\[(ERROR|WARN|INFO)\]", line, re.IGNORECASE)
    if not m:
        return (default_level, line)
    level = m.group(1).upper()
    rest = line[m.end():]
    if rest.startswith(" "):
        rest = rest[1:]
    return (level, rest)

def avg_temperature(readings):
    if not readings:
        return None
    return round(sum(readings) / len(readings), 1)

def split_expense(total, participants, exempt):
    payers = [p for p in participants if p not in exempt]
    if not payers:
        raise ValueError("no one to split with")
    cents = round(total * 100)
    base = cents // len(payers)
    leftover = cents - base * len(payers)
    result = {p: base for p in payers}
    i = 0
    while leftover > 0:
        result[payers[i % len(payers)]] += 1
        leftover -= 1
        i += 1
    return {p: round(v / 100, 2) for p, v in result.items()}

def resolve_import_order(modules):
    visited, visiting, order = set(), set(), []
    def visit(m):
        if m in visited:
            return
        if m in visiting:
            raise ValueError("circular dependency detected")
        visiting.add(m)
        for dep in modules.get(m, []):
            visit(dep)
        visiting.discard(m)
        visited.add(m)
        order.append(m)
    for m in modules:
        visit(m)
    return order

def clean_username(raw):
    s = raw.strip().lower()
    return re.sub(r" +", "_", s)

def nearest_warehouse(order_coords, warehouses):
    ox, oy = order_coords
    best_name, best_dist = None, None
    for name in sorted(warehouses.keys()):
        x, y = warehouses[name]
        d = ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5
        if best_dist is None or d < best_dist:
            best_dist, best_name = d, name
    return best_name

def mask_pii(text, patterns):
    spans = []
    for label, pat in patterns.items():
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), label))
    spans.sort(key=lambda s: s[0])
    result, last = [], 0
    for start, end, label in spans:
        if start < last:
            continue
        result.append(text[last:start])
        result.append(f"[REDACTED:{label}]")
        last = end
    result.append(text[last:])
    return "".join(result)

def stock_alert(prices, threshold):
    return [i for i, p in enumerate(prices) if p > threshold]

def expand_abbreviations(text, glossary):
    for abbr, full in glossary.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        state = {"done": False}
        def repl(m):
            if state["done"]:
                return m.group(0)
            state["done"] = True
            return f"{full} ({abbr})"
        text = re.sub(pattern, repl, text)
    return text

def battery_time_remaining(percent, discharge_curve):
    if percent == 0:
        return 0
    total = 0
    for p in range(percent, 0, -1):
        for threshold, rate in discharge_curve:
            if threshold >= p:
                total += rate
                break
    return total

def filter_valid_ages(ages):
    return [a for a in ages if 0 <= a <= 120]

def assign_seats(passengers, seat_map):
    result = dict(seat_map)
    free = sorted([s for s, v in result.items() if v is None])
    if len(passengers) > len(free):
        raise ValueError("not enough seats")
    for p, seat in zip(passengers, free):
        result[seat] = p
    return result

def consolidate_reviews(reviews, min_count):
    from collections import defaultdict
    sums, counts = defaultdict(int), defaultdict(int)
    for pid, r in reviews:
        sums[pid] += r
        counts[pid] += 1
    return {pid: round(sums[pid] / counts[pid], 2) for pid in sums if counts[pid] >= min_count}

def count_vowels_in_names(names):
    return sum(1 for n in names for c in n.lower() if c in "aeiou")

def retry_backoff(attempt, base_delay, max_delay, jitter_fn):
    delay = min(base_delay * (2 ** attempt), max_delay)
    return jitter_fn(delay)

def anonymize_dataset(records, sensitive_fields):
    result = []
    for r in records:
        new_r = dict(r)
        for k in sensitive_fields:
            if k in new_r:
                h = hashlib.sha256(str(new_r[k]).encode()).hexdigest()[:8]
                new_r[k] = "ANON_" + h
        result.append(new_r)
    return result


REFS = {
    "schedule_meeting": schedule_meeting,
    "validate_pandas_column": validate_pandas_column,
    "route_cost": route_cost,
    "format_greeting": format_greeting,
    "rate_limit_ok": rate_limit_ok,
    "merge_configs": merge_configs,
    "total_invoice": total_invoice,
    "dedupe_contacts": dedupe_contacts,
    "parse_log_level": parse_log_level,
    "avg_temperature": avg_temperature,
    "split_expense": split_expense,
    "resolve_import_order": resolve_import_order,
    "clean_username": clean_username,
    "nearest_warehouse": nearest_warehouse,
    "mask_pii": mask_pii,
    "stock_alert": stock_alert,
    "expand_abbreviations": expand_abbreviations,
    "battery_time_remaining": battery_time_remaining,
    "filter_valid_ages": filter_valid_ages,
    "assign_seats": assign_seats,
    "consolidate_reviews": consolidate_reviews,
    "count_vowels_in_names": count_vowels_in_names,
    "retry_backoff": retry_backoff,
    "anonymize_dataset": anonymize_dataset,
}

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from tasks.hard_tasks_3 import HARD_TASKS_3

    passed, failed = 0, []
    for task in HARD_TASKS_3:
        ep = task["entry_point"]
        ref_fn = REFS.get(ep)
        if ref_fn is None:
            failed.append((task["id"], "no reference implementation"))
            continue
        namespace = {}
        try:
            exec(task["test_code"], namespace)
            ok = namespace["test"](ref_fn)
            if ok:
                passed += 1
                print(f"  OK {task['id']} ({ep})")
            else:
                failed.append((task["id"], "test() returned falsy"))
        except Exception as e:
            failed.append((task["id"], str(e)))

    print(f"\n{passed}/{len(HARD_TASKS_3)} passed")
    if failed:
        print("FAILURES:")
        for tid, err in failed:
            print(f"  {tid}: {err}")
