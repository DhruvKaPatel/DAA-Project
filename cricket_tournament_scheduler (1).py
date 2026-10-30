"""Cricket Tournament Scheduler Using Greedy Algorithms (DAA mini project).
Part 1: model the matches of a round-robin tournament as a conflict graph (two
        matches conflict if they share a team) and colour it greedily
        (Welsh-Powell). Each colour is one "round" in which no team plays twice.
Part 2: build a day-by-day timetable. Every day the greedy rule picks the most
        urgent match that fits the venues and the rest-day rule.
Part 3: compare it with first-fit schedulers on four tournament set-ups.
"""
import math
import random
from itertools import combinations

TEAMS = ["MUM", "CHE", "BLR", "KOL", "DEL", "HYD", "JAI", "AHM", "PUN", "LKO", "GUW", "RAJ"]
GROUNDS = ["Narendra Modi Stadium", "Wankhede Stadium", "Eden Gardens", "Chepauk Stadium"]
AGING = 0.25      # bonus per day a team has been idle (keeps every team playing regularly)

SCENARIOS = {     # teams, grounds (matches per day), rest days needed between a team's matches
    "Small":     dict(teams=6, venues=1, rest=1),
    "Standard":  dict(teams=8, venues=2, rest=1),
    "Large":     dict(teams=10, venues=3, rest=1),
    "Congested": dict(teams=8, venues=4, rest=2),
}


# ---------------------------------------------------------------- Part 1
def make_matches(n):
    """Single round-robin: every pair of teams plays once -> n(n-1)/2 matches."""
    return list(combinations(range(n), 2))


def build_conflict_graph(matches):
    """Vertex = match. Edge = the two matches share a team (cannot be played together)."""
    graph = {i: set() for i in range(len(matches))}
    by_team = {}
    for i, (a, b) in enumerate(matches):
        by_team.setdefault(a, []).append(i)
        by_team.setdefault(b, []).append(i)
    for group in by_team.values():            # the matches of one team form a clique
        for i, j in combinations(group, 2):
            graph[i].add(j)
            graph[j].add(i)
    return graph


def greedy_rounds(graph):
    """Welsh-Powell greedy colouring: highest degree first, smallest free colour."""
    order = sorted(graph, key=lambda v: (-len(graph[v]), v))
    colour = {}
    for v in order:
        used = {colour[u] for u in graph[v] if u in colour}
        c = 0
        while c in used:
            c += 1
        colour[v] = c
    rounds = [[] for _ in range(max(colour.values()) + 1)]
    for v in sorted(graph):
        rounds[colour[v]].append(v)
    return rounds


def circle_method(n):
    """Classical constructive round-robin (n even): n-1 rounds. Used only as a reference."""
    teams, rounds = list(range(n)), []
    for _ in range(n - 1):
        rounds.append([tuple(sorted((teams[i], teams[n - 1 - i]))) for i in range(n // 2)])
        teams = [teams[0], teams[-1]] + teams[1:-1]
    return rounds


# ---------------------------------------------------------------- Part 2
def greedy_timetable(matches, n, venues, rest, rule="priority"):
    """Day by day, repeatedly pick the best match that is feasible today.
    rule: 'list' = first feasible match in fixture list (no priority),
          'remaining' = teams with most matches left first,
          'priority' = remaining matches + AGING * idle days (used)."""
    remaining = [sum(t in m for m in matches) for t in range(n)]
    last = [-rest] * n                      # day of the team's last match
    pending = list(range(len(matches)))
    schedule, day = [], 0
    while pending:
        day += 1
        used = set()
        for venue in range(venues):
            best, best_score = None, None
            for m in pending:
                a, b = matches[m]
                if a in used or b in used or day - last[a] <= rest or day - last[b] <= rest:
                    continue
                if rule == "list":
                    score = -m
                else:
                    score = remaining[a] + remaining[b]
                    if rule == "priority":
                        score += AGING * ((day - last[a]) + (day - last[b]))
                if best_score is None or score > best_score:
                    best, best_score = m, score
            if best is None:
                break
            a, b = matches[best]
            schedule.append((day, venue, best))
            used |= {a, b}
            last[a] = last[b] = day
            remaining[a] -= 1
            remaining[b] -= 1
            pending.remove(best)
    return schedule


def first_fit(matches, order, n, venues, rest):
    """Baseline: take matches in the given order, put each on the earliest day that fits."""
    team_days = [set() for _ in range(n)]
    load, schedule = {}, []
    for m in order:
        a, b = matches[m]
        d = 1
        while load.get(d, 0) >= venues or any(abs(d - x) <= rest for x in team_days[a] | team_days[b]):
            d += 1
        schedule.append((d, load.get(d, 0), m))
        load[d] = load.get(d, 0) + 1
        team_days[a].add(d)
        team_days[b].add(d)
    return sorted(schedule)


# ---------------------------------------------------------------- Measures and checks
def lower_bound(n, venues, rest):
    """No schedule can finish earlier than this many days."""
    per_day = math.ceil(n * (n - 1) / 2 / venues)           # grounds limit
    per_team = (n - 1) * (rest + 1) - rest                   # rest rule limit
    return max(per_day, per_team)


def measures(schedule, matches, n):
    days = max(d for d, _, _ in schedule)
    played = [[] for _ in range(n)]
    for d, _, m in schedule:
        for t in matches[m]:
            played[t].append(d)
    gaps = [y - x for p in played for x, y in zip(sorted(p), sorted(p)[1:])]
    finish = [max(p) for p in played]
    return {"days": days, "max_gap": max(gaps), "spread": max(finish) - min(finish)}


def is_valid(schedule, matches, n, venues, rest):
    if sorted(m for _, _, m in schedule) != list(range(len(matches))):
        return False                                         # every match exactly once
    slots = [(d, v) for d, v, _ in schedule]
    if len(set(slots)) != len(slots) or any(v >= venues for _, v in slots):
        return False                                         # one match per ground per day
    played = [[] for _ in range(n)]
    for d, _, m in schedule:
        for t in matches[m]:
            played[t].append(d)
    return all(y - x > rest for p in played for x, y in zip(sorted(p), sorted(p)[1:]))


def self_check():
    for n in (4, 6, 8, 10):
        matches = make_matches(n)
        graph = build_conflict_graph(matches)
        assert all(i in graph[j] for i in graph for j in graph[i])               # symmetric
        rounds = greedy_rounds(graph)
        assert sorted(m for r in rounds for m in r) == list(range(len(matches)))  # each match once
        assert all(not (graph[a] & set(r)) for r in rounds for a in r)            # conflict-free
        assert len(rounds) >= n - 1                                               # lower bound
        assert len(circle_method(n)) == n - 1
    for cfg in SCENARIOS.values():
        n, v, r = cfg["teams"], cfg["venues"], cfg["rest"]
        matches = make_matches(n)
        for rule in ("list", "remaining", "priority"):
            assert is_valid(greedy_timetable(matches, n, v, r, rule), matches, n, v, r)
        assert is_valid(first_fit(matches, list(range(len(matches))), n, v, r), matches, n, v, r)
        assert measures(greedy_timetable(matches, n, v, r), matches, n)["days"] >= lower_bound(n, v, r)
    return True


# ---------------------------------------------------------------- Output
def show_timetable(schedule, matches, venues):
    for day in sorted({d for d, _, _ in schedule}):
        games = [f"{GROUNDS[v]}: {TEAMS[matches[m][0]]} vs {TEAMS[matches[m][1]]}"
                 for d, v, m in schedule if d == day]
        print(f"  Day {day:2d} | " + " | ".join(games))


def main():
    print("Self-check of all algorithms:", "PASSED" if self_check() else "FAILED")

    print("\nPART 1 - greedy colouring of the match-conflict graph")
    print("teams matches edges greedy_rounds lower_bound circle_method")
    for n in (4, 6, 8, 10, 12):
        matches = make_matches(n)
        graph = build_conflict_graph(matches)
        edges = sum(len(s) for s in graph.values()) // 2
        print(f"{n:5d} {len(matches):7d} {edges:5d} {len(greedy_rounds(graph)):13d} "
              f"{n - 1:11d} {len(circle_method(n)):13d}")

    cfg = SCENARIOS["Standard"]
    matches = make_matches(cfg["teams"])
    print("\nPART 2 - greedy timetable, Standard set-up (8 teams, 2 grounds, 1 rest day)")
    show_timetable(greedy_timetable(matches, cfg["teams"], cfg["venues"], cfg["rest"]),
                   matches, cfg["venues"])

    print("\nPART 3 - days needed to finish the tournament (random order = mean of 20 seeds)")
    print(f"{'Scenario':10s} {'LowerBnd':>8s} {'FixtureFF':>9s} {'RandomFF':>9s} {'Greedy':>7s}"
          f" {'maxgap(F/G)':>12s} {'spread(F/G)':>12s}")
    for name, c in SCENARIOS.items():
        n, v, r = c["teams"], c["venues"], c["rest"]
        matches = make_matches(n)
        fix = first_fit(matches, list(range(len(matches))), n, v, r)
        gre = greedy_timetable(matches, n, v, r)
        rnd = []
        for seed in range(20):
            order = list(range(len(matches)))
            random.Random(seed).shuffle(order)
            rnd.append(measures(first_fit(matches, order, n, v, r), matches, n)["days"])
        mf, mg = measures(fix, matches, n), measures(gre, matches, n)
        print(f"{name:10s} {lower_bound(n, v, r):8d} {mf['days']:9d} {sum(rnd) / 20:9.1f} {mg['days']:7d}"
              f" {mf['max_gap']:>6d}/{mg['max_gap']:<5d} {mf['spread']:>6d}/{mg['spread']:<5d}")


if __name__ == "__main__":
    main()
