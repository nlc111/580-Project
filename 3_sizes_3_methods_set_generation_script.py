# -*- coding: utf-8 -*-
"""
3-sizes_3_methods_set_generation_script.ipynb

This script generates candidate pairing samples starting from a fixed
initial solution. The goal is not to enumerate all possible pairings,
but to expand the available pairing pool using lightweight, heuristic
perturbations that are cheap to compute.

Three generation modes are supported:
    - local: small structural changes within an existing pairing
    - forced: perturbations that protect low-frequency (“forced”) duties
    - mixed: a weighted combination of local, forced, and mild recombination

The resulting samples are intended for downstream optimization
(e.g., set partitioning experiments) and benchmarking generation speed
under different target sizes and time limits.

This file originated as a Colab notebook and was flattened into a script
for repeatable batch execution.
"""



import time
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path



def parse_solution(filepath):

        """
    Parse an initial pairing solution file into a normalized in-memory
    representation.

    The expected input format lists pairings by ID, base, and a
    comma-separated duty sequence on each line. This function performs
    minimal validation and assumes the file is well-formed.

    Parameters
    ----------
    filepath : str or Path
        Path to the solution text file.

    Returns
    -------
    list of dict
        Each dict contains:
            - id: synthetic pairing identifier
            - base: crew base
            - duties: ordered list of duty strings
    """
    pattern = r'Pairing\s+(\d+)\s*:\s*Base\s+(\w+)\s*:\s*([^;]+);'
    with open(filepath, 'r') as f:
        text = f.read()

    pairings = []
    for pid, base, duties in re.findall(pattern, text):
        pairings.append({
            "id": f"SOL_{pid}",
            "base": base,
            "duties": [d.strip() for d in duties.split(",") if d.strip()],
        })
    return pairings




def cheap_cost(duties):

        """
    Assign a simple proxy cost to a pairing based solely on length.

    This is intentionally crude and is only meant to provide a consistent,
    monotonic cost signal during sampling. It is not intended to reflect
    contractual or operational realism.

    Parameters
    ----------
    duties : list
        Sequence of duties in the pairing.

    Returns
    -------
    int
        Estimated cost.
    """
    return 500 + 100 * len(duties)


# --------------------------------------------------
# Pairing generators
# --------------------------------------------------

def local_perturb(pairing):

        """
    Generate small structural variations of a single pairing.

    The perturbations are conservative by design: removing interior
    duties, trimming ends, or splitting the sequence. This tends to keep
    candidates close to the original solution while still exploring
    nearby alternatives.

    Parameters
    ----------
    pairing : dict
        A pairing dictionary containing a "duties" list.

    Returns
    -------
    list of list
        Candidate duty sequences derived from the original.
    """
    
    d = pairing["duties"]
    out = []

    if len(d) > 2:
        out.append(d[1:])
        out.append(d[:-1])

    for i in range(1, len(d) - 1):
        nd = d[:i] + d[i+1:]
        if len(nd) >= 2:
            out.append(nd)

    mid = len(d) // 2
    if mid >= 2 and len(d) - mid >= 2:
        out.append(d[:mid])
        out.append(d[mid:])

    return out


def forced_duty_generators(solution):

        """
    Identify low-frequency (“forced”) duties and the pairings that contain them.

    A forced duty is defined here as a duty that appears exactly once
    in the solution. These duties are treated carefully during sampling
    to avoid eliminating necessary coverage too aggressively.

    Parameters
    ----------
    solution : list of dict
        Initial solution pairings.

    Returns
    -------
    tuple
        forced : list
            Duties appearing only once.
        by_duty : dict
            Mapping from forced duty to pairings that contain it.
    """
    freq = Counter()
    for p in solution:
        freq.update(p["duties"])

    forced = [d for d, c in freq.items() if c == 1]
    by_duty = defaultdict(list)

    for p in solution:
        for d in p["duties"]:
            if d in forced:
                by_duty[d].append(p)

    return forced, by_duty


def forced_alternative(duty, source_pairing):

       """
    Construct a minimal alternative pairing around a forced duty.

    The intent is to keep the forced duty covered while shortening or
    reshaping the sequence as little as possible. Only local two-duty
    windows are considered.

    Parameters
    ----------
    duty : str
        The forced duty of interest.
    source_pairing : dict
        Pairing that contains the forced duty.

    Returns
    -------
    list or None
        A short duty sequence containing the forced duty, or None if
        no valid alternative exists.
    """
    i = source_pairing["duties"].index(duty)
    d = source_pairing["duties"]

    if i > 0:
        return d[i-1:i+1]
    if i < len(d) - 1:
        return d[i:i+2]

    return None


def generate_sample(
    solution,
    target_size,
    time_limit,
    mode,
    seed=0
):

    """
    Generate a pool of candidate pairings using heuristic sampling.

    Sampling always begins with the original solution pairings and
    expands the pool until either the target size is reached or the
    time limit expires. Duplicate duty sequences are filtered.

    Parameters
    ----------
    solution : list of dict
        Initial solution pairings.
    target_size : int
        Desired number of pairings in the output pool.
    time_limit : float
        Maximum generation time in seconds.
    mode : str
        Sampling strategy: "local", "forced", or "mixed".
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    tuple
        pool : list of dict
            Generated pairing candidates with bases and costs.
        elapsed : float
            Wall-clock time spent generating samples.
    """
    random.seed(seed)
    start = time.time()

    pool = []
    seen = set()

    # always include solution pairings
    for p in solution:
        key = tuple(p["duties"])
        seen.add(key)
        pool.append({
            "base": p["base"],
            "duties": p["duties"],
            "cost": cheap_cost(p["duties"])
        })

    forced, forced_map = forced_duty_generators(solution)

    while len(pool) < target_size and time.time() - start < time_limit:
        if mode == "local":
            src = random.choice(solution)
            candidates = local_perturb(src)

        elif mode == "forced":
            d = random.choice(forced)
            src = random.choice(forced_map[d])
            alt = forced_alternative(d, src)
            candidates = [alt] if alt else []

        elif mode == "mixed":
            r = random.random()
            if r < 0.6:
                src = random.choice(solution)
                candidates = local_perturb(src)
            elif r < 0.9:
                d = random.choice(forced)
                src = random.choice(forced_map[d])
                alt = forced_alternative(d, src)
                candidates = [alt] if alt else []
            else:
                # mild random recombination
                p1, p2 = random.sample(solution, 2)
                cut = min(len(p1["duties"]), len(p2["duties"])) // 2
                candidates = [p1["duties"][:cut] + p2["duties"][cut:]]

        else:
            raise ValueError(f"Unknown mode {mode}")

        for d in candidates:
            if not d or len(d) < 2:
                continue

            key = tuple(d)
            if key in seen:
                continue

            seen.add(key)
            pool.append({
                "base": random.choice(solution)["base"],
                "duties": d,
                "cost": cheap_cost(d)
            })

            if len(pool) >= target_size:
                break

    return pool, time.time() - start


# --------------------------------------------------
# Main driver
# --------------------------------------------------

if __name__ == "__main__":
    solution = parse_solution("/content/sample_data/initialSolution.txt")

    configs = [
        ("0K",        0,       10),
        ("50K",   50_000,      30),
        ("500K", 500_000,     180),
    ]

    modes = ["local", "forced", "mixed"]

    out_root = Path("samples")
    out_root.mkdir(exist_ok=True)

    for name, target, time_limit in configs:
        dirpath = out_root / f"size_{name}"
        dirpath.mkdir(exist_ok=True)

        print(f"\n=== Generating {name} samples ===")

        for mode in modes:
            pool, elapsed = generate_sample(
                solution,
                target_size=max(target, len(solution)),
                time_limit=time_limit,
                mode=mode,
                seed=42
            )

            outfile = dirpath / f"{mode}.json"
            with open(outfile, "w") as f:
                json.dump(pool, f, indent=2)

            print(f"{mode:>6}: {len(pool):>7} pairings "
                  f"in {elapsed:5.1f}s")
