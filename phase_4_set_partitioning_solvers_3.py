# -*- coding: utf-8 -*-
"""Phase- 4-SET PARTITIONING-Solvers 4.ipynb

This module implements a CSV-driven Set Partitioning solver for crew pairing selection. Given a universe of flight legs and a 
candidate set of pairings, it formulates a binary integer program that selects a minimum-cost subset of pairings covering each 
leg exactly once. The implementation emphasizes robustness to incomplete inputs, supports externally provided incidence and 
cost data, and includes basic infeasibility diagnostics to aid data validation and model debugging.

MODEL LOGIC:

# constraints: each leg at least once
for i in range(self.m):
    prob += lpSum(self.a[i][j] * x[j] for j in range(self.n)) >= 1



"""


# ============================================================

import csv
from pathlib import Path
import pulp
import sys


class SPPFromCSV:
    def __init__(self, instance_folder):
        """
        Implements a lightweight Set Partitioning solver built from CSV-based inputs.

        Legs represent atomic coverage requirements, pairings represent candidate columns, and the resulting model selects 
        a minimum-cost subset of pairings such that every leg is covered exactly once. The class is designed to tolerate 
        partially specified inputs and infer missing structures when necessary.
        """
        self.instance_folder = Path(instance_folder)
        self.pairings = []           # dict list: {pairing_index, pairing_id, base, legs}
        self.legs = []               # list of leg_id strings
        self.leg_to_index = {}       # map leg_id -> leg_index
        self.n = 0                   # number of pairings
        self.m = 0                   # number of legs
        self.c = []                  # pairing costs
        self.a = None                # incidence matrix (m x n)

    # ============================================================
    #  LOAD legs.csv
    # ============================================================
    def load_legs_csv(self, path=None):
        """
        Initializes the solver state and allocates storage for legs, pairings, costs, and the incidence structure.
        No file I/O is performed here; all data loading is deferred to explicit loader methods to allow flexible execution 
        order during debugging.
        """
        path = Path(path or self.instance_folder / "legs.csv")
        if not path.exists():
            raise FileNotFoundError(f"legs.csv not found at {path}")

        leg_rows = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = [h.strip() for h in reader.fieldnames]

            # Identify leg id column
            leg_id_col = None
            for h in fields:
                if h.lower() in ("leg_id", "leg"):
                    leg_id_col = h
            if not leg_id_col:
                raise ValueError("legs.csv must contain a 'leg_id' column.")

            for row in reader:
                leg_rows.append(row[leg_id_col].strip())

        self.legs = leg_rows
        self.leg_to_index = {leg: i for i, leg in enumerate(self.legs)}
        self.m = len(self.legs)

        print(f"Loaded {self.m} legs")

    # ============================================================
    #  LOAD pairings.csv
    # ============================================================
    def load_pairings_csv(self, path=None):
        """
        Loads the universe of legs from legs.csv and assigns each a stable internal index.
        This index set defines the coverage constraints in the SPP and is treated as authoritative unless additional legs are 
        discovered when inferring incidence.
        """
        path = Path(path or self.instance_folder / "pairings.csv")
        if not path.exists():
            raise FileNotFoundError(f"pairings.csv not found at {path}")

        pairings = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = [h.strip() for h in reader.fieldnames]

            # Identify columns
            idx_col = None
            id_col = None
            base_col = None
            legs_col = None

            for h in fields:
                low = h.lower()
                if low in ("pairing_index", "index"):
                    idx_col = h
                elif low in ("pairing_id", "id"):
                    id_col = h
                elif low == "base":
                    base_col = h
                elif low in ("legs_semicolon", "legs", "legs_list"):
                    legs_col = h

            auto_idx = 0

            for row in reader:
                # pairing index
                pairing_index = int(row[idx_col]) if idx_col and row[idx_col].strip() else auto_idx
                auto_idx += 1

                pairing_id = row[id_col].strip() if id_col and row[id_col].strip() else str(pairing_index)
                base = row[base_col].strip() if base_col and row[base_col].strip() else None

                legs = []
                if legs_col and row[legs_col].strip():
                    raw = row[legs_col].strip()
                    if ";" in raw:
                        legs = [t.strip() for t in raw.split(";") if t.strip()]
                    else:
                        legs = [t.strip() for t in raw.split(",") if t.strip()]

                pairings.append({
                    "pairing_index": pairing_index,
                    "pairing_id": pairing_id,
                    "base": base,
                    "legs": legs
                })

        # sort and normalize indices
        pairings.sort(key=lambda p: p["pairing_index"])
        for new_idx, p in enumerate(pairings):
            p["pairing_index"] = new_idx

        self.pairings = pairings
        self.n = len(pairings)
        print(f"Loaded {self.n} pairings")

    # ============================================================
    #  LOAD incidence.csv (optional)
    # ============================================================
    def load_incidence_csv(self, path=None):
        """
        Loads a precomputed leg–pairing incidence matrix when available.
        This allows external generation or modification of the constraint structure and bypasses inference from 
        pairing definitions. If the file is missing, incidence is inferred instead.
        """
        path = Path(path or self.instance_folder / "incidence.csv")
        if not path.exists():
            print("No incidence.csv found. Will infer from pairings instead.")
            return False

        self.a = [[0] * self.n for _ in range(self.m)]

        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = [h.strip() for h in reader.fieldnames]

            li_col = None
            pj_col = None

            for h in fields:
                if h.lower() in ("leg_index", "leg"):
                    li_col = h
                if h.lower() in ("pairing_index", "pairing"):
                    pj_col = h

            for row in reader:
                li = int(row[li_col])
                pj = int(row[pj_col])
                if 0 <= li < self.m and 0 <= pj < self.n:
                    self.a[li][pj] = 1

        print("Loaded incidence matrix from incidence.csv")
        return True

    # ============================================================
    #  INFER incidence from pairings.csv
    # ============================================================
    def infer_incidence(self):
        """
        Constructs the leg–pairing incidence matrix directly from the leg lists embedded in each pairing.
        Any previously unseen legs encountered during this process are added dynamically, expanding the leg universe to ensure 
        model feasibility rather than failing early.
        """
        print("Inferring incidence from pairings...")
        # ensure all legs exist
        for p in self.pairings:
            for leg in p["legs"]:
                if leg not in self.leg_to_index:
                    print(f"WARNING: leg {leg} not in legs.csv — adding it.")
                    self.leg_to_index[leg] = self.m
                    self.legs.append(leg)
                    self.m += 1

        # build matrix
        self.a = [[0] * self.n for _ in range(self.m)]
        for p in self.pairings:
            pj = p["pairing_index"]
            for leg in p["legs"]:
                li = self.leg_to_index[leg]
                self.a[li][pj] = 1

        print("Incidence matrix constructed.")

    # ============================================================
    #  LOAD costs.csv (optional)
    # ============================================================
    def load_costs_csv(self, path=None):
        """
        Loads pairing costs from costs.csv, resolving entries by pairing index or identifier.
        If no explicit costs are provided, a simple proxy cost based on pairing length is used to keep the optimization 
        well-posed during early testing.
        NOTE: costs.csv needs to be FIXED!!!!
        """
        path = Path(path or self.instance_folder / "costs.csv")
        if not path.exists():
            print("No costs.csv found — using cost = number of legs per pairing.")
            self.c = [len(p["legs"]) for p in self.pairings]
            return

        self.c = [len(p["legs"]) for p in self.pairings]

        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = [h.strip() for h in reader.fieldnames]

            idx_col = None
            id_col = None
            cost_col = None

            for h in fields:
                low = h.lower()
                if low in ("pairing_index", "index"):
                    idx_col = h
                elif low in ("pairing_id", "id"):
                    id_col = h
                elif "cost" in low:
                    cost_col = h

            for row in reader:
                # resolve pairing index
                if idx_col and row[idx_col].strip():
                    pj = int(row[idx_col])
                elif id_col and row[id_col].strip():
                    pid = row[id_col].strip()
                    pj = next((p["pairing_index"] for p in self.pairings if p["pairing_id"] == pid), None)
                else:
                    continue

                if 0 <= pj < self.n:
                    self.c[pj] = float(row[cost_col])

        print("Loaded costs from costs.csv")

    # ============================================================
    #  Solve Set-Partitioning Problem
    # ============================================================
    def solve_spp(self):
        """
        Builds and solves the Set Partitioning Problem using a binary linear program. Pulp in this case.
        The objective minimizes total pairing cost subject to exact coverage of every leg. Solution parsing is 
        restricted to optimal solver outcomes to avoid propagating infeasible or partial results.
        """
        if self.a is None:
            self.infer_incidence()

        if not self.c or len(self.c) != self.n:
            self.load_costs_csv()

        prob = pulp.LpProblem("SPP", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(self.n)]

        # objective
        prob += pulp.lpSum(self.c[j] * x[j] for j in range(self.n))

        # constraints: each leg exactly once
        for i in range(self.m):
            prob += pulp.lpSum(self.a[i][j] * x[j] for j in range(self.n)) == 1

        print("Solving...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        status = pulp.LpStatus[prob.status]
        print("Status:", status)

        # FIXED: Only process solution if optimal
        if status == "Optimal":
            obj_value = pulp.value(prob.objective)
            print("Objective value:", obj_value)

            solution = [int(pulp.value(x[j])) for j in range(self.n)]
            selected = [j for j, val in enumerate(solution) if val == 1]

            print(f"\nSelected {len(selected)} pairings out of {self.n}")
            return selected
        else:
            print("Problem is not optimal. No solution available.")
            print("\nDiagnosing infeasibility...")
            self.diagnose_infeasibility()
            return []

    # ============================================================
    #  Diagnose Infeasibility
    # ============================================================
    def diagnose_infeasibility(self):
        """Check which legs cannot be covered
        
        Performs simple structural diagnostics when the SPP is infeasible.
        Identifies legs that are not covered by any pairing and reports coverage multiplicity for sanity checking the incidence structure.
        """
        uncoverable_legs = []

        for i in range(self.m):
            # Check if leg i appears in any pairing
            if sum(self.a[i][j] for j in range(self.n)) == 0:
                uncoverable_legs.append(i)

        if uncoverable_legs:
            print(f"\nFound {len(uncoverable_legs)} legs that don't appear in any pairing:")
            for i in uncoverable_legs[:10]:  # Show first 10
                print(f"  - Leg index {i}: {self.legs[i]}")
            if len(uncoverable_legs) > 10:
                print(f"  ... and {len(uncoverable_legs) - 10} more")

        # Check for legs that appear in multiple pairings (good for debugging)
        multi_coverage = []
        for i in range(self.m):
            count = sum(self.a[i][j] for j in range(self.n))
            if count > 1:
                multi_coverage.append((i, count))

        if multi_coverage:
            print(f"\n{len(multi_coverage)} legs appear in multiple pairings (this is OK)")
            print(f"Average coverage: {sum(c for _, c in multi_coverage) / len(multi_coverage):.2f} pairings per leg")

    # ============================================================
    #  Convenience Pipeline
    # ============================================================
    def run_all(self):
        self.load_legs_csv()
        self.load_pairings_csv()

        loaded = self.load_incidence_csv()
        if not loaded:
            self.infer_incidence()

        self.load_costs_csv()
        return self.solve_spp()


# ============================================================
#  COLAB RUNTIME EXECUTION
# ============================================================

# CHANGE THIS TO YOUR FOLDER
INSTANCE_PATH = "/content/sample_data/instance1/"

solver = SPPFromCSV(INSTANCE_PATH)
solution = solver.run_all()

print("\nDone.")
