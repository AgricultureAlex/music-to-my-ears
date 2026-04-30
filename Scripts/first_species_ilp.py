"""
First Species Counterpoint — Integer Linear Program
Based on: Tanaka (2022), SMC, "Formulating First Species Counterpoint with Integer Programming"
Reference rules: Fux, Gradus ad Parnassum (1725), via Jeppesen and the Hiroshi/Noel-Marcel tradition.

Install: pip install pulp
Run:     python first_species_ilp.py

All rules are numbered to match the paper's Section 3 numbering.
Fux coverage notes are inline. Deviations from Fux are flagged with [FUX-DIFF].
"""

import time

from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    LpBinary,
    LpInteger,
    lpSum,
    LpStatus,
    value,
    PULP_CBC_CMD,
)

# =============================================================================
# CONSTANTS (Section 3.1.1)
# =============================================================================

Cf_list = [
    [0, -5, -7, -8, -7, -10, -5, -8, -10, -12],

    [0, 2, 4, 2, 5, 7, 5, 4, 2, 0],
    [0, 4, 5, 7, 4, 5, 2, 0],
    [0, 2, 0, 4, 5, 4, 2, 0],

    [0, 7, 9, 7, 5, 4, 2, 0],
    [0, 2, 5, 4, 7, 5, 4, 2, 0],
    [0, 4, 2, 5, 7, 9, 7, 5, 2, 0],

    [0, 2, 4, 7, 9, 7, 5, 4, 2, 0],
    [0, 5, 7, 9, 7, 4, 5, 2, 0],
    [0, 2, 4, 5, 9, 7, 5, 4, 2, 0],

    [0, 2, 4, 5, 7, 4, 5, 9, 7, 5, 4, 2, 0],
    [0, 4, 5, 7, 9, 7, 5, 4, 2, 4, 2, 0],
    [0, 2, 5, 7, 9, 5, 7, 4, 2, 0],

    [0, 2, 4, 5, 7, 5, 4, -1, 0],
    [0, 4, 5, 7, 9, 7, 5, -1, 0],
    [0, 2, 5, 7, 4, 5, -1, 0],

    [0, 2, 4, 5, 7, 5, 4, 0],
    [0, 4, 7, 5, 4, 0],
    [0, 2, 4, 7, 5, 4, 0],

    [0, 2, 4, 5, 9, 7, 5, 4, 0],
    [0, 5, 7, 9, 7, 5, 4, 0],
    [0, 2, 5, 7, 5, 4, 0],

    [0, 4, 5, 7, 9, 5, 4, 0],
    [0, 2, 4, 7, 9, 7, 4, 0],
    [0, 5, 4, 7, 5, 4, 0],
]

results = []

for Cf in Cf_list:
    T = len(Cf)  # number of bars

    # Precompute CF direction arrays
    # CfUp[t] = 1 if CF moves UP from bar t to t+1
    CfUp = [1 if Cf[t + 1] > Cf[t] else 0 for t in range(T - 1)]
    # CfDown[t] = 1 if CF moves DOWN from bar t to t+1
    CfDown = [1 if Cf[t + 1] < Cf[t] else 0 for t in range(T - 1)]

    # Index sets
    T0 = range(T)  # all bars
    T1 = range(T - 1)  # bars with a successor (for melodic intervals)
    T2 = range(T - 2)  # bars with two successors (for consecutive leaps)
    T3 = range(T - 3)  # bars with three successors (for parallel 3rds/6ths run)
    T4 = range(T - 4)  # bars with four successors (for pitch repetition window)

    # Harmonic intervals (semitones above CF) that are consonant in first species.
    H = [0, 3, 4, 7, 8, 9, 12, 15, 16, 17, 19, 20, 21, 24]

    # Melodic intervals allowed in the counterpoint voice (semitones, signed).
    M_pos = [1, 2, 3, 4, 5, 7, 8, 12]  # upward melodic intervals
    M_neg = [-1, -2, -3, -4, -5, -7, -8, -12]  # downward melodic intervals
    M = M_pos + M_neg  # all allowed melodic intervals (no 0 = no repeated pitch)

    # Leaps: any melodic interval that is NOT a step (semitone or whole tone)
    smallLeaps = [i for i in M if abs(i) not in (1, 2)]

    # Large leaps: anything bigger than a third
    largeLeaps = [i for i in M if abs(i) > 4]

    # Bars where CF itself makes a large leap (so we can apply rule 3.4.7)
    CfLargeLeapT = [t for t in T1 if abs(Cf[t + 1] - Cf[t]) > 4]

    # Possible pitches for the soprano counterpoint voice (diatonic, semitones).
    P = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21]

    # Soft constraint thresholds
    MinContrary = T // 2
    MinConjunct = T // 2

    # Big-M constant for climax constraint
    Width = 100

    # =============================================================================
    # MODEL
    # =============================================================================

    prob = LpProblem("FirstSpeciesCounterpoint", LpMinimize)

    # =============================================================================
    # DECISION VARIABLES (Section 3.1.2 & 3.1.3)
    # =============================================================================

    h = {(t, i): LpVariable(f"h_{t}_{i}", cat=LpBinary) for t in T0 for i in H}
    m = {(t, i): LpVariable(f"m_{t}_{i}", cat=LpBinary) for t in T1 for i in M}
    p = {(t, u): LpVariable(f"p_{t}_{u}", cat=LpBinary) for t in T0 for u in P}

    hInterval = {t: LpVariable(f"hInterval_{t}", cat=LpInteger) for t in T0}
    mInterval = {t: LpVariable(f"mInterval_{t}", cat=LpInteger) for t in T1}

    up = {t: LpVariable(f"up_{t}", cat=LpBinary) for t in T1}
    down = {t: LpVariable(f"down_{t}", cat=LpBinary) for t in T1}
    conjunct = {t: LpVariable(f"conjunct_{t}", cat=LpBinary) for t in T1}
    contrary = {t: LpVariable(f"contrary_{t}", cat=LpBinary) for t in T1}
    turn = {t: LpVariable(f"turn_{t}", cat=LpBinary) for t in T1}

    consecutiveLeap = {
        t: LpVariable(f"consLeap_{t}", lowBound=0, upBound=2, cat=LpInteger) for t in T2
    }

    largeLeap = {t: LpVariable(f"largeLeap_{t}", cat=LpBinary) for t in T1}
    climax = {t: LpVariable(f"climax_{t}", cat=LpBinary) for t in T0}
    maxP = LpVariable("maxP", lowBound=0, upBound=21, cat=LpInteger)

    # =============================================================================
    # SECTION 3.2 — Basic relationships between variables
    # =============================================================================

    for t in T1:
        prob += lpSum(m[t, i] for i in M) == 1, f"one_melodic_interval_{t}"

    for t in T0:
        prob += lpSum(p[t, u] for u in P) == 1, f"one_pitch_{t}"

    for t in T0:
        prob += lpSum(h[t, i] for i in H) == 1, f"one_harmonic_interval_{t}"

    for t in T0:
        prob += hInterval[t] == lpSum(i * h[t, i] for i in H), f"hInterval_def_{t}"

    for t in T1:
        prob += mInterval[t] == lpSum(i * m[t, i] for i in M), f"mInterval_def_{t}"

    for t in T1:
        prob += (
            mInterval[t] == (hInterval[t + 1] + Cf[t + 1]) - (hInterval[t] + Cf[t]),
            f"mh_consistency_{t}",
        )

    for t in T0:
        prob += (
            lpSum(u * p[t, u] for u in P) == Cf[t] + hInterval[t],
            f"pitch_consistency_{t}",
        )

    # =============================================================================
    # SECTION 3.3 — Auxiliary variable constraints
    # =============================================================================

    for t in T1:
        for i in M_pos:
            prob += up[t] >= m[t, i], f"up_forced_{t}_{i}"
        for i in M_neg:
            prob += down[t] >= m[t, i], f"down_forced_{t}_{i}"
        prob += up[t] + down[t] == 1, f"up_down_exclusive_{t}"

    for t in T1:
        prob += (contrary[t] == CfUp[t] * down[t] + CfDown[t] * up[t], f"contrary_def_{t}")

    for t in T1:
        prob += (
            conjunct[t] == m[t, 1] + m[t, -1] + m[t, 2] + m[t, -2],
            f"conjunct_def_{t}",
        )

    for t in T2:
        prob += (
            consecutiveLeap[t] == lpSum(m[t, i] + m[t + 1, i] for i in smallLeaps),
            f"consLeap_def_{t}",
        )

    for t in T1:
        prob += (largeLeap[t] == lpSum(m[t, i] for i in largeLeaps), f"largeLeap_def_{t}")

    for t in T1:
        if t + 1 in T1:
            prob += up[t] + down[t + 1] <= 1 + turn[t], f"turn_ud_{t}"
            prob += down[t] + up[t + 1] <= 1 + turn[t], f"turn_du_{t}"
            prob += up[t] + up[t + 1] <= 1 + (1 - turn[t]), f"turn_uu_{t}"
            prob += down[t] + down[t + 1] <= 1 + (1 - turn[t]), f"turn_dd_{t}"

    # =============================================================================
    # SECTION 3.4 — Rules of Counterpoint
    # =============================================================================

    # --- Rule 3.4.1: No interior unisons ---
    for t in range(1, T - 1):
        prob += h[t, 0] == 0, f"no_interior_unison_{t}"

    # --- Rule 3.4.1a: Unison or octave enforced at last bar ---
    IC = sorted(set(i % 12 for i in H))
    hClass = {(t, c): LpVariable(f"hClass_{t}_{c}", cat=LpBinary) for t in T0 for c in IC}
    for t in T0:
        for c in IC:
            members = [i for i in H if i % 12 == c]
            prob += hClass[t, c] == lpSum(h[t, i] for i in members), f"hClass_def_{t}_{c}"
    prob += hClass[T - 1, 0] == 1, "final_perfect_unison_class"

    # --- Rule 3.4.2: No parallel fifths or octaves ---
    for t in T1:
        for i1 in [0, 12, 24]:
            for i2 in [0, 12, 24]:
                prob += h[t, i1] + h[t + 1, i2] <= 1, f"no_par_octave_{t}_{i1}_{i2}"
        for i1 in [7, 19]:
            for i2 in [7, 19]:
                prob += h[t, i1] + h[t + 1, i2] <= 1, f"no_par_fifth_{t}_{i1}_{i2}"

    # --- Rule 3.4.3: No hidden (direct) fifths or octaves ---
    similar = {t: LpVariable(f"similar_{t}", cat=LpBinary) for t in T1}
    for t in T1:
        if CfUp[t]:
            prob += similar[t] == up[t], f"similar_up_{t}"
        elif CfDown[t]:
            prob += similar[t] == down[t], f"similar_down_{t}"

    hiddenTrigger = {t: LpVariable(f"hiddenTrig_{t}", cat=LpBinary) for t in T1}
    for t in T1:
        prob += hiddenTrigger[t] <= similar[t], f"hidTrig_le_sim_{t}"
        prob += hiddenTrigger[t] <= 1 - conjunct[t], f"hidTrig_le_leap_{t}"
        prob += hiddenTrigger[t] >= similar[t] + (1 - conjunct[t]) - 1, f"hidTrig_ge_{t}"
    for t in T1:
        for i in [0, 7, 12, 19, 24]:
            prob += h[t + 1, i] <= 1 - hiddenTrigger[t], f"no_hidden_{t}_{i}"

    # --- Rule 3.4.4: No arpeggio of triads in one direction ---
    for t in T2:
        prob += m[t, 3] + m[t, 4] + m[t + 1, 3] + m[t + 1, 4] <= 1, f"no_arp_up_33_{t}"
        prob += m[t, -3] + m[t, -4] + m[t + 1, -3] + m[t + 1, -4] <= 1, f"no_arp_dn_33_{t}"
        prob += m[t, 3] + m[t, 4] + m[t + 1, 5] <= 1, f"no_arp_up_34_{t}"
        prob += m[t, -5] + m[t + 1, -3] + m[t + 1, -4] <= 1, f"no_arp_dn_34_{t}"
        prob += m[t, 5] + m[t + 1, 3] + m[t + 1, 4] <= 1, f"no_arp_up_43_{t}"
        prob += m[t, -3] + m[t, -4] + m[t + 1, -5] <= 1, f"no_arp_dn_43_{t}"

    # --- Rule 3.4.5: No two consecutive leaps in the same direction ---
    for t in T2:
        prob += consecutiveLeap[t] <= 1 + turn[t], f"no_consec_same_dir_leaps_{t}"

    # --- Rule 3.4.6: No pitch repeated more than twice in 5 consecutive notes ---
    for t in T4:
        for u in P:
            prob += (
                lpSum(p[s, u] for s in range(t, t + 5)) <= 2,
                f"no_triple_pitch_{t}_{u}",
            )

    # --- Rule 3.4.7: No simultaneous large leaps in both voices ---
    for t in CfLargeLeapT:
        prob += largeLeap[t] <= contrary[t], f"no_simult_large_leaps_{t}"

    # --- Rule 3.4.10: No more than 3 consecutive parallel thirds or sixths ---
    for t in T3:
        prob += (
            lpSum(h[s, 3] + h[s, 4] for s in range(t, t + 4)) <= 3,
            f"no_4par_thirds_{t}",
        )
        prob += (
            lpSum(h[s, 8] + h[s, 9] for s in range(t, t + 4)) <= 3,
            f"no_4par_sixths_{t}",
        )
        prob += (
            lpSum(h[s, 15] + h[s, 16] for s in range(t, t + 4)) <= 3,
            f"no_4par_tenths_{t}",
        )
        prob += (
            lpSum(h[s, 20] + h[s, 21] for s in range(t, t + 4)) <= 3,
            f"no_4par_thirteenths_{t}",
        )

    # --- Rule 3.4.11: Soft/global melodic quality constraints ---
    prob += lpSum(contrary[t] for t in T1) >= MinContrary, "min_contrary"
    prob += lpSum(conjunct[t] for t in T1) >= MinConjunct, "min_conjunct"

    prob += lpSum(climax[t] for t in T0) == 1, "unique_climax"
    for t in T0:
        prob += (
            (Cf[t] + hInterval[t]) + (1 - climax[t]) <= maxP,
            f"climax_upper_{t}",
        )
        prob += (
            maxP <= (Cf[t] + hInterval[t]) + Width * (1 - climax[t]),
            f"climax_lower_{t}",
        )

    # --- Rule: Opening interval must be unison, fifth, or octave ---
    prob += h[0, 0] + h[0, 7] + h[0, 12] == 1, "opening_perfect_consonance"

    # --- Rule: Penultimate note approaches final by step ---
    prob += (
        m[T - 2, -2] + m[T - 2, -1] + m[T - 2, 1] + m[T - 2, 2] == 1,
        "penultimate_stepwise_approach",
    )

    # =============================================================================
    # OBJECTIVE FUNCTION (Section 3.5)
    # =============================================================================

    prob += (
        lpSum(turn[t] for t in T2)
        - lpSum(contrary[t] for t in T2)
        - lpSum(conjunct[t] for t in T2),
        "objective",
    )

    # =============================================================================
    # SOLVE
    # =============================================================================

    print(f"\nSolving CF: {Cf}")
    solver = PULP_CBC_CMD(msg=1, timeLimit=600)
    t0 = time.time()
    prob.solve(solver)
    solve_time = time.time() - t0

    # =============================================================================
    # OUTPUT
    # =============================================================================

    print(f"\nStatus: {LpStatus[prob.status]}")
    print(f"Objective value: {value(prob.objective):.0f}")
    print()

    if LpStatus[prob.status] == "Optimal":
        print("Cantus firmus (semitones):", Cf)

        cp_pitches = []
        cp_intervals_h = []
        for t in T0:
            pitch = sum(u * value(p[t, u]) for u in P)
            hi = sum(i * value(h[t, i]) for i in H)
            cp_pitches.append(int(round(pitch)))
            cp_intervals_h.append(int(round(hi)))
        print("Counterpoint pitches (semitones):", cp_pitches)
        print("Harmonic intervals (semitones):  ", cp_intervals_h)

        print()
        print("Bar-by-bar:")
        print(
            f"{'Bar':>4} {'CF':>4} {'CP':>4} {'HInt':>5} {'MInt':>5} "
            f"{'Up':>3} {'Ctr':>4} {'Cnj':>4} {'Trn':>4}"
        )
        for t in T0:
            cf_p = Cf[t]
            cp_p = cp_pitches[t]
            hi = cp_intervals_h[t]
            mi = int(round(value(mInterval[t]))) if t in T1 else "-"
            u = int(round(value(up[t]))) if t in T1 else "-"
            ctr = int(round(value(contrary[t]))) if t in T1 else "-"
            cnj = int(round(value(conjunct[t]))) if t in T1 else "-"
            trn_val = value(turn[t]) if t in T1 and (t + 1) in T1 else None
            trn = int(round(trn_val)) if trn_val is not None else "-"
            print(
                f"{t:>4} {cf_p:>4} {cp_p:>4} {hi:>5} {str(mi):>5} "
                f"{str(u):>3} {str(ctr):>4} {str(cnj):>4} {str(trn):>4}"
            )

        n_contrary = sum(int(round(value(contrary[t]))) for t in T1)
        n_conjunct = sum(int(round(value(conjunct[t]))) for t in T1)
        n_turns = sum(int(round(value(turn[t]))) for t in T1 if t + 1 in T1)
        climax_bar = next(t for t in T0 if round(value(climax[t])) == 1)
        print(f"\nContrary motions: {n_contrary}/{T-1}")
        print(f"Conjunct motions: {n_conjunct}/{T-1}")
        print(f"Turns:            {n_turns}/{T-2}")
        print(f"Climax at bar:    {climax_bar} (pitch={cp_pitches[climax_bar]})")

        results.append((Cf, cp_pitches, solve_time))

    else:
        results.append((Cf, None, solve_time))

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translator import show_combined_first_species
show_combined_first_species(results, "First Species Counterpoint")
