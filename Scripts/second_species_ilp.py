"""
Second Species Counterpoint — Integer Linear Program

Install:
    pip install pulp

Run:
    python second_species_ilp.py
"""

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
# CONSTANTS
# =============================================================================

# Cantus firmus
# Cf = [0, -5, -7, -8, -7, -10, -5, -8, -10, -12]
Cf_list =[
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
    [0, 5, 4, 7, 5, 4, 0]
]

for Cf in Cf_list:
    N = len(Cf)

    # Second species timeline:
    # two CP notes per CF note, except the final CF note gets only one final CP note.
    # Example for N=10: CP has 19 notes.
    S = 2 * N - 1

    S0 = range(S)
    S1 = range(S - 1)
    S2 = range(S - 2)
    S3 = range(S - 3)
    S4 = range(S - 4)

    # Map each CP note to its current CF note.
    CfAt = [Cf[s // 2] for s in S0]

    # Beat type
    Downbeats = [s for s in S0 if s % 2 == 0]
    Upbeats = [s for s in S0 if s % 2 == 1]

    # CF movement between adjacent CP timepoints
    CfStep = [CfAt[s + 1] - CfAt[s] for s in S1]
    CfUp = [1 if CfStep[s] > 0 else 0 for s in S1]
    CfDown = [1 if CfStep[s] < 0 else 0 for s in S1]

    # Harmonic intervals
    H_CONS = [0, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 19, 20, 21, 24]
    H_DISS = [1, 2, 6, 10, 11, 13, 14, 18, 22, 23]
    H = sorted(set(H_CONS + H_DISS))

    # Melodic intervals allowed in CP
    M_pos = [1, 2, 3, 4, 5, 7, 8, 12]
    M_neg = [-1, -2, -3, -4, -5, -7, -8, -12]
    M = M_pos + M_neg

    smallLeaps = [i for i in M if abs(i) not in (1, 2)]
    largeLeaps = [i for i in M if abs(i) > 4]

    # CP pitch domain
    P = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21]

    # Soft-ish thresholds
    MinConjunct = S // 2
    MinContrary = max(1, N // 3)

    Width = 100

    # =============================================================================
    # MODEL
    # =============================================================================

    prob = LpProblem("SecondSpeciesCounterpoint", LpMinimize)

    # =============================================================================
    # DECISION VARIABLES
    # =============================================================================

    h = {(s, i): LpVariable(f"h_{s}_{i}", cat=LpBinary) for s in S0 for i in H}
    m = {(s, i): LpVariable(f"m_{s}_{i}", cat=LpBinary) for s in S1 for i in M}
    p = {(s, u): LpVariable(f"p_{s}_{u}", cat=LpBinary) for s in S0 for u in P}

    hInterval = {s: LpVariable(f"hInterval_{s}", cat=LpInteger) for s in S0}
    mInterval = {s: LpVariable(f"mInterval_{s}", cat=LpInteger) for s in S1}

    up = {s: LpVariable(f"up_{s}", cat=LpBinary) for s in S1}
    down = {s: LpVariable(f"down_{s}", cat=LpBinary) for s in S1}
    conjunct = {s: LpVariable(f"conjunct_{s}", cat=LpBinary) for s in S1}
    contrary = {s: LpVariable(f"contrary_{s}", cat=LpBinary) for s in S1}
    turn = {s: LpVariable(f"turn_{s}", cat=LpBinary) for s in S2}

    consecutiveLeap = {
        s: LpVariable(f"consLeap_{s}", lowBound=0, upBound=2, cat=LpInteger)
        for s in S2
    }

    largeLeap = {s: LpVariable(f"largeLeap_{s}", cat=LpBinary) for s in S1}

    isDissonant = {s: LpVariable(f"diss_{s}", cat=LpBinary) for s in S0}

    climax = {s: LpVariable(f"climax_{s}", cat=LpBinary) for s in S0}
    maxP = LpVariable("maxP", lowBound=0, upBound=21, cat=LpInteger)

    # =============================================================================
    # BASIC RELATIONSHIPS
    # =============================================================================

    for s in S1:
        prob += lpSum(m[s, i] for i in M) == 1, f"one_melodic_interval_{s}"

    for s in S0:
        prob += lpSum(p[s, u] for u in P) == 1, f"one_pitch_{s}"
        prob += lpSum(h[s, i] for i in H) == 1, f"one_harmonic_interval_{s}"

    for s in S0:
        prob += hInterval[s] == lpSum(i * h[s, i] for i in H), f"hInterval_def_{s}"

    for s in S1:
        prob += mInterval[s] == lpSum(i * m[s, i] for i in M), f"mInterval_def_{s}"

    for s in S1:
        prob += (
            mInterval[s] == (hInterval[s + 1] + CfAt[s + 1]) - (hInterval[s] + CfAt[s]),
            f"mh_consistency_{s}",
        )

    for s in S0:
        prob += (
            lpSum(u * p[s, u] for u in P) == CfAt[s] + hInterval[s],
            f"pitch_consistency_{s}",
        )

    # =============================================================================
    # AUXILIARY MOTION VARIABLES
    # =============================================================================

    for s in S1:
        for i in M_pos:
            prob += up[s] >= m[s, i], f"up_forced_{s}_{i}"
        for i in M_neg:
            prob += down[s] >= m[s, i], f"down_forced_{s}_{i}"
        prob += up[s] + down[s] == 1, f"up_down_exclusive_{s}"

    for s in S1:
        prob += (
            contrary[s] == CfUp[s] * down[s] + CfDown[s] * up[s],
            f"contrary_def_{s}",
        )

    for s in S1:
        prob += (
            conjunct[s] == m[s, 1] + m[s, -1] + m[s, 2] + m[s, -2],
            f"conjunct_def_{s}",
        )

    for s in S2:
        prob += (
            consecutiveLeap[s] == lpSum(m[s, i] + m[s + 1, i] for i in smallLeaps),
            f"consLeap_def_{s}",
        )

    for s in S1:
        prob += largeLeap[s] == lpSum(m[s, i] for i in largeLeaps), f"largeLeap_def_{s}"

    for s in S2:
        prob += up[s] + down[s + 1] <= 1 + turn[s], f"turn_ud_{s}"
        prob += down[s] + up[s + 1] <= 1 + turn[s], f"turn_du_{s}"
        prob += up[s] + up[s + 1] <= 1 + (1 - turn[s]), f"turn_uu_{s}"
        prob += down[s] + down[s + 1] <= 1 + (1 - turn[s]), f"turn_dd_{s}"

    # =============================================================================
    # SECOND SPECIES HARMONY RULES
    # =============================================================================

    # Downbeats must be consonant.
    for s in Downbeats:
        prob += lpSum(h[s, i] for i in H_CONS) == 1, f"downbeat_consonant_{s}"

    # Upbeats may be consonant or dissonant.
    for s in Upbeats:
        prob += isDissonant[s] == lpSum(h[s, i] for i in H_DISS), f"diss_def_{s}"

    # Downbeats are never dissonant.
    for s in Downbeats:
        prob += isDissonant[s] == 0, f"downbeat_not_diss_{s}"

    # Dissonant upbeats must be passing tones:
    # step into the dissonance, step out of it, and continue in the same direction.
    for s in Upbeats:
        if s - 1 in S1 and s in S1:
            prob += conjunct[s - 1] >= isDissonant[s], f"diss_step_in_{s}"
            prob += conjunct[s] >= isDissonant[s], f"diss_step_out_{s}"

            prob += up[s - 1] - up[s] <= 1 - isDissonant[s], f"diss_same_dir_a_{s}"
            prob += up[s] - up[s - 1] <= 1 - isDissonant[s], f"diss_same_dir_b_{s}"

    # --- Rule [NEW] Opening interval: unison, fifth, or octave ---
    # Semantic: at s=0 (first downbeat), only h[0,0], h[0,7], or h[0,12] may be active.
    # Fux: the opening must be a perfect consonance.
    prob += (
        h[0, 0] + h[0, 7] + h[0, 12] == 1,
        "opening_perfect_consonance",
    )

    # --- Rule [NEW] Penultimate note: stepwise approach to final whole note ---
    # Semantic: the upbeat at s = S-2 is the CP note immediately before the final
    # whole note at s = S-1. Its melodic interval into the final must be ±1 or ±2.
    # Since ∑_i m[S-2, i] == 1 already, this simply restricts which interval is active.
    # Fux: the penultimate note approaches the final by step (leading tone or supertonic).
    prob += (
        m[S - 2, -2] + m[S - 2, -1] + m[S - 2, 1] + m[S - 2, 2] == 1,
        "penultimate_stepwise_approach",
    )

    # =============================================================================
    # COUNTERPOINT RULES
    # =============================================================================

    # No interior unisons on downbeats.
    for s in Downbeats:
        if s not in (0, S - 1):
            if 0 in H:
                prob += h[s, 0] == 0, f"no_interior_downbeat_unison_{s}"

    # No parallel fifths/octaves between consecutive downbeats.
    for b in range(N - 1):
        s = 2 * b
        nxt = 2 * (b + 1)

        for i1 in [0, 12, 24]:
            for i2 in [0, 12, 24]:
                prob += h[s, i1] + h[nxt, i2] <= 1, f"no_par_octave_{s}_{nxt}_{i1}_{i2}"

        for i1 in [7, 19]:
            for i2 in [7, 19]:
                prob += h[s, i1] + h[nxt, i2] <= 1, f"no_par_fifth_{s}_{nxt}_{i1}_{i2}"

    # No hidden fifths/octaves into downbeats when CP leaps in similar motion.
    similarDownbeat = {
        b: LpVariable(f"similarDownbeat_{b}", cat=LpBinary)
        for b in range(N - 1)
    }

    hiddenTrigger = {
        b: LpVariable(f"hiddenTrigDownbeat_{b}", cat=LpBinary)
        for b in range(N - 1)
    }

    for b in range(N - 1):
        s = 2 * b
        prev_to_downbeat = 2 * b + 1

        if prev_to_downbeat in S1:
            # Use the melodic motion into the next downbeat.
            if Cf[b + 1] > Cf[b]:
                prob += similarDownbeat[b] == up[prev_to_downbeat], f"sim_db_up_{b}"
            elif Cf[b + 1] < Cf[b]:
                prob += similarDownbeat[b] == down[prev_to_downbeat], f"sim_db_down_{b}"
            else:
                prob += similarDownbeat[b] == 0, f"sim_db_static_{b}"

            prob += hiddenTrigger[b] <= similarDownbeat[b], f"hid_db_le_sim_{b}"
            prob += hiddenTrigger[b] <= 1 - conjunct[prev_to_downbeat], f"hid_db_le_leap_{b}"
            prob += (
                hiddenTrigger[b] >= similarDownbeat[b] + (1 - conjunct[prev_to_downbeat]) - 1,
                f"hid_db_ge_{b}",
            )

            arrival = 2 * (b + 1)
            for i in [0, 7, 12, 19, 24]:
                prob += h[arrival, i] <= 1 - hiddenTrigger[b], f"no_hidden_db_{b}_{i}"

    # No triadic arpeggio patterns.
    for s in S2:
        prob += m[s, 3] + m[s, 4] + m[s + 1, 3] + m[s + 1, 4] <= 1, f"no_arp_up_33_{s}"
        prob += m[s, -3] + m[s, -4] + m[s + 1, -3] + m[s + 1, -4] <= 1, f"no_arp_dn_33_{s}"
        prob += m[s, 3] + m[s, 4] + m[s + 1, 5] <= 1, f"no_arp_up_34_{s}"
        prob += m[s, -5] + m[s + 1, -3] + m[s + 1, -4] <= 1, f"no_arp_dn_34_{s}"
        prob += m[s, 5] + m[s + 1, 3] + m[s + 1, 4] <= 1, f"no_arp_up_43_{s}"
        prob += m[s, -3] + m[s, -4] + m[s + 1, -5] <= 1, f"no_arp_dn_43_{s}"

    # No two consecutive leaps in the same direction.
    for s in S2:
        prob += consecutiveLeap[s] <= 1 + turn[s], f"no_consec_same_dir_leaps_{s}"

    # Avoid overusing a pitch.
    for s in S4:
        for u in P:
            prob += lpSum(p[k, u] for k in range(s, s + 5)) <= 2, f"no_triple_pitch_{s}_{u}"

    # If CF leaps from one downbeat to next, CP should not also leap similarly into that downbeat.
    for b in range(N - 1):
        if abs(Cf[b + 1] - Cf[b]) > 4:
            into_next_downbeat = 2 * b + 1
            if into_next_downbeat in S1:
                prob += (
                    largeLeap[into_next_downbeat] <= contrary[into_next_downbeat],
                    f"no_simult_large_leaps_{b}",
                )

    # No more than 3 consecutive downbeat thirds or sixths.
    for b in range(N - 3):
        dbs = [2 * k for k in range(b, b + 4)]

        prob += (
            lpSum(h[s, 3] + h[s, 4] + h[s, 15] + h[s, 16] for s in dbs) <= 3,
            f"no_4par_thirds_{b}",
        )

        prob += (
            lpSum(h[s, 8] + h[s, 9] + h[s, 20] + h[s, 21] for s in dbs) <= 3,
            f"no_4par_sixths_{b}",
        )

    # Prefer stepwise motion.
    prob += lpSum(conjunct[s] for s in S1) >= MinConjunct, "min_conjunct"

    # Prefer some contrary motion only on transitions where the CF actually moves.
    moving_transitions = [s for s in S1 if CfStep[s] != 0]
    prob += lpSum(contrary[s] for s in moving_transitions) >= MinContrary, "min_contrary"

    # Unique climax.
    prob += lpSum(climax[s] for s in S0) == 1, "unique_climax"

    for s in S0:
        cp_pitch = CfAt[s] + hInterval[s]

        prob += cp_pitch + (1 - climax[s]) <= maxP, f"climax_upper_{s}"
        prob += maxP <= cp_pitch + Width * (1 - climax[s]), f"climax_lower_{s}"

    # =============================================================================
    # OBJECTIVE
    # =============================================================================

    prob += (
        lpSum(turn[s] for s in S2)
        - lpSum(conjunct[s] for s in S1)
        - lpSum(contrary[s] for s in moving_transitions),
        "objective",
    )

    # =============================================================================
    # SOLVE
    # =============================================================================

    solver = PULP_CBC_CMD(msg=1)
    prob.solve(solver)

    # =============================================================================
    # OUTPUT
    # =============================================================================

    print(f"\nStatus: {LpStatus[prob.status]}")
    print(f"Objective value: {value(prob.objective):.0f}")
    print()

    if LpStatus[prob.status] == "Optimal":
        print("Cantus firmus:", Cf)
        print("Second species: 2 CP notes per CF note, final bar has 1 CP note")
        print()

        cp_pitches = []
        cp_intervals_h = []

        for s in S0:
            pitch = sum(u * value(p[s, u]) for u in P)
            hi = sum(i * value(h[s, i]) for i in H)

            cp_pitches.append(int(round(pitch)))
            cp_intervals_h.append(int(round(hi)))

        print("Counterpoint pitches:", cp_pitches)
        print("Harmonic intervals:  ", cp_intervals_h)

        print()
        print("Subbeat-by-subbeat:")
        print(
            f"{'s':>4} {'bar':>4} {'beat':>6} {'CF':>4} {'CP':>4} {'HInt':>5} "
            f"{'Diss':>5} {'MInt':>5} {'Up':>3} {'Ctr':>4} {'Cnj':>4} {'Trn':>4}"
        )

        for s in S0:
            bar = s // 2
            beat = "down" if s % 2 == 0 else "up"
            cf_p = CfAt[s]
            cp_p = cp_pitches[s]
            hi = cp_intervals_h[s]

            diss = int(round(value(isDissonant[s]))) if value(isDissonant[s]) is not None else "-"

            if s in S1:
                mi = int(round(value(mInterval[s])))
                u = int(round(value(up[s])))
                ctr = int(round(value(contrary[s])))
                cnj = int(round(value(conjunct[s])))
            else:
                mi = "-"
                u = "-"
                ctr = "-"
                cnj = "-"

            trn_val = value(turn[s]) if s in S2 else None
            trn = int(round(trn_val)) if trn_val is not None else "-"

            print(
                f"{s:>4} {bar:>4} {beat:>6} {cf_p:>4} {cp_p:>4} {hi:>5} "
                f"{str(diss):>5} {str(mi):>5} {str(u):>3} {str(ctr):>4} "
                f"{str(cnj):>4} {str(trn):>4}"
            )

        n_diss = sum(int(round(value(isDissonant[s]))) for s in S0)
        n_conjunct = sum(int(round(value(conjunct[s]))) for s in S1)
        n_contrary = sum(int(round(value(contrary[s]))) for s in moving_transitions)
        n_turns = sum(int(round(value(turn[s]))) for s in S2)

        climax_bar = next(s for s in S0 if round(value(climax[s])) == 1)

        print()
        print(f"Dissonant upbeats: {n_diss}/{len(Upbeats)}")
        print(f"Conjunct motions:  {n_conjunct}/{S-1}")
        print(f"Contrary motions:  {n_contrary}/{len(moving_transitions)}")
        print(f"Turns:             {n_turns}/{S-2}")
        print(f"Climax at subbeat: {climax_bar} (pitch={cp_pitches[climax_bar]})")

        # --- Musical notation ---
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from translator import show_second_species
        show_second_species(Cf, cp_pitches, "Second Species Counterpoint")