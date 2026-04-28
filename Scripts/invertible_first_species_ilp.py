# TODO: Verify with a good cantus firmus

"""
First Species Counterpoint — Integer Linear Program
Based on: Tanaka (2022), SMC, "Formulating First Species Counterpoint with Integer Programming"
Reference rules: Fux, Gradus ad Parnassum (1725), via Jeppesen and the Hiroshi/Noël-Marcel tradition.

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

# Cantus firmus as semitone offsets from tonic (C=0).
# Negative = below the soprano register reference pitch.
# This is the example CF from the paper (D minor: D E F G A Bb C D E D).
#Cf = [0, -5, -7, -8, -7, -10, -5, -8, -10, -12]
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

_M_SET = {1, 2, 3, 4, 5, 7, 8, 12, -1, -2, -3, -4, -5, -7, -8, -12}
_STEPS = {1, 2}


def validate_cf_for_invertible(cf):
    """
    Check that the CF satisfies every melodic rule that the ILP enforces on the
    counterpoint voice.  Returns (True, []) on success, or (False, [reasons]).
    """
    violations = []
    n = len(cf)
    ivs = [cf[t + 1] - cf[t] for t in range(n - 1)]

    # All intervals must be in the allowed melodic set (no repeated pitch, no augmented 2nd, etc.)
    for t, d in enumerate(ivs):
        if d not in _M_SET:
            violations.append(f"interval {d:+d} at bar {t}→{t+1} not in M")

    # Penultimate bar must approach final by step
    if abs(ivs[-1]) not in _STEPS:
        violations.append(f"final interval {ivs[-1]:+d} is not a step (need ±1 or ±2)")

    # No two consecutive leaps in the same direction
    for t in range(n - 2):
        d0, d1 = ivs[t], ivs[t + 1]
        if abs(d0) not in _STEPS and abs(d1) not in _STEPS and (d0 > 0) == (d1 > 0):
            violations.append(f"consecutive same-direction leaps at bars {t}–{t+2}")

    # No triadic arpeggio outlines
    for t in range(n - 2):
        d0, d1 = ivs[t], ivs[t + 1]
        if (
            (d0 in (3, 4)  and d1 in (3, 4))   or
            (d0 in (-3,-4) and d1 in (-3,-4))   or
            (d0 in (3, 4)  and d1 == 5)         or
            (d0 == -5      and d1 in (-3,-4))   or
            (d0 == 5       and d1 in (3, 4))    or
            (d0 in (-3,-4) and d1 == -5)
        ):
            violations.append(f"triadic arpeggio at bars {t}–{t+2} (intervals {d0:+d},{d1:+d})")

    # Minimum conjunct motion (≥ T//2 steps, matching the ILP threshold)
    n_conj = sum(1 for d in ivs if abs(d) in _STEPS)
    if n_conj < n // 2:
        violations.append(f"too few steps: {n_conj}/{n-1} (need ≥ {n // 2})")

    # Unique climax (highest pitch appears exactly once)
    peak = max(cf)
    if cf.count(peak) > 1:
        violations.append(f"climax pitch {peak} appears {cf.count(peak)} times (must be unique)")

    # No pitch repeated more than twice in any 5-bar window
    if n >= 5:
        for t in range(n - 4):
            window = cf[t:t + 5]
            for pitch in set(window):
                if window.count(pitch) > 2:
                    violations.append(
                        f"pitch {pitch} appears {window.count(pitch)}× in bars {t}–{t+4}"
                    )

    return len(violations) == 0, violations


results = []

for Cf in Cf_list:
    valid, violations = validate_cf_for_invertible(Cf)
    if not valid:
        print(f"\nSkipping CF {Cf} — fails melodic validation:")
        for v in violations:
            print(f"  • {v}")
        results.append((Cf, None, None))
        continue

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
    # Fux: unison(0), minor third(3), major third(4), perfect fourth(5) [above bass],
    #      perfect fifth(7), minor sixth(8), major sixth(9), octave(12).
    # NOTE: 5 (perfect fourth) is consonant only when CF is in the bass (standard
    # two-voice 1st species). Tanaka includes it; Fux allows it above the bass.

    # Thirds and sixths for interior bars; unisons/octaves allowed at first/last.
    # Fifths excluded: P5 inverts to P4 (dissonant in strict counterpoint).
    H = [0, 3, 4, 8, 9, 12, 15, 16, 20, 21, 24]

    # Melodic intervals allowed in the counterpoint voice (semitones, signed).
    # Fux allows: semitone (1), whole tone (2), minor third (3), major third (4),
    # perfect fourth (5), perfect fifth (7), minor sixth (8, ascending only in Fux),
    # octave (12). Tanaka encodes 8 as a "small leap" and allows both directions.
    # [FUX-DIFF]: Fux restricts ascending minor sixth (8) but Tanaka allows ±8.
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
    # Tanaka uses C major/D minor soprano register: C4 to A5 roughly.
    # P = {0,2,4,5,7,9,11,12,14,16,17,19,21} (white keys + one chromatic)
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

    # h[t][i]: 1 iff harmonic interval at bar t is i semitones
    h = {(t, i): LpVariable(f"h_{t}_{i}", cat=LpBinary) for t in T0 for i in H}

    # m[t][i]: 1 iff melodic interval from bar t to t+1 is i semitones
    m = {(t, i): LpVariable(f"m_{t}_{i}", cat=LpBinary) for t in T1 for i in M}

    # p[t][u]: 1 iff pitch of counterpoint at bar t is u semitones
    p = {(t, u): LpVariable(f"p_{t}_{u}", cat=LpBinary) for t in T0 for u in P}

    # hInterval[t]: actual harmonic interval at bar t (integer-valued)
    hInterval = {t: LpVariable(f"hInterval_{t}", cat=LpInteger) for t in T0}

    # mInterval[t]: actual melodic interval from bar t to t+1
    mInterval = {t: LpVariable(f"mInterval_{t}", cat=LpInteger) for t in T1}

    # Auxiliary direction and motion variables
    up = {t: LpVariable(f"up_{t}", cat=LpBinary) for t in T1}
    down = {t: LpVariable(f"down_{t}", cat=LpBinary) for t in T1}
    conjunct = {t: LpVariable(f"conjunct_{t}", cat=LpBinary) for t in T1}
    contrary = {t: LpVariable(f"contrary_{t}", cat=LpBinary) for t in T1}
    turn = {t: LpVariable(f"turn_{t}", cat=LpBinary) for t in T1}

    # consecutiveLeap[t]: number of leaps at bars t and t+1 (0, 1, or 2)
    consecutiveLeap = {
        t: LpVariable(f"consLeap_{t}", lowBound=0, upBound=2, cat=LpInteger) for t in T2
    }

    # largeLeap[t]: 1 iff counterpoint has a large leap at bar t
    largeLeap = {t: LpVariable(f"largeLeap_{t}", cat=LpBinary) for t in T1}

    # climax[t]: 1 iff bar t is the climax (highest pitch) of the counterpoint
    climax = {t: LpVariable(f"climax_{t}", cat=LpBinary) for t in T0}

    # maxP: the pitch value of the climax note
    maxP = LpVariable("maxP", lowBound=0, upBound=21, cat=LpInteger)

    # =============================================================================
    # SECTION 3.2 — Basic relationships between variables
    # =============================================================================

    # (1) Exactly one melodic interval per bar transition
    # Semantic: the counterpoint moves by exactly one interval per bar;
    #           no repeated pitch (0 excluded from M), no ambiguity.
    for t in T1:
        prob += lpSum(m[t, i] for i in M) == 1, f"one_melodic_interval_{t}"

    # (2) Exactly one pitch per bar
    # Semantic: the counterpoint has a unique pitch at each bar.
    for t in T0:
        prob += lpSum(p[t, u] for u in P) == 1, f"one_pitch_{t}"

    # (3) Exactly one harmonic interval per bar
    # Semantic: exactly one consonant harmonic interval is realized at each bar.
    for t in T0:
        prob += lpSum(h[t, i] for i in H) == 1, f"one_harmonic_interval_{t}"

    # (4) hInterval is the weighted sum of active h[t,i]
    # Semantic: links the binary encoding h to the integer hInterval.
    for t in T0:
        prob += hInterval[t] == lpSum(i * h[t, i] for i in H), f"hInterval_def_{t}"

    # (5) mInterval is the weighted sum of active m[t,i]
    # Semantic: links binary m to integer mInterval.
    for t in T1:
        prob += mInterval[t] == lpSum(i * m[t, i] for i in M), f"mInterval_def_{t}"

    # (6) Consistency: melodic interval = change in absolute pitch of counterpoint
    # Semantic: ensures m and h are mutually consistent given the fixed CF.
    #   pitch of CP at t   = Cf[t]   + hInterval[t]
    #   pitch of CP at t+1 = Cf[t+1] + hInterval[t+1]
    #   mInterval[t]       = (Cf[t+1] + hInterval[t+1]) - (Cf[t] + hInterval[t])
    for t in T1:
        prob += (
            mInterval[t] == (hInterval[t + 1] + Cf[t + 1]) - (hInterval[t] + Cf[t]),
            f"mh_consistency_{t}",
        )

    # (7) Pitch consistency: p[t,u] encoding agrees with hInterval
    # Semantic: the weighted sum of active pitches equals CF[t] + hInterval[t],
    #           ensuring all three encodings (h, m, p) are mutually consistent.
    for t in T0:
        prob += (
            lpSum(u * p[t, u] for u in P) == Cf[t] + hInterval[t],
            f"pitch_consistency_{t}",
        )

    # =============================================================================
    # SECTION 3.3 — Auxiliary variable constraints
    # =============================================================================

    # (8)–(10) Direction variables up/down
    # Semantic: up[t]=1 iff CP moves up; down[t]=1 iff CP moves down.
    #           Exactly one of {up, down} is active per bar (no unison in M).
    for t in T1:
        for i in M_pos:
            prob += up[t] >= m[t, i], f"up_forced_{t}_{i}"
        for i in M_neg:
            prob += down[t] >= m[t, i], f"down_forced_{t}_{i}"
        prob += up[t] + down[t] == 1, f"up_down_exclusive_{t}"

    # (11) Contrary motion
    # Semantic: contrary[t]=1 iff CP moves opposite to CF.
    #   CF up   (CfUp[t]=1)   and CP down (down[t]=1) → contrary
    #   CF down (CfDown[t]=1) and CP up   (up[t]=1)   → contrary
    # This is a direct equality (both terms are disjoint given CF direction is fixed).
    for t in T1:
        prob += (contrary[t] == CfUp[t] * down[t] + CfDown[t] * up[t], f"contrary_def_{t}")

    # (12) Conjunct motion
    # Semantic: conjunct[t]=1 iff CP moves by a step (±1 or ±2 semitones).
    for t in T1:
        prob += (
            conjunct[t] == m[t, 1] + m[t, -1] + m[t, 2] + m[t, -2],
            f"conjunct_def_{t}",
        )

    # (13) Consecutive leaps
    # Semantic: counts how many of {bar t, bar t+1} have a small leap (non-step).
    #           Value is 0, 1, or 2. Used to detect two consecutive leaps.
    for t in T2:
        prob += (
            consecutiveLeap[t] == lpSum(m[t, i] + m[t + 1, i] for i in smallLeaps),
            f"consLeap_def_{t}",
        )

    # (14) Large leap indicator
    # Semantic: largeLeap[t]=1 iff CP has a large leap (>4 semitones) at bar t.
    #           Only defined at bars where CF itself leaps (paper's choice),
    #           but semantically should cover all bars for rule 3.4.5 completeness.
    #           [FUX-DIFF]: Tanaka only constrains largeLeap at CfLargeLeapT;
    #           Fux prohibits large leaps in CP anywhere without compensation.
    for t in T1:
        prob += (largeLeap[t] == lpSum(m[t, i] for i in largeLeaps), f"largeLeap_def_{t}")

    # (15)–(18) Turn variable
    # Semantic: turn[t]=1 iff CP changes melodic direction between bars t and t+2.
    #   Constraints encode all four cases of direction pairs:
    #   (up,down) or (down,up) → turn=1; (up,up) or (down,down) → turn=0.
    for t in T1:
        if t + 1 in T1:
            prob += up[t] + down[t + 1] <= 1 + turn[t], f"turn_ud_{t}"  # (15)
            prob += down[t] + up[t + 1] <= 1 + turn[t], f"turn_du_{t}"  # (16)
            prob += up[t] + up[t + 1] <= 1 + (1 - turn[t]), f"turn_uu_{t}"  # (17)
            prob += down[t] + down[t + 1] <= 1 + (1 - turn[t]), f"turn_dd_{t}"  # (18)

    # =============================================================================
    # SECTION 3.4 — Rules of Counterpoint
    # =============================================================================

    # --- Rule 3.4.1: No interior unisons or octaves ---
    # Unisons and octaves are only permitted at the first and last bars.
    for t in range(1, T - 1):
        for i in [0, 12, 24]:
            prob += h[t, i] == 0, f"no_interior_perfect_{t}_{i}"

    # --- Rule 3.4.2: No parallel octaves ---
    # Fifths are not in H so only octave-class pairs need blocking.
    for t in T1:
        for i1 in [0, 12, 24]:
            for i2 in [0, 12, 24]:
                prob += h[t, i1] + h[t + 1, i2] <= 1, f"no_par_octave_{t}_{i1}_{i2}"

    # --- Rule 3.4.3: No hidden (direct) octaves ---
    # Fifths not in H, so only block octave-class arrivals via similar motion + leap.
    similar = {t: LpVariable(f"similar_{t}", cat=LpBinary) for t in T1}
    for t in T1:
        if CfUp[t]:
            prob += similar[t] == up[t], f"similar_up_{t}"
        elif CfDown[t]:
            prob += similar[t] == down[t], f"similar_down_{t}"
        else:
            prob += similar[t] == 0, f"similar_static_{t}"

    hiddenTrigger = {t: LpVariable(f"hiddenTrig_{t}", cat=LpBinary) for t in T1}
    for t in T1:
        prob += hiddenTrigger[t] <= similar[t], f"hidTrig_le_sim_{t}"
        prob += hiddenTrigger[t] <= 1 - conjunct[t], f"hidTrig_le_leap_{t}"
        prob += hiddenTrigger[t] >= similar[t] + (1 - conjunct[t]) - 1, f"hidTrig_ge_{t}"
    for t in T1:
        for i in [0, 12, 24]:
            prob += h[t + 1, i] <= 1 - hiddenTrigger[t], f"no_hidden_{t}_{i}"

    # --- Rule 3.4.4: No arpeggio of triads in one direction ---
    # FUX: "Avoid outlining a triad across three consecutive notes in one direction."
    # Semantic: if two consecutive melodic intervals both belong to triad intervals
    #           (m3, M3, P4, etc.) and no turn occurs, a triad arpeggio is outlined.
    #           Each constraint says: at most one of {third, third+third, third+fourth,
    #           fourth+third} can occur in a two-step window without a turn.
    # The six constraints cover: (m3+m3), (m3+M3), (M3+P4), and their mirror images.
    # STATUS: ✓ Semantically correct. Note this is more restrictive than some Fux
    #           readings which only forbid 3-note triadic outlines > an octave.
    for t in T2:
        prob += (
            m[t, 3] + m[t, 4] + m[t + 1, 3] + m[t + 1, 4] <= 1,
            f"no_arp_up_33_{t}",
        )  # (32)
        prob += (
            m[t, -3] + m[t, -4] + m[t + 1, -3] + m[t + 1, -4] <= 1,
            f"no_arp_dn_33_{t}",
        )  # (33)
        prob += m[t, 3] + m[t, 4] + m[t + 1, 5] <= 1, f"no_arp_up_34_{t}"  # (34)
        prob += m[t, -5] + m[t + 1, -3] + m[t + 1, -4] <= 1, f"no_arp_dn_34_{t}"  # (35)
        prob += m[t, 5] + m[t + 1, 3] + m[t + 1, 4] <= 1, f"no_arp_up_43_{t}"  # (36)
        prob += m[t, -3] + m[t, -4] + m[t + 1, -5] <= 1, f"no_arp_dn_43_{t}"  # (37)

    # --- Rule 3.4.5: No two consecutive leaps in the same direction ---
    # FUX: "Two successive leaps in the same direction are forbidden."
    # Semantic: consecutiveLeap[t]=2 means both intervals at t and t+1 are leaps.
    #           This is only allowed if there is a turn (direction change) between them.
    #           consecutiveLeap[t] ≤ 1 + turn[t] enforces: if no turn, at most 1 leap.
    # STATUS: ✓ Correct. Note turn[t] is defined for t in T1, and consecutiveLeap
    #           for t in T2, so t+1 must also be in T1. The indexing aligns correctly.
    for t in T2:
        prob += consecutiveLeap[t] <= 1 + turn[t], f"no_consec_same_dir_leaps_{t}"  # (38)

    # --- Rule 3.4.6: No pitch repeated 3 times in 5 consecutive notes ---
    # FUX: Fux does not state this precisely, but the spirit is against monotony/
    #      overuse of any single pitch. The standard rule is no note repeated
    #      consecutively (already handled by M excluding 0) and avoid overuse.
    # Semantic: sum of p[s,u] over 5 consecutive bars ≤ 2 means pitch u
    #           appears at most twice in any window of 5 notes.
    # [FUX-DIFF]: This is a derived/editorial rule not literally in Gradus. It is
    #              common in French conservatoire pedagogy (Noël-Marcel tradition).
    # STATUS: ✓ Correctly encodes the stated rule. Fux completeness: partial.
    for t in T4:
        for u in P:
            prob += (
                lpSum(p[s, u] for s in range(t, t + 5)) <= 2,
                f"no_triple_pitch_{t}_{u}",
            )  # (39)

    # --- Rule 3.4.7: No simultaneous large leaps in both voices ---
    # FUX: "Avoid large leaps in the counterpoint when the CF also leaps."
    # Semantic: at bars where CF makes a large leap (CfLargeLeapT), if CP also
    #           makes a large leap (largeLeap[t]=1), the motion must be contrary
    #           (contrary[t]=1). Equivalently: largeLeap[t] ≤ contrary[t].
    #           If contrary=0 (parallel/similar), largeLeap must be 0.
    # STATUS: ✓ Correct. Only applied where CF leaps (per Tanaka); a strict Fux
    #           reading would additionally require leap compensation (step in
    #           opposite direction after any large leap) — see FUX-MISSING below.
    for t in CfLargeLeapT:
        prob += largeLeap[t] <= contrary[t], f"no_simult_large_leaps_{t}"  # (40)

    # --- Rule 3.4.8: Large leaps compensated by opposite movement ---
    # FUX: "After a large leap, the melody should move stepwise in the opposite direction."
    # Tanaka's note: "This rule is unnecessary because 3.4.5 is sufficient."
    # ASSESSMENT: [FUX-MISSING] This is NOT fully covered by 3.4.5.
    #   Rule 3.4.5 prevents two consecutive same-direction leaps.
    #   Rule 3.4.8 requires stepwise (not just opposite) motion after a large leap.
    #   Example: leap of a sixth followed by a third in the opposite direction
    #   satisfies 3.4.5 but violates 3.4.8.
    #   A correct encoding would be:
    #     For each t in T1 where largeLeap[t]=1:
    #       m[t+1, -1] + m[t+1, -2] >= largeLeap[t]  (if leap was upward)
    #       m[t+1,  1] + m[t+1,  2] >= largeLeap[t]  (if leap was downward)
    #   This requires linearizing the product (largeLeap[t] * up[t]), which needs
    #   an additional auxiliary variable. Omitted here to match Tanaka exactly,
    #   but flagged as a real deviation from Fux.
    # STATUS: ⚠ NOT IMPLEMENTED — see note above.

    # --- Rule 3.4.9: Avoid voice crossing ---
    # FUX: "The voices must not cross." (fundamental rule)
    # Tanaka's note: "Unnecessary because H is defined to only include positive
    #                 intervals, meaning CP is always above CF."
    # ASSESSMENT: ✓ Correct. Since H = {0,3,4,5,7,8,9,12} (all non-negative),
    #   hInterval[t] ≥ 0 always, so CP pitch = CF[t] + hInterval[t] ≥ CF[t].
    #   Voice crossing is structurally impossible given the variable domain.
    # STATUS: ✓ Handled implicitly by domain of H.

    # --- Rule 3.4.10: No more than 3 consecutive parallel thirds or sixths ---
    # FUX: "Parallel thirds and sixths are pleasant but should not exceed 3 in a row."
    # Semantic: in any window of 4 consecutive bars, the sum of h values that are
    #           thirds (3,4) or sixths (8,9) must be ≤ 3.
    #           This means at most 3 out of 4 consecutive bars can be thirds (or sixths).
    # STATUS: ✓ Correctly encodes Fux's guideline. Note: Tanaka applies this
    #           separately to thirds and sixths (not combined), which is correct—
    #           3 consecutive thirds followed by a sixth would be fine by Fux.
    for t in T3:
        prob += (
            lpSum(h[s, 3] + h[s, 4] for s in range(t, t + 4)) <= 3,
            f"no_4par_thirds_{t}",
        )  # (41)
        prob += (
            lpSum(h[s, 8] + h[s, 9] for s in range(t, t + 4)) <= 3,
            f"no_4par_sixths_{t}",
        )  # (42)
        prob += (
            lpSum(h[s, 15] + h[s, 16] for s in range(t, t + 4)) <= 3,
            f"no_4par_tenths_{t}",
        )  # (43)
        prob += (
            lpSum(h[s, 20] + h[s, 21] for s in range(t, t + 4)) <= 3,
            f"no_4par_thirteenths_{t}",
        )  # (44)

    # --- Rules 3.4.11: Soft/global melodic quality constraints ---
    # FUX: "Contrary motion is preferred. Stepwise motion gives the melody flow.
    #       The melody should have a single climax."
    # Implemented partly as hard thresholds, partly in the objective function below.

    # (45) Minimum contrary motions
    # Semantic: at least T/2 bars must have contrary motion between voices.
    # STATUS: ✓ Encodes Fux's preference for contrary motion as a hard floor.
    prob += lpSum(contrary[t] for t in T1) >= MinContrary, "min_contrary"

    # (46) Minimum conjunct motions
    # Semantic: at least T/2 melodic intervals in CP must be steps.
    # STATUS: ✓ Encodes Fux's preference for stepwise melody as a hard floor.
    prob += lpSum(conjunct[t] for t in T1) >= MinConjunct, "min_conjunct"

    # (47)–(49) Unique climax via Big-M method
    # Semantic: exactly one bar is designated the climax (climax[t]=1).
    #   At the climax bar: CP pitch = maxP (constraints 48-49 activate).
    #   At non-climax bars: constraint (48) is loose (CP pitch ≤ maxP always true
    #   since Width is large), and constraint (49) is trivially satisfied.
    #   Together, this forces the climax to be the unique maximum pitch.
    # [FUX-DIFF]: Fux doesn't prescribe a unique climax explicitly; this is a
    #   French conservatoire addition. It does improve melodic shape significantly.
    # STATUS: ✓ Correct Big-M encoding.
    prob += lpSum(climax[t] for t in T0) == 1, "unique_climax"  # (47)
    for t in T0:
        prob += (
            (Cf[t] + hInterval[t]) + (1 - climax[t]) <= maxP,
            f"climax_upper_{t}",
        )  # (48)
        prob += (
            maxP <= (Cf[t] + hInterval[t]) + Width * (1 - climax[t]),
            f"climax_lower_{t}",
        )  # (49)

    # --- Rule: Opening interval must be unison or octave (no fifth in invertible) ---
    IC = sorted(set(i % 12 for i in H))
    hClass = {(t, c): LpVariable(f"hClass_{t}_{c}", cat=LpBinary) for t in T0 for c in IC}
    for t in T0:
        for c in IC:
            members = [i for i in H if i % 12 == c]
            prob += hClass[t, c] == lpSum(h[t, i] for i in members), f"hClass_def_{t}_{c}"

    prob += hClass[0, 0] == 1, "opening_perfect_consonance"

    # --- Rule: Final bar must be unison or octave ---
    prob += hClass[T - 1, 0] == 1, "final_perfect_unison_class"

    # --- Rule [NEW] Penultimate bar: stepwise approach to final note ---
    # FUX: "The penultimate note should approach the final by step."
    prob += (
        m[T - 2, -2] + m[T - 2, -1] + m[T - 2, 1] + m[T - 2, 2] == 1,
        "penultimate_stepwise_approach",
    )

    # =============================================================================
    # OBJECTIVE FUNCTION (Section 3.5)
    # =============================================================================
    # Minimize turns, maximize contrary and conjunct motions.
    # Only defined over T2 (bars with two successors, where turn is defined).
    # [NOTE]: Tanaka sums over T2 for all three terms for consistency, but
    #         contrary and conjunct are defined over T1. This is a minor
    #         indexing choice that slightly underweights the last bar's
    #         contribution. Semantically harmless for typical CF lengths.
    prob += (
        lpSum(turn[t] for t in T2)
        - lpSum(contrary[t] for t in T2)
        - lpSum(conjunct[t] for t in T2),
        "objective",
    )

    # =============================================================================
    # SOLVE
    # =============================================================================

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
        results.append((cp_pitches, [p + 24 for p in Cf], None, "inverted"))

    else:
        results.append((Cf, None, solve_time))

    # =============================================================================
    # FUX COVERAGE SUMMARY
    # =============================================================================
    print(
        """
    === FUX FIRST SPECIES COVERAGE SUMMARY ===

    RULE                                      TANAKA  STATUS
    ---------------------------------------------------------
    1. Only consonances                       H domain  ✓ (by variable domain)
    2. Unisons at start/end only              3.4.1     ✓
    3. No parallel perfect consonances        3.4.2     ✓ (P5, P8, P1)
    4. No hidden fifths/octaves               3.4.3     ✓
    5. No voice crossing                      3.4.9     ✓ (implicit via H ≥ 0)
    6. Prefer contrary motion                 3.4.11    ✓ (soft + threshold)
    7. Stepwise melody preferred              3.4.11    ✓ (soft + threshold)
    8. Single climax                          3.4.11    ✓ (Big-M)
    9. No consecutive leaps same direction    3.4.5     ✓
    10. No triad arpeggios in one direction   3.4.4     ✓
    11. ≤3 consecutive parallel 3rds/6ths    3.4.10    ✓
    12. No large leap simultaneously both     3.4.7     ✓ (at CF large leaps)
    13. Large leap → stepwise compensation   3.4.8     ⚠ NOT IMPLEMENTED
        (Tanaka claims 3.4.5 covers this — it does not; see comment above)
    14. Avoid pitch overuse                   3.4.6     ✓ (≤2 in any 5-bar window)
    15. Begin/end on tonic or fifth           —         ✗ NOT IN TANAKA
        (Fux: first and last notes should be unison, fifth, or octave
        on the tonic. Tanaka's H allows any consonance at bar 0 and T-1.)
    16. Penultimate bar: leading tone motion  —         ✗ NOT IN TANAKA
        (Fux: the bar before the last should approach the final by step,
        often using the leading tone. Not encoded.)
    """
    )

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translator import show_combined_first_species
show_combined_first_species(results, "Invertible First Species Counterpoint")
