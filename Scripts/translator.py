"""
Translator: converts ILP counterpoint output (semitone offsets from C4) to
music21 Score objects for display/export.

Semitone encoding used by the ILP scripts:
    0  = C4 (MIDI 60)
    positive = above C4, negative = below C4
    Diatonic CP range: P = [0,2,4,5,7,9,11,12,14,16,17,19,21] = C4..A5

Species support:
    First species (and invertible):  both voices use whole notes
    Second species:                  CF uses whole notes, CP uses half notes
                                     (last CP note is a whole note)

Usage as a module:
    from translator import show_first_species, show_second_species
    show_first_species(cf, cp_pitches)

Usage as a script (demo with hardcoded CF):
    python translator.py
"""

from music21 import stream, note, meter, metadata, expressions, bar, layout

_MIDI_BASE = 60  # C4


def _make_note(semitone_offset: int, duration_type: str) -> note.Note:
    n = note.Note()
    n.pitch.midi = _MIDI_BASE + semitone_offset
    n.duration.type = duration_type
    return n


def build_first_species_score(
    cf: list[int],
    cp: list[int],
    title: str = "First Species Counterpoint",
    solve_time: float | None = None,
) -> stream.Score:
    """
    cf : semitone offsets for the cantus firmus, one per bar
    cp : semitone offsets for the counterpoint, one per bar (same length as cf)
    """
    if len(cf) != len(cp):
        raise ValueError(f"CF ({len(cf)}) and CP ({len(cp)}) must have equal length")

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = title

    cp_part = stream.Part(id="CTP")
    cp_part.partName = "CTP"
    cp_part.insert(0, meter.TimeSignature("4/4"))
    if solve_time is not None:
        te = expressions.TextExpression(f"solved in {solve_time:.2f}s")
        cp_part.insert(0, te)
    for offset in cp:
        cp_part.append(_make_note(offset, "whole"))

    cf_part = stream.Part(id="CF")
    cf_part.partName = "CF"
    cf_part.insert(0, meter.TimeSignature("4/4"))
    for offset in cf:
        cf_part.append(_make_note(offset, "whole"))

    score.append(cp_part)
    score.append(cf_part)
    return score


def build_second_species_score(
    cf: list[int],
    cp: list[int],
    title: str = "Second Species Counterpoint",
    solve_time: float | None = None,
) -> stream.Score:
    """
    cf : semitone offsets for CF, one per bar (N values)
    cp : semitone offsets for CP, two per bar except last (2*N-1 values)
         — the final value maps to a whole note
    """
    n_bars = len(cf)
    expected = 2 * n_bars - 1
    if len(cp) != expected:
        raise ValueError(f"Second species: expected {expected} CP notes, got {len(cp)}")

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = title

    cp_part = stream.Part(id="CTP")
    cp_part.partName = "CTP"
    cp_part.insert(0, meter.TimeSignature("4/4"))
    if solve_time is not None:
        te = expressions.TextExpression(f"solved in {solve_time:.2f}s")
        cp_part.insert(0, te)
    for i, offset in enumerate(cp):
        dur = "whole" if i == len(cp) - 1 else "half"
        cp_part.append(_make_note(offset, dur))

    cf_part = stream.Part(id="CF")
    cf_part.partName = "CF"
    cf_part.insert(0, meter.TimeSignature("4/4"))
    for offset in cf:
        cf_part.append(_make_note(offset, "whole"))

    score.append(cp_part)
    score.append(cf_part)
    return score


def show_first_species(
    cf: list[int], cp: list[int], title: str = "First Species Counterpoint",
    solve_time: float | None = None,
):
    build_first_species_score(cf, cp, title, solve_time).show()


def build_combined_first_species_score(
    results: list[tuple[list[int], list[int] | None, float | None]],
    title: str = "First Species Counterpoint",
) -> stream.Score:
    """
    results : list of (cf, cp, solve_time)
              cp is None if the solver found no solution → whole rests used
    """
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = title

    cp_part = stream.Part(id="CTP")
    cp_part.partName = "CTP"

    cf_part = stream.Part(id="CF")
    cf_part.partName = "CF"

    for section_idx, (cf, cp, solve_time) in enumerate(results):
        n = len(cf)
        for i in range(n):
            cp_m = stream.Measure()
            cf_m = stream.Measure()

            if section_idx == 0 and i == 0:
                cp_m.insert(0, meter.TimeSignature("4/4"))
                cf_m.insert(0, meter.TimeSignature("4/4"))

            if section_idx > 0 and i == 0:
                cp_m.insert(0, layout.SystemLayout(isNew=True))

            if i == 0:
                if cp is None:
                    cp_m.insert(0, expressions.TextExpression("no solution"))
                elif solve_time is not None:
                    cp_m.insert(0, expressions.TextExpression(f"solved in {solve_time:.2f}s"))

            if cp is not None:
                cp_m.append(_make_note(cp[i], "whole"))
            else:
                r = note.Rest()
                r.duration.type = "whole"
                cp_m.append(r)

            cf_m.append(_make_note(cf[i], "whole"))

            if i == n - 1:
                cp_m.rightBarline = bar.Barline("double")
                cf_m.rightBarline = bar.Barline("double")

            cp_part.append(cp_m)
            cf_part.append(cf_m)

    score.append(cp_part)
    score.append(cf_part)
    return score


def show_combined_first_species(
    results: list[tuple[list[int], list[int] | None, float | None]],
    title: str = "First Species Counterpoint",
):
    build_combined_first_species_score(results, title).show()


def show_second_species(
    cf: list[int], cp: list[int], title: str = "Second Species Counterpoint",
    solve_time: float | None = None,
):
    build_second_species_score(cf, cp, title, solve_time).show()


def _print_score(score: stream.Score):
    """Fallback text display when a GUI renderer is unavailable."""
    for part in score.parts:
        print(f"\n--- {part.partName} ---")
        for n in part.flatten().notes:
            print(f"  {n.pitch.nameWithOctave:<5}  (MIDI {n.pitch.midi})")


if __name__ == "__main__":
    # Demo using the default CF from the ILP files
    cf_demo = [0, -5, -7, -8, -7, -10, -5, -8, -10, -12]
    # Thirds above the CF as a trivial placeholder counterpoint
    cp_demo = [v + 4 for v in cf_demo]

    score = build_first_species_score(cf_demo, cp_demo, "Demo: First Species")

    print("Text representation of the score:")
    _print_score(score)
    print("\nAttempting graphical display...")
    try:
        score.show()
    except Exception as e:
        print(f"Graphical display unavailable ({e}). See text output above.")
