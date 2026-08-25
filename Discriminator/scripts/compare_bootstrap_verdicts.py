"""Compare max-severity scores between two bootstrap-null runs.

Guards refactors of the scoring path: every corruption x metric score should
be unchanged unless the science deliberately changed. Exits non-zero on any
difference outside tolerance, so it can gate a dependent cluster job.
"""
import csv
import sys

USAGE = "usage: compare_bootstrap_verdicts.py <reference-verdicts.csv> <current-verdicts.csv>"

# SCWD runs through a float32 sparse matmul on GPU, which is not bit-reproducible.
# A relative tolerance alone is the wrong test for cells whose score is
# numerically zero (a corruption a metric is completely blind to): there, a
# 2e-11 absolute wobble is a 1e-6 relative one. Require both to be exceeded.
RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-9


def value_metrics(row):
    return {key[: -len("__value")] for key in row if key.endswith("__value")}


def main(reference_path, current_path):
    reference = {row["corruption"]: row for row in csv.DictReader(open(reference_path))}
    current = {row["corruption"]: row for row in csv.DictReader(open(current_path))}
    if not reference or not current:
        raise SystemExit("One of the verdict files is empty.")
    metrics = sorted(value_metrics(next(iter(reference.values())))
                     & value_metrics(next(iter(current.values()))))
    print(f"comparing {len(metrics)} metrics over {len(reference)} corruptions")

    worst_relative = worst_absolute = 0.0
    offenders = []
    for corruption, reference_row in reference.items():
        if corruption not in current:
            offenders.append(f"{corruption}: missing from the new run")
            continue
        for metric in metrics:
            before = float(reference_row[f"{metric}__value"])
            after = float(current[corruption][f"{metric}__value"])
            absolute = abs(before - after)
            relative = absolute / max(abs(before), 1e-30)
            worst_relative = max(worst_relative, relative)
            worst_absolute = max(worst_absolute, absolute)
            if relative > RELATIVE_TOLERANCE and absolute > ABSOLUTE_TOLERANCE:
                offenders.append(
                    f"{corruption} x {metric}: was {before:.8g}, now {after:.8g} "
                    f"(rel {relative:.2e}, abs {absolute:.2e})"
                )
    print(f"worst relative difference: {worst_relative:.3e}")
    print(f"worst absolute difference: {worst_absolute:.3e}")
    if offenders:
        print("REGRESSION:")
        for line in offenders:
            print("  " + line)
        return 1
    print("OK: every max-severity score is unchanged within tolerance.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(USAGE)
    sys.exit(main(sys.argv[1], sys.argv[2]))
