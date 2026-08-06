"""Regenerate the benchmark data from the committed runs in results/.

The benchmark is a reasoning test, so what is *withheld* matters as much as what
is included:

* the two VGG-19 runs differ only in which pretrained checkpoint was loaded, and
  that fact is **blinded** -- they appear as ``run_a`` and ``run_b``. Their layer
  name sequences are identical, which is the structural clue a solver can find.
* run-level provenance that would leak the checkpoint (the command line, the
  weights source and digest, the run slug, the notes) is dropped.
* method is documented in README.md; findings are not. The answer key lives
  outside data/ and is not shipped in the zip.

Usage: python benchmark/lambda-per-layer/build.py
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# The matched core set: every one is trained, reps=250, seed=0, --layers all.
# Architecture and (for the two vgg19 entries) checkpoint are the only things
# that vary, so a difference has few possible causes.
SPEC = [
    ("vgg19-r250-s0-alllayers-fixed-caffe", "vgg19",    "run_a"),
    ("vgg19-r250-s0-alllayers-fixed",       "vgg19",    "run_b"),
    ("vgg19-bn-r250-s0",                    "vgg19_bn", None),
    ("alexnet-r250-s0",                     "alexnet",  None),
    ("resnet50-r250-s0",                    "resnet50", None),
    ("vit-b-16-r250-s0",                    "vit_b_16", None),
]


def _f(v):
    """None for anything not finite -- a fit that failed is not a number."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def main() -> int:
    runs, rows, per_freq = [], [], {}
    freq_ref = contrast_ref = None

    for slug, arch, variant in SPEC:
        path = os.path.join(RESULTS, slug, "result.json")
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            return 1
        r = json.load(open(path))
        meta = r["metadata"]
        assert not meta.get("scramble"), f"{slug} is a scrambled run"

        run_id = f"{arch}__{variant}" if variant else arch
        if freq_ref is None:
            freq_ref, contrast_ref = r["frequencies"], r["contrasts"]
        assert r["frequencies"] == freq_ref, f"{slug}: different frequency grid"
        assert r["contrasts"] == contrast_ref, f"{slug}: different contrast grid"

        # Deliberately NOT copied: metadata.command, metadata.weights,
        # metadata.notes, metadata.code, the slug. Each would name the checkpoint.
        runs.append({
            "run_id": run_id,
            "architecture": arch,
            "n_layers": len(r["layers"]),
            "repetitions": r["repetitions"],
            "seed": meta.get("seed"),
            "input_size_px": r["size"],
        })

        for i, e in enumerate(r["layers"]):
            ci = e.get("lambda_ci") or [None, None]
            mod = (e.get("mean_of_distances") or {}).get("lambda")
            rows.append({
                "run_id": run_id,
                "architecture": arch,
                "layer_index": i,
                "layer_name": e["layer"],
                "lambda": _f(e.get("lambda")),
                "lambda_ci_lo": _f(ci[0]),
                "lambda_ci_hi": _f(ci[1]),
                "lambda_r2": _f(e.get("lambda_r2")),
                "log_fit_r2": _f(e.get("mean_r2")),
                "lambda_alt_metric": _f(mod),
            })
            per_freq.setdefault(run_id, []).append({
                "layer_index": i,
                "layer_name": e["layer"],
                "lambda_per_frequency": [_f(p.get("lambda")) for p in e["per_frequency"]],
            })

    os.makedirs(DATA, exist_ok=True)
    cols = list(rows[0])
    with open(os.path.join(DATA, "lambda_by_layer.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    json.dump({"runs": runs,
               "frequencies_cycles_per_image": freq_ref,
               "contrasts_michelson": contrast_ref,
               "lambda_search_range": [-3.0, 4.0]},
              open(os.path.join(DATA, "runs.json"), "w"), indent=1)
    json.dump({"frequencies_cycles_per_image": freq_ref, "runs": per_freq},
              open(os.path.join(DATA, "per_frequency.json"), "w"), indent=1)

    # A leak check that fails the build rather than warning: no shipped byte may
    # name a checkpoint lineage.
    banned = ("caffe", "in1k", "IMAGENET1K", "oxford", "torchvision")
    for name in os.listdir(DATA):
        blob = open(os.path.join(DATA, name)).read().lower()
        hit = [b for b in banned if b.lower() in blob]
        assert not hit, f"{name} leaks {hit}"

    print(f"{len(runs)} runs, {len(rows)} layer rows -> {DATA}")
    print("leak check passed: no checkpoint lineage in data/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
