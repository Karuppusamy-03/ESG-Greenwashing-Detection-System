from pathlib import Path

for name in ("runner_final.txt", "runner_err.txt"):
    path = Path(name)
    if not path.exists():
        print(name, "MISSING")
        continue
    raw = path.read_bytes()
    text = raw.decode("utf-16", errors="replace")
    print(f"=== {name} ({len(raw)} bytes) ===")
    interesting = [
        line
        for line in text.splitlines()
        if any(
            key in line
            for key in (
                "PASS:", "FAIL:", "Ran ", "TEST SUMMARY", "Syntax OK",
                "SKIPPED", "skipped=", "workflow",
            )
        )
    ]
    print("\n".join(interesting[-25:]) if interesting else "(no summary lines)")
