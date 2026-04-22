"""
Portfolio Project 3 — Quality Audit Script
===========================================

Mechanical quality check for the P3 inflation-forecasting repository.
Run from the repository root:

    python portfolio_audit.py

Or with optional flags:

    python portfolio_audit.py --notebook-output-ok   # skip notebook output warnings
    python portfolio_audit.py --quiet                # score only

Exits 0 if no CRITICAL issues, 1 otherwise.
Uses only the Python 3.8+ standard library — no pip install required.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ========================================================================
# Configuration — what P3 portfolio is expected to contain
# ========================================================================

REPO_ROOT = Path(".").resolve()

REQUIRED_FILES = [
    "README.md",
    "ProjectDriven.md",
    "requirements.txt",
    ".gitignore",
    "docs/findings.md",
    "docs/methodology.md",
    "outputs/portfolio/P3_onepager.pdf",
]

RECOMMENDED_FILES = [
    "ProjectScope_v1.md",
    ".env.example",
    "phase8_summary.md",
]

REQUIRED_DIRS = [
    "docs",
    "notebooks",
    "outputs",
    "outputs/figures",
    "outputs/portfolio",
    "scripts",
    "src",
    "data/documentation",
]

EXPECTED_NOTEBOOKS_MIN = 8     # nb 01..09 with some flexibility
EXPECTED_PHASE_SUMMARIES = {
    "phase3_summary.md",
    "phase4_summary.md",
    "phase5_summary.md",
    "phase6_summary.md",
    "phase7_summary.md",
}

# Patterns used for scanning
PLACEHOLDER_PATTERNS = [
    (r"\[your[-_ ]handle\]",          "Generic placeholder"),
    (r"\bkota2003\b",                 "P2 repo owner — verify this matches P3"),
    (r"YOUR[-_ ]GITHUB",              "Uppercase placeholder"),
    (r"TODO\b|FIXME\b|XXX\b",         "Dev-time marker left in docs"),
]

SECRET_PATTERNS = [
    (r"FRED_API_KEY\s*=\s*['\"][a-f0-9]{16,}", "Committed FRED API key"),
    (r"api[_-]?key['\"]?\s*[:=]\s*['\"][A-Za-z0-9]{20,}", "Generic api key"),
    (r"ghp_[A-Za-z0-9]{20,}",        "GitHub personal access token"),
    (r"AKIA[0-9A-Z]{16}",            "AWS access key"),
    (r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----", "Private key block"),
]

LARGE_FILE_THRESHOLD_MB = 50.0   # GitHub soft-warn at 50 MB, hard-reject at 100
WARN_FILE_THRESHOLD_MB = 10.0

BINARY_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".gif", ".ico",
               ".pkl", ".joblib", ".npy", ".npz", ".zip", ".gz"}
SCAN_EXTS = {".md", ".py", ".ipynb", ".txt", ".yml", ".yaml", ".toml", ".cfg"}


# ========================================================================
# Data model
# ========================================================================

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_WARN     = "WARN"
SEVERITY_INFO     = "INFO"
SEVERITY_OK       = "OK"

# Console markers
MARKERS = {
    SEVERITY_OK:       "[ OK ]",
    SEVERITY_INFO:     "[info]",
    SEVERITY_WARN:     "[WARN]",
    SEVERITY_CRITICAL: "[FAIL]",
}


@dataclass
class Issue:
    section: str
    severity: str
    msg: str
    detail: str = ""


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)

    def add(self, section, severity, msg, detail=""):
        self.issues.append(Issue(section, severity, msg, detail))

    def ok(self, section, msg):
        self.add(section, SEVERITY_OK, msg)

    def warn(self, section, msg, detail=""):
        self.add(section, SEVERITY_WARN, msg, detail)

    def critical(self, section, msg, detail=""):
        self.add(section, SEVERITY_CRITICAL, msg, detail)

    def info(self, section, msg, detail=""):
        self.add(section, SEVERITY_INFO, msg, detail)

    def counts(self):
        c = {SEVERITY_OK: 0, SEVERITY_INFO: 0,
             SEVERITY_WARN: 0, SEVERITY_CRITICAL: 0}
        for i in self.issues:
            c[i.severity] += 1
        return c


# ========================================================================
# Check #1 — Repository structure
# ========================================================================

def check_structure(r: Report):
    section = "1. STRUCTURE"

    for path in REQUIRED_FILES:
        p = REPO_ROOT / path
        if not p.exists():
            r.critical(section, f"Missing required file: {path}")
        elif p.stat().st_size == 0:
            r.critical(section, f"Required file is empty: {path}")
        else:
            r.ok(section, f"Found: {path}")

    for path in RECOMMENDED_FILES:
        p = REPO_ROOT / path
        if not p.exists():
            r.warn(section, f"Recommended file missing: {path}")

    for path in REQUIRED_DIRS:
        p = REPO_ROOT / path
        if not p.is_dir():
            r.critical(section, f"Missing required directory: {path}/")

    # Notebook count
    nb_dir = REPO_ROOT / "notebooks"
    if nb_dir.is_dir():
        nbs = sorted(nb_dir.glob("*.ipynb"))
        if len(nbs) < EXPECTED_NOTEBOOKS_MIN:
            r.warn(section,
                   f"Only {len(nbs)} notebooks found (expected >= "
                   f"{EXPECTED_NOTEBOOKS_MIN})")
        else:
            r.ok(section, f"Found {len(nbs)} notebooks")

    # Phase summaries
    found_summaries = {p.name for p in REPO_ROOT.glob("phase*_summary.md")}
    missing = EXPECTED_PHASE_SUMMARIES - found_summaries
    if missing:
        r.warn(section,
               f"Missing phase summary file(s): {', '.join(sorted(missing))}")
    if "phase8_summary.md" not in found_summaries:
        r.info(section, "phase8_summary.md not yet written (optional)")


# ========================================================================
# Check #2 — README integrity
# ========================================================================

RE_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
RE_MD_IMG  = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
RE_HEADER  = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def slugify_github(title: str) -> str:
    """Approximate GitHub's anchor-slug algorithm."""
    s = title.lower()
    # strip emojis and non-word except spaces and hyphens
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s.strip())
    return s


def check_readme(r: Report):
    section = "2. README"
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        r.critical(section, "README.md not found — stopping this section.")
        return

    text = readme.read_text(encoding="utf-8", errors="replace")

    # --- Size sanity ---
    n_chars = len(text)
    n_lines = text.count("\n") + 1
    r.info(section, f"README size: {n_chars} chars / {n_lines} lines")
    if n_chars < 2000:
        r.warn(section, "README very short — may lack depth for recruiters")

    # --- Badges ---
    badges = re.findall(r"!\[[^\]]*\]\(https://img\.shields\.io/[^)]+\)", text)
    if len(badges) < 4:
        r.warn(section, f"Only {len(badges)} shields.io badges detected "
                        "(P2 reference has 8)")
    else:
        r.ok(section, f"{len(badges)} badges detected")

    # --- Placeholders ---
    for pat, label in PLACEHOLDER_PATTERNS:
        hits = re.findall(pat, text, flags=re.IGNORECASE)
        if hits:
            r.critical(section,
                       f"Placeholder detected ({label}): "
                       f"{len(hits)} occurrence(s)",
                       f"Pattern: {pat}")

    # --- Headers and TOC ---
    headers = RE_HEADER.findall(text)
    header_slugs = {slugify_github(h[1]) for h in headers}

    toc_block_match = re.search(
        r"##\s*.*Table of Contents.*?(?=\n## |\Z)",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    if not toc_block_match:
        r.warn(section, "No '## Table of Contents' section detected")
    else:
        toc_block = toc_block_match.group(0)
        # Find all TOC anchor links
        toc_anchors = re.findall(r"\]\(#([^)]+)\)", toc_block)
        broken_anchors = [a for a in toc_anchors if a not in header_slugs]
        if broken_anchors:
            r.critical(section,
                       f"{len(broken_anchors)} TOC anchors do NOT match "
                       "any section header",
                       detail="First 3: " + ", ".join(broken_anchors[:3]))
        else:
            r.ok(section,
                 f"TOC anchor integrity: all {len(toc_anchors)} resolve")

    # --- Relative link integrity ---
    all_links = RE_MD_LINK.findall(text) + RE_MD_IMG.findall(text)
    missing_rel, missing_raw = [], []
    raw_github_urls = []
    for _label, target in all_links:
        target = target.split("#")[0].split("?")[0]  # strip anchor/query
        if not target:
            continue
        if target.startswith(("http://", "https://", "mailto:")):
            # Note raw.githubusercontent references for later check
            if "raw.githubusercontent.com" in target:
                raw_github_urls.append(target)
            continue
        # Relative path
        p = (REPO_ROOT / target).resolve()
        if not p.exists():
            missing_rel.append(target)

    if missing_rel:
        r.critical(section,
                   f"{len(missing_rel)} broken relative link(s) in README",
                   detail="; ".join(sorted(set(missing_rel))[:5]))
    else:
        r.ok(section, "All relative links resolve")

    if raw_github_urls:
        r.info(section,
               f"{len(raw_github_urls)} raw.githubusercontent URLs "
               "(will not be checked for liveness)")


# ========================================================================
# Check #3 — Cross-reference consistency (D-xxx anchors)
# ========================================================================

RE_DECISION = re.compile(r"\bD-0\d{2}\b")


def check_cross_refs(r: Report):
    section = "3. DECISION REFS"

    pd_path = REPO_ROOT / "ProjectDriven.md"
    if not pd_path.exists():
        r.critical(section, "ProjectDriven.md not found")
        return

    pd_text = pd_path.read_text(encoding="utf-8", errors="replace")
    pd_ids = set(RE_DECISION.findall(pd_text))
    r.info(section, f"ProjectDriven.md defines {len(pd_ids)} unique D-xxx IDs")

    scan_files = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "findings.md",
        REPO_ROOT / "docs" / "methodology.md",
    ]

    for f in scan_files:
        if not f.exists():
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        refs = set(RE_DECISION.findall(txt))
        missing = refs - pd_ids
        if missing:
            r.critical(section,
                       f"{f.name} references D-IDs not in ProjectDriven.md: "
                       f"{', '.join(sorted(missing))}")
        else:
            r.ok(section, f"{f.name}: all {len(refs)} D-refs resolve")

    # Highest D-ID
    if pd_ids:
        nums = [int(d.split("-")[1]) for d in pd_ids]
        max_d = max(nums)
        r.info(section, f"Highest D-ID in ProjectDriven.md: D-{max_d:03d}")


# ========================================================================
# Check #4 — Git hygiene (secrets, large files, .gitignore)
# ========================================================================

def iter_tracked_files() -> Iterable[Path]:
    """Iterate over all files except common VCS/venv dirs."""
    skip = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv",
            "node_modules", ".mypy_cache", ".pytest_cache"}
    for p in REPO_ROOT.rglob("*"):
        if p.is_dir():
            continue
        parts = set(p.relative_to(REPO_ROOT).parts)
        if parts & skip:
            continue
        yield p


def check_git_hygiene(r: Report):
    section = "4. GIT HYGIENE"

    # --- .gitignore basics ---
    gi_path = REPO_ROOT / ".gitignore"
    if gi_path.exists():
        gi = gi_path.read_text(encoding="utf-8", errors="replace")
        needed = ["__pycache__", ".ipynb_checkpoints", ".env"]
        missing = [n for n in needed if n not in gi]
        if missing:
            r.warn(section,
                   f".gitignore missing entries: {', '.join(missing)}")
        else:
            r.ok(section, ".gitignore covers core patterns")
    else:
        r.critical(section, ".gitignore not found")

    # --- Secrets scan ---
    secret_hits = []
    for p in iter_tracked_files():
        if p.suffix.lower() in BINARY_EXTS:
            continue
        if p.suffix.lower() not in SCAN_EXTS and p.name != ".env":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat, label in SECRET_PATTERNS:
            if re.search(pat, text):
                rel = p.relative_to(REPO_ROOT)
                secret_hits.append((str(rel), label))

    if secret_hits:
        r.critical(section, f"{len(secret_hits)} potential secret(s) detected",
                   detail=" | ".join(
                       f"{f}: {lbl}" for f, lbl in secret_hits[:3]))
    else:
        r.ok(section, "No secrets detected by pattern scan")

    # --- Large files ---
    large = []
    warn_list = []
    for p in iter_tracked_files():
        size_mb = p.stat().st_size / (1024 * 1024)
        rel = str(p.relative_to(REPO_ROOT))
        if size_mb >= LARGE_FILE_THRESHOLD_MB:
            large.append((rel, size_mb))
        elif size_mb >= WARN_FILE_THRESHOLD_MB:
            warn_list.append((rel, size_mb))
    if large:
        r.critical(section,
                   f"{len(large)} file(s) >= {LARGE_FILE_THRESHOLD_MB} MB",
                   detail=" | ".join(
                       f"{f} ({s:.1f} MB)" for f, s in large[:3]))
    if warn_list:
        r.warn(section,
               f"{len(warn_list)} file(s) >= {WARN_FILE_THRESHOLD_MB} MB "
               "(consider Git-LFS or .gitignore)",
               detail=" | ".join(
                   f"{f} ({s:.1f} MB)" for f, s in warn_list[:3]))

    # --- .env committed? ---
    if (REPO_ROOT / ".env").exists():
        r.critical(section,
                   "'.env' file exists in repo — must be gitignored")


# ========================================================================
# Check #5 — Notebook output state
# ========================================================================

def check_notebooks(r: Report, allow_outputs: bool):
    section = "5. NOTEBOOKS"
    nb_dir = REPO_ROOT / "notebooks"
    if not nb_dir.is_dir():
        return
    nbs = sorted(nb_dir.glob("*.ipynb"))
    if not nbs:
        r.warn(section, "No notebooks found")
        return

    with_outputs = []
    cell_counts = []
    for nb_path in nbs:
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except Exception as e:
            r.warn(section, f"Could not parse {nb_path.name}: {e}")
            continue
        cells = nb.get("cells", [])
        cell_counts.append(len(cells))
        has_output = any(
            (c.get("cell_type") == "code" and c.get("outputs"))
            for c in cells
        )
        if has_output:
            with_outputs.append(nb_path.name)

    if cell_counts:
        mean_cells = sum(cell_counts) / len(cell_counts)
        r.info(section, f"Mean cells per notebook: {mean_cells:.1f}")

    if with_outputs:
        msg = (f"{len(with_outputs)} notebook(s) have output cells saved")
        # Portfolio reviewers can read outputs without running — this is
        # generally a GOOD thing for portfolio display.
        if allow_outputs:
            r.ok(section, msg + " (--notebook-output-ok passed)")
        else:
            r.info(section, msg +
                   " — portfolio-friendly; pass --notebook-output-ok to silence")
    else:
        r.info(section, "All notebooks have cleared outputs "
                        "(reviewers must run to see results)")


# ========================================================================
# Check #6 — Portability
# ========================================================================

def check_portability(r: Report):
    section = "6. PORTABILITY"

    # requirements.txt pinning
    req = REPO_ROOT / "requirements.txt"
    if req.exists():
        req_text = req.read_text(encoding="utf-8", errors="replace")
        lines = [l.strip() for l in req_text.splitlines()
                 if l.strip() and not l.startswith("#")]
        pinned = sum(1 for l in lines if any(op in l for op in
                                             ("==", ">=", "~=", "<", ">")))
        if lines:
            ratio = pinned / len(lines)
            if ratio < 0.5:
                r.warn(section,
                       f"requirements.txt has {pinned}/{len(lines)} pinned "
                       "(< 50% — reproducibility risk)")
            else:
                r.ok(section,
                     f"requirements.txt pinning: {pinned}/{len(lines)}")

    # .env.example
    env_example = REPO_ROOT / ".env.example"
    if env_example.exists():
        r.ok(section, ".env.example present (guides secret setup)")
    else:
        r.warn(section, ".env.example missing (reviewers won't know what "
                        "env vars are required)")

    # random_state fixed?
    rs_hits = 0
    files_scanned = 0
    for p in (list((REPO_ROOT / "notebooks").glob("*.ipynb"))
              + list((REPO_ROOT / "src").rglob("*.py"))
              + list((REPO_ROOT / "scripts").rglob("*.py"))):
        files_scanned += 1
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if re.search(r"random_state\s*=\s*\d+", txt):
            rs_hits += 1
    if files_scanned > 0 and rs_hits == 0:
        r.warn(section,
               "No fixed random_state= detected across notebooks / src / scripts")
    elif rs_hits > 0:
        r.ok(section,
             f"random_state= found in {rs_hits}/{files_scanned} files")


# ========================================================================
# Rendering
# ========================================================================

def render(report: Report, quiet: bool):
    # Group by section
    sections: dict[str, list[Issue]] = {}
    for i in report.issues:
        sections.setdefault(i.section, []).append(i)

    if not quiet:
        print()
        print("=" * 66)
        print(" PORTFOLIO PROJECT 3 — QUALITY AUDIT")
        print(f" Repo root: {REPO_ROOT}")
        print("=" * 66)
        for sec_name, items in sections.items():
            print()
            print(f"-- {sec_name} --")
            for it in items:
                marker = MARKERS[it.severity]
                print(f"  {marker} {it.msg}")
                if it.detail:
                    print(f"         -> {it.detail}")

    # Score
    c = report.counts()
    total = sum(c.values())
    # Scoring: OK = +2, INFO = 0, WARN = -3, CRITICAL = -8
    raw = c[SEVERITY_OK] * 2 - c[SEVERITY_WARN] * 3 - c[SEVERITY_CRITICAL] * 8
    max_raw = total * 2
    # Scale to 0..100
    if max_raw > 0:
        score = max(0, min(100, int(100 * (raw + max_raw) / (2 * max_raw))))
    else:
        score = 0

    print()
    print("=" * 66)
    print(" SUMMARY")
    print("=" * 66)
    print(f"  OK       : {c[SEVERITY_OK]:>4}")
    print(f"  INFO     : {c[SEVERITY_INFO]:>4}")
    print(f"  WARN     : {c[SEVERITY_WARN]:>4}")
    print(f"  CRITICAL : {c[SEVERITY_CRITICAL]:>4}")
    print()
    if c[SEVERITY_CRITICAL] == 0 and c[SEVERITY_WARN] == 0:
        verdict = "READY — portfolio-grade"
    elif c[SEVERITY_CRITICAL] == 0:
        verdict = "NEAR-READY — fix warnings when possible"
    elif c[SEVERITY_CRITICAL] <= 2:
        verdict = "MINOR GAPS — address critical issues"
    else:
        verdict = "NOT READY — several critical issues"
    print(f"  SCORE    : {score}/100 — {verdict}")
    print("=" * 66)

    return c[SEVERITY_CRITICAL]


# ========================================================================
# Main
# ========================================================================

def main():
    # Windows console: force UTF-8 output so non-ASCII detail strings
    # (e.g. Japanese comments in requirements.txt, paths with special
    # characters) don't crash the print() calls under cp932 default.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # Older Python or non-standard streams — non-fatal

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook-output-ok", action="store_true",
                    help="treat populated notebook outputs as OK")
    ap.add_argument("--quiet", action="store_true",
                    help="summary only, no per-check lines")
    args = ap.parse_args()

    r = Report()
    check_structure(r)
    check_readme(r)
    check_cross_refs(r)
    check_git_hygiene(r)
    check_notebooks(r, allow_outputs=args.notebook_output_ok)
    check_portability(r)

    n_critical = render(r, quiet=args.quiet)
    sys.exit(1 if n_critical > 0 else 0)


if __name__ == "__main__":
    main()
