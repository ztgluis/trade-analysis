#!/usr/bin/env python3
"""
pine_lint.py  —  Pine Script v6 syntax linter

Catches common mistakes before pasting into TradingView:

  ✖  error   [invalid-param]   tooltip= used in plot/fill/table/etc. — only valid for input.*
  ✖  error   [invalid-func]    function that does not exist in Pine Script v6 (e.g. ta.adx())
  ⚠  warning [deprecated]      v5 bare names missing ta./ str./ math./ request. prefix
  ⚠  warning [no-version]      missing //@version=N declaration at top of file
  ⚠  warning [trailing-comma]  trailing comma before ) — common typo
  ⚠  warning [empty-string-kw] keyword= with empty string value — often a forgotten value

Usage:
  python lint_pine.py  script.pine              # lint one file
  python lint_pine.py  *.pine                   # lint all scripts
  python lint_pine.py  --fix  script.pine       # auto-fix known-safe issues in-place
  python lint_pine.py  --strict  script.pine    # warnings are errors (for CI use)

Exit codes:  0 = clean   1 = errors found
"""

from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Issue:
    level:   str   # "error" | "warning"
    line:    int
    rule:    str
    message: str

    def __str__(self) -> str:
        icon = "✖" if self.level == "error" else "⚠"
        lvl  = f"{self.level.upper():7s}"
        return f"  {icon}  {lvl}  line {self.line:4d}  [{self.rule}]  {self.message}"


# ─────────────────────────────────────────────────────────────────────────────
# Pine Script v6 knowledge base
# ─────────────────────────────────────────────────────────────────────────────

# Functions that DO accept tooltip=
_TOOLTIP_VALID_PREFIXES = ("input.",)

# Functions that do NOT accept tooltip=
_TOOLTIP_INVALID_FUNCS: list[str] = [
    # Plot functions
    "plot", "plotshape", "plotchar", "plotarrow", "plotcandle", "plotbar",
    # Chart aesthetics
    "fill", "bgcolor", "hline", "barcolor",
    # Table / drawing
    "table.new", "table.cell", "table.merge_cells",
    "label.new", "line.new", "box.new", "polyline.new",
    # Strategy
    "strategy", "strategy.entry", "strategy.exit", "strategy.close",
    "strategy.order", "strategy.cancel", "strategy.cancel_all",
    # Alerts
    "alert", "alertcondition",
    # Data
    "request.security", "request.security_lower_tf",
]

# Deprecated v5 function names that should use a namespace prefix.
# Format: (regex_pattern, old_name, suggested_replacement)
_DEPRECATED: list[tuple[str, str, str]] = [
    # Technical analysis
    (r"(?<![.\w])security\s*\(",    "security()",   "request.security()"),
    (r"(?<![.\w])crossover\s*\(",   "crossover()",  "ta.crossover()"),
    (r"(?<![.\w])crossunder\s*\(",  "crossunder()", "ta.crossunder()"),
    (r"(?<![.\w])highest\s*\(",     "highest()",    "ta.highest()"),
    (r"(?<![.\w])lowest\s*\(",      "lowest()",     "ta.lowest()"),
    (r"(?<![.\w])barssince\s*\(",   "barssince()",  "ta.barssince()"),
    (r"(?<![.\w])valuewhen\s*\(",   "valuewhen()",  "ta.valuewhen()"),
    (r"(?<![.\w])pivothigh\s*\(",   "pivothigh()",  "ta.pivothigh()"),
    (r"(?<![.\w])pivotlow\s*\(",    "pivotlow()",   "ta.pivotlow()"),
    (r"(?<![.\w])stdev\s*\(",       "stdev()",      "ta.stdev()"),
    (r"(?<![.\w])wma\s*\(",         "wma()",        "ta.wma()"),
    (r"(?<![.\w])vwma\s*\(",        "vwma()",       "ta.vwma()"),
    (r"(?<![.\w])bb\s*\(",          "bb()",         "ta.bb()"),
    (r"(?<![.\w])bbw\s*\(",         "bbw()",        "ta.bbw()"),
    (r"(?<![.\w])cci\s*\(",         "cci()",        "ta.cci()"),
    (r"(?<![.\w])mom\s*\(",         "mom()",        "ta.mom()"),
    (r"(?<![.\w])change\s*\(",      "change()",     "ta.change()"),
    (r"(?<![.\w])cum\s*\(",         "cum()",        "ta.cum()"),
    # Math — bare names (pine v5 had these as globals, v6 needs math. prefix)
    (r"(?<![.\w])na\s*\(\s*\)",     "na()",         "na  (use as a literal, not a function call)"),
    (r"(?<![.\w])round\s*\(",       "round()",      "math.round()"),
    (r"(?<![.\w])sqrt\s*\(",        "sqrt()",       "math.sqrt()"),
    (r"(?<![.\w])pow\s*\(",         "pow()",        "math.pow()"),
    (r"(?<![.\w])floor\s*\(",       "floor()",      "math.floor()"),
    (r"(?<![.\w])ceil\s*\(",        "ceil()",       "math.ceil()"),
    (r"(?<![.\w])sign\s*\(",        "sign()",       "math.sign()"),
    (r"(?<![.\w])log\s*\(",         "log()",        "math.log()"),
    # String
    (r"(?<![.\w])tostring\s*\(",    "tostring()",   "str.tostring()"),
    (r"(?<![.\w])tonumber\s*\(",    "tonumber()",   "str.tonumber()"),
    (r"(?<![.\w])contains\s*\(",    "contains()",   "str.contains()"),
    (r"(?<![.\w])startswith\s*\(",  "startswith()", "str.startswith()"),
    (r"(?<![.\w])endswith\s*\(",    "endswith()",   "str.endswith()"),
    (r"(?<![.\w])replace\s*\(",     "replace()",    "str.replace()"),
    (r"(?<![.\w])lower\s*\(",       "lower()",      "str.lower()"),
    (r"(?<![.\w])upper\s*\(",       "upper()",      "str.upper()"),
    (r"(?<![.\w])length\s*\(",      "length()",     "str.length()"),
    (r"(?<![.\w])substring\s*\(",   "substring()",  "str.substring()"),
]


# Functions that do NOT exist in Pine Script v6 — will cause a compile error.
# Format: (regex_pattern, invalid_name, correct_replacement)
_INVALID_FUNCS: list[tuple[str, str, str]] = [
    # ta.adx() was never a Pine Script function.
    # Use ta.dmi(diLength, adxSmoothing) → [plusDI, minusDI, adx]
    (
        r"(?<![.\w])ta\.adx\s*\(",
        "ta.adx()",
        "ta.dmi(diLength, adxSmoothing)  →  [plus_di, minus_di, adx]",
    ),
    # ta.pivots() does not exist — use ta.pivothigh() / ta.pivotlow()
    (
        r"(?<![.\w])ta\.pivots\s*\(",
        "ta.pivots()",
        "ta.pivothigh() / ta.pivotlow()",
    ),
    # math.abs() does not exist in Pine — use math.abs() … actually it does.
    # math.log10() does not exist — use math.log() (natural log)
    (
        r"(?<![.\w])math\.log10\s*\(",
        "math.log10()",
        "math.log() (returns natural log; divide by math.log(10) for log10)",
    ),
    # array.size() was renamed to array.size() … but array.length() never existed
    (
        r"(?<![.\w])array\.length\s*\(",
        "array.length()",
        "array.size()",
    ),
    # str.format_time() does not exist — use str.format() with time formatting
    (
        r"(?<![.\w])str\.format_time\s*\(",
        "str.format_time()",
        "str.format() with a time/date format string",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Source masking
# Replace string-literal contents and // comment bodies with spaces.
# Preserves all character positions so line numbers stay correct.
# ─────────────────────────────────────────────────────────────────────────────

_MASK_CHAR = '\x00'   # sentinel used to replace string/comment content
                      # Must NOT be a space/tab so trailing-comma rule
                      # doesn't get false-positives from masked format strings
                      # like str.tostring(rsi, "#.0")


def _mask(source: str) -> str:
    """
    Replace string-literal contents and // comment bodies with _MASK_CHAR.
    Preserves all character positions (no insertions/deletions).
    Newlines are always kept so line-number lookups stay correct.
    Quote characters themselves are also replaced, so the masked source
    no longer contains any string delimiters — safe for simple regex rules.
    """
    M   = _MASK_CHAR
    buf = list(source)
    n   = len(buf)
    i   = 0
    while i < n:
        # Line comment: // → mask to end of line (preserve the newline)
        if buf[i] == '/' and i + 1 < n and buf[i + 1] == '/':
            i += 2
            while i < n and buf[i] != '\n':
                buf[i] = M
                i += 1
        # Double-quoted string
        elif buf[i] == '"':
            buf[i] = M
            i += 1
            while i < n and buf[i] != '"' and buf[i] != '\n':
                if buf[i] == '\\' and i + 1 < n:
                    buf[i] = M; i += 1
                buf[i] = M
                i += 1
            if i < n and buf[i] == '"':
                buf[i] = M
                i += 1
        # Single-quoted string
        elif buf[i] == "'":
            buf[i] = M
            i += 1
            while i < n and buf[i] != "'" and buf[i] != '\n':
                if buf[i] == '\\' and i + 1 < n:
                    buf[i] = M; i += 1
                buf[i] = M
                i += 1
            if i < n and buf[i] == "'":
                buf[i] = M
                i += 1
        else:
            i += 1
    return ''.join(buf)


def _line_of(source: str, pos: int) -> int:
    """1-based line number for a character position in source."""
    return source[:pos].count('\n') + 1


def _args_span(masked: str, paren_pos: int) -> tuple[int, int]:
    """
    Given position of '(' in masked source, return (start, end) of the
    argument text, i.e. masked[start:end] is everything inside the parens.
    """
    depth = 0
    i     = paren_pos
    while i < len(masked):
        if   masked[i] == '(': depth += 1
        elif masked[i] == ')':
            depth -= 1
            if depth == 0:
                return paren_pos + 1, i
        i += 1
    return paren_pos + 1, len(masked)  # unterminated — shouldn't happen in valid Pine


# ─────────────────────────────────────────────────────────────────────────────
# Lint rules
# ─────────────────────────────────────────────────────────────────────────────

def _rule_no_version(source: str) -> list[Issue]:
    if not re.search(r'//@version\s*=\s*\d', source):
        return [Issue("warning", 1, "no-version",
                      "Missing //@version=N declaration — add //@version=6 at the top")]
    return []


def _rule_invalid_params(source: str, masked: str) -> list[Issue]:
    """tooltip= is only valid inside input.* calls."""
    issues = []
    for func in _TOOLTIP_INVALID_FUNCS:
        # Escape dots (table.new → table\.new) and anchor at word boundary
        pat = re.compile(r'\b' + re.escape(func) + r'\s*\(')
        for m in pat.finditer(masked):
            paren_pos        = m.end() - 1
            a_start, a_end   = _args_span(masked, paren_pos)
            if re.search(r'\btooltip\s*=', masked[a_start:a_end]):
                ln = _line_of(source, m.start())
                issues.append(Issue(
                    "error", ln, "invalid-param",
                    f"`{func}()` does not support `tooltip=`  "
                    f"(tooltip= is only valid for input.* functions)"
                ))
    return issues


def _rule_invalid_funcs(source: str, masked: str) -> list[Issue]:
    """Functions that do not exist in Pine Script v6 — will fail to compile."""
    issues = []
    for pattern, invalid, replacement in _INVALID_FUNCS:
        for m in re.finditer(pattern, masked):
            ln = _line_of(source, m.start())
            issues.append(Issue(
                "error", ln, "invalid-func",
                f"`{invalid}` does not exist in Pine Script v6 — use `{replacement}` instead"
            ))
    return issues


def _rule_deprecated(source: str, masked: str) -> list[Issue]:
    """Deprecated v5 function names that should use a namespace prefix."""
    issues = []
    for pattern, old, new in _DEPRECATED:
        for m in re.finditer(pattern, masked):
            ln = _line_of(source, m.start())
            issues.append(Issue(
                "warning", ln, "deprecated",
                f"`{old}` is a Pine Script v5 name — use `{new}` instead"
            ))
    return issues


def _rule_trailing_comma(source: str, masked: str) -> list[Issue]:
    """Trailing comma before ) is a common typo."""
    issues = []
    for m in re.finditer(r',[ \t]*\)', masked):
        ln = _line_of(source, m.start())
        issues.append(Issue("warning", ln, "trailing-comma",
                            "Trailing comma before `)` — likely a typo"))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Auto-fix
# ─────────────────────────────────────────────────────────────────────────────

def _autofix(source: str) -> str:
    """
    Remove tooltip= keyword argument from non-input function calls.
    Handles multi-line calls:
        func(...,
             tooltip="some text")    →  func(...)
        func(...,
             tooltip="some text",
             nextArg)               →  func(...,
                                            nextArg)
    """
    # Pattern covers:  ,  optional-whitespace  optional-newline  optional-whitespace
    #                  tooltip  =  <string-value>
    # The string value can be double- or single-quoted.
    pattern = re.compile(
        r',[ \t]*\n?[ \t]*tooltip\s*=\s*(?:"[^"]*"|\'[^\']*\')',
        re.DOTALL,
    )
    return pattern.sub('', source)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def lint(path: Path, fix: bool = False) -> tuple[list[Issue], bool]:
    """
    Lint a single .pine file.

    Returns:
        (issues, was_fixed)
        was_fixed is True only when fix=True and the file was actually changed.
    """
    source = path.read_text(encoding="utf-8")
    masked = _mask(source)

    issues: list[Issue] = []
    issues += _rule_no_version(source)
    issues += _rule_invalid_params(source, masked)
    issues += _rule_invalid_funcs(source, masked)
    issues += _rule_deprecated(source, masked)
    issues += _rule_trailing_comma(source, masked)
    issues.sort(key=lambda x: x.line)

    was_fixed = False
    if fix:
        fixed = _autofix(source)
        if fixed != source:
            path.write_text(fixed, encoding="utf-8")
            was_fixed = True
            # Re-lint after fix so the reported issues are accurate
            source = fixed
            masked = _mask(fixed)
            issues = []
            issues += _rule_no_version(source)
            issues += _rule_invalid_params(source, masked)
            issues += _rule_invalid_funcs(source, masked)
            issues += _rule_deprecated(source, masked)
            issues += _rule_trailing_comma(source, masked)
            issues.sort(key=lambda x: x.line)

    return issues, was_fixed


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="pine_lint",
        description="Pine Script v6 syntax linter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes:  0 = clean   1 = errors found",
    )
    parser.add_argument("files",   nargs="+", type=Path,
                        help=".pine files to lint")
    parser.add_argument("--fix",   action="store_true",
                        help="Auto-fix known-safe issues in-place")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors (non-zero exit)")
    args = parser.parse_args()

    total_errors   = 0
    total_warnings = 0

    for path in args.files:
        if not path.exists():
            print(f"pine_lint: file not found: {path}", file=sys.stderr)
            continue

        issues, was_fixed = lint(path, fix=args.fix)
        errors   = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]

        header = f"  {path.name}"
        if was_fixed:
            header += "  [auto-fixed]"

        if issues:
            print(f"\n{'─' * 64}")
            print(header)
            print(f"{'─' * 64}")
            for issue in issues:
                print(issue)
            print(f"\n  {len(errors)} error(s), {len(warnings)} warning(s)")
        else:
            print(f"  ✓  {header.strip()}  — no issues found")

        total_errors   += len(errors)
        total_warnings += len(warnings)

    print()
    if total_errors > 0 or (args.strict and total_warnings > 0):
        print(f"  ✖  FAILED — {total_errors} error(s), {total_warnings} warning(s)")
        return 1
    elif total_warnings > 0:
        print(f"  ⚠  {total_warnings} warning(s)  (use --strict to fail on warnings)")
        return 0
    else:
        print(f"  ✓  All files clean")
        return 0


if __name__ == "__main__":
    sys.exit(main())
