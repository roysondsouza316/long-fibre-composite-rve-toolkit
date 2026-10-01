"""Stacking sequences in the usual laminate notation.

Plies are listed from the bottom of the laminate to the top, angles in degrees measured
from the laminate x axis (the loading direction of a coupon) about the laminate normal.

Supported notation (separators ``/`` or ``,``; spaces are ignored)::

    [0/90/90/0]        explicit list (an optional trailing T, "total", changes nothing)
    [0/90]s            symmetric: the list followed by its mirror image -> 0/90/90/0
    [0/90]2s           repeat twice, then mirror -> 0/90/0/90/90/0/90/0
    [0/90]3            repeat three times
    [0/±45/90]s        ±45 is +45/-45 and ∓45 is -45/+45 (also written +-45 and -+45)
    [0_2/90]s          a subscript repeats one ply (0_2 = 0/0; 0₂ works too)
    [(0/90)_2/45]      a group in parentheses with a repeat count
    [0/90/45]s̄ or [0/90/45]sb   symmetric about the middle of the last ply (odd count)

A list of numbers is accepted as well.
"""

from __future__ import annotations

import re

from rve2d.exceptions import ConfigError

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SIGNS = (("±", (1.0, -1.0)), ("+-", (1.0, -1.0)), ("∓", (-1.0, 1.0)), ("-+", (-1.0, 1.0)))


def parse_stacking_sequence(sequence: str | list[float] | tuple[float, ...]) -> list[float]:
    """Ply angles (degrees, bottom to top) of a stacking sequence."""
    if isinstance(sequence, list | tuple):
        try:
            angles = [float(angle) for angle in sequence]
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Stacking sequence {sequence!r} must list ply angles.") from exc
    elif isinstance(sequence, str):
        angles = _parse_text(sequence)
    else:
        raise ConfigError(f"Stacking sequence {sequence!r} must be a string or a list.")
    if not angles:
        raise ConfigError(f"Stacking sequence {sequence!r} has no plies.")
    return angles


def format_stacking_sequence(angles: list[float]) -> str:
    """Explicit ``[a/b/c]`` form of a list of angles."""
    return "[" + "/".join(_format_angle(angle) for angle in angles) + "]"


def _format_angle(angle: float) -> str:
    return str(int(angle)) if float(angle).is_integer() else f"{angle:g}"


def _parse_text(text: str) -> list[float]:
    subscripted = re.sub("[₀-₉]+", lambda m: "_" + m.group().translate(_SUBSCRIPTS), text)
    compact = subscripted.replace(" ", "").replace("\u0304", "b")
    match = re.fullmatch(r"\[(.*)\]([0-9]*)([sS]?)([bB]?)([tT]?)", compact)
    if match is None:
        if compact.startswith("["):
            raise ConfigError(f"Cannot read the stacking sequence {text!r}.")
        body, repeat, symmetric, odd, _ = compact, "", "", "", ""
    else:
        body, repeat, symmetric, odd, _ = match.groups()
    parser = _Parser(body, text)
    angles = parser.items()
    if parser.position != len(body):
        raise ConfigError(
            f"Cannot read the stacking sequence {text!r} at {body[parser.position :]!r}."
        )
    angles = angles * (int(repeat) if repeat else 1)
    if odd and not symmetric:
        raise ConfigError(f"In {text!r} the bar (b) only applies to symmetric sequences.")
    if symmetric:
        mirror = angles[::-1][1:] if odd else angles[::-1]
        angles = angles + mirror
    return angles


class _Parser:
    """Recursive-descent parser for ``item ('/' item)*`` with groups and repeat counts."""

    def __init__(self, body: str, text: str) -> None:
        self.body = body
        self.text = text
        self.position = 0

    def items(self) -> list[float]:
        angles = self.item()
        while self.peek() in ("/", ","):
            self.position += 1
            angles += self.item()
        return angles

    def item(self) -> list[float]:
        if self.peek() == "(":
            self.position += 1
            angles = self.items()
            if self.peek() != ")":
                raise ConfigError(f"Missing ')' in the stacking sequence {self.text!r}.")
            self.position += 1
        else:
            angles = self.ply()
        return angles * self.count()

    def ply(self) -> list[float]:
        signs: tuple[float, ...] = (1.0,)
        for symbol, pair in _SIGNS:
            if self.body.startswith(symbol, self.position):
                signs = pair
                self.position += len(symbol)
                break
        match = re.compile(r"[+-]?[0-9]*\.?[0-9]+").match(self.body, self.position)
        if match is None:
            raise ConfigError(
                f"Expected a ply angle at {self.body[self.position :]!r} in {self.text!r}."
            )
        self.position = match.end()
        value = float(match.group())
        return [sign * value for sign in signs]

    def count(self) -> int:
        match = re.compile(r"_?\{?([0-9]+)\}?").match(self.body, self.position)
        if match is None:
            return 1
        self.position = match.end()
        count = int(match.group(1))
        if count < 1:
            raise ConfigError(f"Repeat counts must be positive in {self.text!r}.")
        return count

    def peek(self) -> str:
        return self.body[self.position] if self.position < len(self.body) else ""
