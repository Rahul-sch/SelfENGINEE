"""
Syntactic state machine for Python code boundary detection.

Tracks bracket depth, string state, comments, and keywords to identify
decision boundaries WITHOUT false positives from strings/comments.

Boundary offsets are returned as positions AFTER the delimiter character.

Boundaries trigger ONLY on:
1. Newline at top level (bracket_depth==0, not in string/comment)
2. Decision keyword at statement start (def/class/if/for/while/try/etc.)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum, auto


class StringState(Enum):
    """Current string literal state."""
    NONE = auto()
    SINGLE = auto()       # '...'
    DOUBLE = auto()       # "..."
    TRIPLE_SINGLE = auto()  # '''...'''
    TRIPLE_DOUBLE = auto()  # """..."""


DECISION_KEYWORDS: Set[str] = frozenset({
    'def', 'class', 'if', 'elif', 'else', 'for', 'while',
    'try', 'except', 'finally', 'with', 'match', 'case',
    'return', 'yield', 'raise', 'assert', 'pass', 'break', 'continue',
    'async', 'await', 'lambda'
})

_MAX_KEYWORD_LEN = max(len(k) for k in DECISION_KEYWORDS)
_ROLLING_BUFFER_SIZE = 32


@dataclass
class SyntacticState:
    """
    Tracks Python syntactic state across incremental text updates.
    """
    bracket_depth: int = 0
    string_state: StringState = StringState.NONE
    in_comment: bool = False
    at_line_start: bool = True
    line_indent_only: bool = True

    @property
    def in_string(self) -> bool:
        return self.string_state != StringState.NONE

    @property
    def in_triple_string(self) -> bool:
        return self.string_state in (StringState.TRIPLE_SINGLE, StringState.TRIPLE_DOUBLE)

    @property
    def at_statement_start(self) -> bool:
        return (
            self.at_line_start and
            self.line_indent_only and
            self.bracket_depth == 0 and
            not self.in_string and
            not self.in_comment
        )

    @property
    def is_top_level(self) -> bool:
        return (
            self.bracket_depth == 0 and
            not self.in_string and
            not self.in_comment
        )


@dataclass
class _PendingIdentifier:
    """Tracks a partial identifier that may span chunk boundaries."""
    chars: str = ""
    started_at_statement_start: bool = False
    start_offset_in_chunk: int = -1


class ConstraintTracker:
    """
    Tracks Python syntax to detect decision boundaries.

    Handles:
    - Bracket depth (parentheses, brackets, braces)
    - String literals (single, double, triple-quoted)
    - Escape sequences in strings
    - Comments (# to end of line)
    - Keywords at statement start (with cross-chunk detection)

    Boundary positions are returned as offsets AFTER the delimiter.

    Usage:
        tracker = ConstraintTracker()
        boundaries = tracker.update("def foo():\\n    return 1")
    """

    def __init__(self):
        self._state = SyntacticState()
        self._pending_escape: bool = False
        self._pending_id: _PendingIdentifier = _PendingIdentifier()
        self._total_chars_seen: int = 0

    def reset(self) -> None:
        """Reset all state for new generation."""
        self._state = SyntacticState()
        self._pending_escape = False
        self._pending_id = _PendingIdentifier()
        self._total_chars_seen = 0

    @property
    def state(self) -> SyntacticState:
        """Current syntactic state (read-only view)."""
        return self._state

    def update(self, text: str) -> List[int]:
        """
        Process new text and return boundary offsets within it.

        Boundary conditions (per spec):
        1. Newline at top level (bracket_depth==0, not in string/comment)
           -> offset is position AFTER the newline
        2. Decision keyword at statement start
           -> offset is position AFTER the keyword

        Args:
            text: New text chunk to process

        Returns:
            List of character offsets (0-based within `text`) representing
            positions AFTER decision delimiters. Empty if no boundaries.
        """
        if not text:
            return []

        boundaries: List[int] = []
        s = self._state
        i = 0
        n = len(text)

        # Check if pending identifier from previous chunk completes a keyword
        if self._pending_id.chars and self._pending_id.started_at_statement_start:
            keyword_boundary = self._complete_pending_identifier(text)
            if keyword_boundary is not None and keyword_boundary <= n:
                boundaries.append(keyword_boundary)

        while i < n:
            char = text[i]
            lookahead2 = text[i:i + 2]
            lookahead3 = text[i:i + 3]

            # Handle escape sequences in strings
            if self._pending_escape:
                self._pending_escape = False
                i += 1
                continue

            # Handle comment state
            if s.in_comment:
                if char == '\n':
                    s.in_comment = False
                    s.at_line_start = True
                    s.line_indent_only = True
                    self._pending_id = _PendingIdentifier()
                    if s.bracket_depth == 0:
                        boundaries.append(i + 1)
                i += 1
                continue

            # Handle string states
            if s.in_string:
                if char == '\\':
                    self._pending_escape = True
                    i += 1
                    continue

                if s.string_state == StringState.TRIPLE_SINGLE:
                    if lookahead3 == "'''":
                        s.string_state = StringState.NONE
                        i += 3
                        continue
                elif s.string_state == StringState.TRIPLE_DOUBLE:
                    if lookahead3 == '"""':
                        s.string_state = StringState.NONE
                        i += 3
                        continue
                elif s.string_state == StringState.SINGLE:
                    if char == "'":
                        s.string_state = StringState.NONE
                    elif char == '\n':
                        s.string_state = StringState.NONE
                        s.at_line_start = True
                        s.line_indent_only = True
                        self._pending_id = _PendingIdentifier()
                elif s.string_state == StringState.DOUBLE:
                    if char == '"':
                        s.string_state = StringState.NONE
                    elif char == '\n':
                        s.string_state = StringState.NONE
                        s.at_line_start = True
                        s.line_indent_only = True
                        self._pending_id = _PendingIdentifier()

                i += 1
                continue

            # Not in string or comment
            if char == '#':
                s.in_comment = True
                s.line_indent_only = False
                self._clear_pending_id()
                i += 1
                continue

            if char == '"':
                if lookahead3 == '"""':
                    s.string_state = StringState.TRIPLE_DOUBLE
                    s.line_indent_only = False
                    self._clear_pending_id()
                    i += 3
                    continue
                else:
                    s.string_state = StringState.DOUBLE
                    s.line_indent_only = False
                    self._clear_pending_id()
                    i += 1
                    continue

            if char == "'":
                if lookahead3 == "'''":
                    s.string_state = StringState.TRIPLE_SINGLE
                    s.line_indent_only = False
                    self._clear_pending_id()
                    i += 3
                    continue
                else:
                    s.string_state = StringState.SINGLE
                    s.line_indent_only = False
                    self._clear_pending_id()
                    i += 1
                    continue

            # Track brackets
            if char in '([{':
                s.bracket_depth += 1
                s.line_indent_only = False
                self._clear_pending_id()
                i += 1
                continue
            elif char in ')]}':
                s.bracket_depth = max(0, s.bracket_depth - 1)
                s.line_indent_only = False
                self._clear_pending_id()
                i += 1
                continue

            # Track newlines
            if char == '\n':
                if s.bracket_depth == 0:
                    boundaries.append(i + 1)
                s.at_line_start = True
                s.line_indent_only = True
                self._pending_id = _PendingIdentifier()
                i += 1
                continue

            # Track whitespace vs content
            if char in ' \t':
                i += 1
                continue

            # Non-whitespace, non-bracket, non-string-start character
            # Check if we're continuing an existing identifier
            if self._pending_id.chars and self._is_id_continue(char):
                self._pending_id.chars += char
                i += 1
                # Check if identifier is complete (next char not id continue)
                if i >= n or not self._is_id_continue(text[i]):
                    keyword_end = self._finalize_identifier(i)
                    if keyword_end is not None:
                        boundaries.append(keyword_end)
                continue

            # Check if we're starting a new identifier (must start with letter or _)
            if self._is_id_start(char):
                self._pending_id = _PendingIdentifier(
                    chars=char,
                    started_at_statement_start=s.at_statement_start,
                    start_offset_in_chunk=i
                )
                i += 1
                # Check if identifier is complete (next char not id continue)
                if i >= n or not self._is_id_continue(text[i]):
                    keyword_end = self._finalize_identifier(i)
                    if keyword_end is not None:
                        boundaries.append(keyword_end)
                continue

            # Other characters (operators, punctuation except brackets)
            s.line_indent_only = False
            s.at_line_start = False
            self._clear_pending_id()
            i += 1

        self._total_chars_seen += n
        return boundaries

    def _is_id_start(self, c: str) -> bool:
        """Check if character can start an identifier (letter or underscore)."""
        return c.isalpha() or c == '_'

    def _is_id_continue(self, c: str) -> bool:
        """Check if character can continue an identifier (letter, digit, or underscore)."""
        return c.isalnum() or c == '_'

    def _clear_pending_id(self) -> None:
        """Clear pending identifier state."""
        self._pending_id = _PendingIdentifier()

    def _finalize_identifier(self, end_offset: int) -> Optional[int]:
        """
        Check if completed identifier is a decision keyword at statement start.
        Returns boundary offset (after keyword) if so, else None.
        """
        pid = self._pending_id
        if not pid.chars or not pid.started_at_statement_start:
            self._state.line_indent_only = False
            self._state.at_line_start = False
            self._clear_pending_id()
            return None

        word = pid.chars
        self._state.line_indent_only = False
        self._state.at_line_start = False
        self._clear_pending_id()

        if word in DECISION_KEYWORDS:
            return end_offset
        return None

    def _complete_pending_identifier(self, text: str) -> Optional[int]:
        """
        Complete a pending identifier that started in the previous chunk.
        Returns boundary offset in new chunk if keyword found at statement start.
        """
        pid = self._pending_id
        if not pid.chars or not pid.started_at_statement_start:
            return None

        # Extend identifier with chars from new chunk
        i = 0
        n = len(text)
        while i < n and self._is_id_continue(text[i]):
            pid.chars += text[i]
            i += 1

        word = pid.chars
        self._state.line_indent_only = False
        self._state.at_line_start = False
        self._clear_pending_id()

        if word in DECISION_KEYWORDS:
            return i  # Offset after keyword end in new chunk
        return None

    def is_boundary_now(self) -> bool:
        """
        Check if current position (after all updates) is at a boundary.
        """
        s = self._state
        return s.is_top_level and s.at_line_start

    def get_state_summary(self) -> dict:
        """Return current state for debugging."""
        s = self._state
        return {
            'bracket_depth': s.bracket_depth,
            'string_state': s.string_state.name,
            'in_comment': s.in_comment,
            'at_line_start': s.at_line_start,
            'at_statement_start': s.at_statement_start,
            'pending_id': self._pending_id.chars,
        }
