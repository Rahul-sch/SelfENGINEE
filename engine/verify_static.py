"""
Hardened static security analysis using AST.

Detects and hard-fails on dangerous patterns including:
- eval/exec/compile/__import__
- importlib (any form)
- ctypes/signal/subprocess/shutil (any import)
- os.environ access (read or write)
- subprocess.Popen (even through aliases)
- pathlib.Path dangerous methods (write_text, unlink, etc.)
- socket/urllib/requests and network modules

Tracks import aliases to detect:
- from pathlib import Path as P; P(...).write_text(...)
- from subprocess import Popen as X; X(...)
- import shutil as s; s.rmtree(...)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple, Dict, NamedTuple
from collections import OrderedDict
from enum import Enum, auto
import ast
import hashlib


class Severity(Enum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class SecurityIssue:
    """Single security issue found."""
    severity: Severity
    code: str
    message: str
    lineno: int
    col: int


@dataclass
class StaticAnalysisResult:
    """Result of static security analysis."""
    passed: bool
    issues: List[SecurityIssue] = field(default_factory=list)
    score: float = 1.0

    def add_issue(
        self,
        severity: Severity,
        code: str,
        message: str,
        lineno: int = 0,
        col: int = 0
    ) -> None:
        self.issues.append(SecurityIssue(severity, code, message, lineno, col))
        if severity in (Severity.ERROR, Severity.CRITICAL):
            self.passed = False
            self.score = 0.0


# Dangerous built-in functions
DANGEROUS_BUILTINS: Set[str] = frozenset({
    'eval', 'exec', 'compile', '__import__',
    'breakpoint', 'globals', 'locals', 'vars',
    'delattr', 'setattr', 'getattr',
})

# Dangerous modules - ANY import is blocked
DANGEROUS_MODULES: Set[str] = frozenset({
    'ctypes', 'signal', 'subprocess', 'shutil', 'multiprocessing',
    'socket', 'urllib', 'requests', 'httplib', 'http.client',
    'ftplib', 'smtplib', 'telnetlib',
    'importlib', 'builtins', '__builtins__',
    'pickle', 'marshal', 'shelve', 'dill',
    'pty', 'tty', 'termios', 'fcntl',
    'resource', 'mmap', 'sysconfig',
})

# Dangerous Path methods (called on any Path-like object)
DANGEROUS_PATH_METHODS: Set[str] = frozenset({
    'write_text', 'write_bytes', 'unlink', 'rmdir',
    'rename', 'replace', 'chmod', 'mkdir', 'touch',
})

# Dangerous os module attributes/methods
DANGEROUS_OS_ATTRS: Set[str] = frozenset({
    'system', 'popen', 'spawn', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
    'spawnv', 'spawnve', 'spawnvp', 'spawnvpe',
    'execl', 'execle', 'execlp', 'execlpe', 'execv', 'execve', 'execvp', 'execvpe',
    'environ', 'putenv', 'unsetenv', 'getenv',
    'remove', 'unlink', 'rmdir', 'removedirs',
    'rename', 'renames', 'replace',
    'chmod', 'chown', 'chroot', 'fchmod', 'fchown',
    'kill', 'killpg', 'fork', 'forkpty',
})

# Write modes for open()
WRITE_MODES: Set[str] = frozenset({'w', 'a', 'x', '+', 'wb', 'ab', 'xb', 'w+', 'a+', 'r+'})


class SecurityVisitor(ast.NodeVisitor):
    """AST visitor that flags security issues with comprehensive alias tracking."""

    def __init__(self):
        self.result = StaticAnalysisResult(passed=True)
        # Track module imports: alias -> full module name
        self._module_aliases: Dict[str, str] = {}
        # Track class/function imports: alias -> (module, name)
        self._name_aliases: Dict[str, Tuple[str, str]] = {}
        # Track which dangerous modules are imported
        self._imported_dangerous: Set[str] = set()

    def _add_critical(self, node: ast.AST, code: str, msg: str) -> None:
        self.result.add_issue(
            Severity.CRITICAL, code, msg,
            getattr(node, 'lineno', 0),
            getattr(node, 'col_offset', 0)
        )

    def visit_Import(self, node: ast.Import) -> None:
        """Handle: import X, import X as Y"""
        for alias in node.names:
            full_name = alias.name
            local_name = alias.asname or alias.name
            root_module = full_name.split('.')[0]

            # Track the alias
            self._module_aliases[local_name] = full_name

            # Check for dangerous modules
            if root_module in DANGEROUS_MODULES:
                self._imported_dangerous.add(root_module)
                self._add_critical(node, 'S001', f"Dangerous import: {full_name}")

            # Specific check for importlib
            if full_name == 'importlib' or full_name.startswith('importlib.'):
                self._add_critical(node, 'S002', f"importlib import forbidden: {full_name}")

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle: from X import Y, from X import Y as Z"""
        if not node.module:
            self.generic_visit(node)
            return

        root_module = node.module.split('.')[0]

        # Check for dangerous module
        if root_module in DANGEROUS_MODULES:
            self._imported_dangerous.add(root_module)
            self._add_critical(node, 'S001', f"Dangerous import from: {node.module}")

        # Specific check for importlib
        if node.module == 'importlib' or node.module.startswith('importlib.'):
            self._add_critical(node, 'S002', f"importlib import forbidden: {node.module}")

        # Track aliases for each imported name
        for alias in node.names:
            imported_name = alias.name
            local_name = alias.asname or alias.name

            # Track: local_name -> (module, imported_name)
            self._name_aliases[local_name] = (node.module, imported_name)

            # Check specific dangerous imports
            if imported_name == 'import_module':
                self._add_critical(node, 'S002', "import_module forbidden")

            # Track Path imports for method detection
            if node.module == 'pathlib' and imported_name in ('Path', 'PurePath', 'PosixPath', 'WindowsPath'):
                pass  # Will be caught by call visitor

            # Track subprocess.Popen etc
            if root_module == 'subprocess':
                self._add_critical(node, 'S006', f"subprocess import: {imported_name}")

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function/method calls for dangerous patterns."""
        # Check direct built-in calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Check dangerous built-ins
            if func_name in DANGEROUS_BUILTINS:
                self._add_critical(node, 'S003', f"Dangerous built-in: {func_name}()")

            # Check if it's an aliased dangerous function
            if func_name in self._name_aliases:
                module, name = self._name_aliases[func_name]
                root = module.split('.')[0]
                if root == 'subprocess':
                    self._add_critical(node, 'S006', f"subprocess call via alias: {func_name}()")
                if module == 'pathlib' and name in ('Path', 'PurePath', 'PosixPath', 'WindowsPath'):
                    # Path constructor - check if followed by dangerous method
                    pass  # Will be caught if chained

            # Check open() with write modes
            if func_name == 'open':
                mode = self._get_open_mode(node)
                if mode and any(m in mode for m in WRITE_MODES):
                    self._add_critical(node, 'S007', f"open() with write mode: '{mode}'")

        # Check attribute calls: obj.method()
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            obj = node.func.value

            # Get the resolved object name
            obj_info = self._resolve_object(obj)

            # Check os.* calls
            if obj_info == 'os' or self._module_aliases.get(obj_info) == 'os':
                if attr_name in DANGEROUS_OS_ATTRS:
                    self._add_critical(node, 'S004', f"Dangerous os call: os.{attr_name}()")

            # Check subprocess.* calls
            if obj_info == 'subprocess' or self._module_aliases.get(obj_info, '').startswith('subprocess'):
                self._add_critical(node, 'S006', f"subprocess call: {attr_name}()")

            # Check shutil.* calls
            if obj_info == 'shutil' or self._module_aliases.get(obj_info, '').startswith('shutil'):
                self._add_critical(node, 'S010', f"shutil call: {attr_name}()")

            # Check Path method calls
            if attr_name in DANGEROUS_PATH_METHODS:
                if self._could_be_path(obj):
                    self._add_critical(node, 'S005', f"Dangerous Path method: {attr_name}()")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access for dangerous patterns."""
        obj_info = self._resolve_object(node.value)

        # Check os.environ access
        if (obj_info == 'os' or self._module_aliases.get(obj_info) == 'os') and node.attr == 'environ':
            self._add_critical(node, 'S008', "os.environ access forbidden")

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Check subscript access like os.environ['KEY']."""
        if isinstance(node.value, ast.Attribute):
            obj_info = self._resolve_object(node.value.value)
            if (obj_info == 'os' or self._module_aliases.get(obj_info) == 'os') and node.value.attr == 'environ':
                self._add_critical(node, 'S008', "os.environ access forbidden")

        self.generic_visit(node)

    def _resolve_object(self, node: ast.AST) -> str:
        """Resolve an AST node to its name/alias."""
        if isinstance(node, ast.Name):
            name = node.id
            # Check if it's a module alias
            if name in self._module_aliases:
                return self._module_aliases[name]
            # Check if it's a name alias
            if name in self._name_aliases:
                module, _ = self._name_aliases[name]
                return module
            return name
        elif isinstance(node, ast.Attribute):
            base = self._resolve_object(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ''

    def _could_be_path(self, node: ast.AST) -> bool:
        """
        Heuristic: could this AST node be a Path object?
        Checks for Path constructors (including aliases) and any variable.
        """
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                # Direct Path() call
                if func_name in ('Path', 'PurePath', 'PosixPath', 'WindowsPath'):
                    return True
                # Aliased Path call: from pathlib import Path as P
                if func_name in self._name_aliases:
                    module, name = self._name_aliases[func_name]
                    if module == 'pathlib' and name in ('Path', 'PurePath', 'PosixPath', 'WindowsPath'):
                        return True
            elif isinstance(node.func, ast.Attribute):
                # pathlib.Path() call
                obj_info = self._resolve_object(node.func.value)
                if obj_info == 'pathlib' and node.func.attr in ('Path', 'PurePath', 'PosixPath', 'WindowsPath'):
                    return True

        # Any variable could potentially hold a Path
        if isinstance(node, ast.Name):
            return True

        # Method chain: path.parent.write_text() etc
        if isinstance(node, ast.Attribute):
            return True

        return False

    def _get_open_mode(self, node: ast.Call) -> Optional[str]:
        """Extract mode argument from open() call."""
        # Check positional args
        if len(node.args) >= 2:
            mode_arg = node.args[1]
            if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                return mode_arg.value

        # Check keyword args
        for kw in node.keywords:
            if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    return kw.value.value

        return None


def analyze_code(code: str) -> StaticAnalysisResult:
    """
    Perform static security analysis on Python code.

    Args:
        code: Python source code string

    Returns:
        StaticAnalysisResult with passed status, issues, and score
    """
    result = StaticAnalysisResult(passed=True)

    # Parse code
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result.add_issue(
            Severity.ERROR,
            'E001',
            f"Syntax error: {e.msg}",
            e.lineno or 0,
            e.offset or 0
        )
        return result

    # Run security visitor
    visitor = SecurityVisitor()
    visitor.visit(tree)

    return visitor.result


def is_code_safe(code: str) -> Tuple[bool, List[dict]]:
    """
    Convenience function: check if code is safe.

    Returns:
        (is_safe, list of issue dicts)
    """
    result = analyze_code(code)
    issues = [
        {
            'severity': i.severity.name,
            'code': i.code,
            'message': i.message,
            'lineno': i.lineno,
            'col': i.col
        }
        for i in result.issues
    ]
    return result.passed, issues


# =============================================================================
# SABS Extensions: Security-Aware Beam Search
# =============================================================================

class ParseState(Enum):
    """
    TRI-STATE parse outcome for incremental code analysis.

    PARSE_OK: AST succeeded; security issues are meaningful
    PARSE_INCOMPLETE: Syntax error at/near EOF; code is likely incomplete (no penalty)
    PARSE_INVALID: Syntax error early; code is likely unrecoverable
    """
    PARSE_OK = auto()
    PARSE_INCOMPLETE = auto()
    PARSE_INVALID = auto()


# Severity weights for computing security penalty
SEVERITY_WEIGHTS: Dict[Severity, float] = {
    Severity.CRITICAL: 10.0,   # Hard fail candidate
    Severity.ERROR:    5.0,    # Strong penalty
    Severity.WARNING:  1.0,    # Soft penalty
    Severity.INFO:     0.1,    # Negligible
}


def is_error_near_end(code: str, error: SyntaxError) -> bool:
    """
    Determine if a syntax error is near the end of code (likely incomplete)
    vs early in the code (likely unrecoverable).

    Args:
        code: The source code string
        error: The SyntaxError exception

    Returns:
        True if error is near the end (last 20% or last line)
    """
    lines = code.split('\n')
    total_lines = len(lines)
    total_chars = len(code)

    if total_lines == 0 or total_chars == 0:
        return True  # Empty code is "incomplete"

    # Check line position - if error is on last line or second-to-last
    if error.lineno and error.lineno >= total_lines - 1:
        return True

    # Check character offset (if available)
    if error.lineno and error.offset and total_chars > 0:
        # Estimate character position of error
        try:
            error_pos = sum(len(lines[i]) + 1 for i in range(min(error.lineno - 1, total_lines)))
            error_pos += (error.offset or 0)
            # Error in last 20% of code is considered "near end"
            if error_pos >= total_chars * 0.8:
                return True
        except (IndexError, TypeError):
            pass

    return False


def classify_parse_error(code: str, error: SyntaxError) -> ParseState:
    """
    Classify a syntax error as INCOMPLETE or INVALID.

    Heuristics for INCOMPLETE:
      - Error near end of code (line/offset check)
      - Message contains "unexpected EOF", "expected ':'", etc.
      - Code ends with incomplete construct

    Heuristics for INVALID:
      - Error early in code
      - Error in middle of closed construct

    Args:
        code: The source code string
        error: The SyntaxError exception

    Returns:
        ParseState.PARSE_INCOMPLETE or ParseState.PARSE_INVALID
    """
    # Check message for EOF indicators
    msg = str(error.msg).lower() if error.msg else ""
    if "eof" in msg or "unexpected end" in msg:
        return ParseState.PARSE_INCOMPLETE
    if "expected" in msg and is_error_near_end(code, error):
        return ParseState.PARSE_INCOMPLETE

    # Check if code ends with incomplete construct
    stripped = code.rstrip()
    incomplete_endings = (
        ':', '(', '[', '{', ',', '\\',
        'def', 'if', 'class', 'for', 'while', 'try', 'with', 'elif', 'else',
        'async', 'await', 'return', 'yield', 'raise', 'import', 'from',
    )
    if any(stripped.endswith(e) for e in incomplete_endings):
        return ParseState.PARSE_INCOMPLETE

    # Check position - if error is near end, likely incomplete
    if is_error_near_end(code, error):
        return ParseState.PARSE_INCOMPLETE

    # Otherwise, likely invalid (early error)
    return ParseState.PARSE_INVALID


def compute_sec_penalty(issues: List[SecurityIssue]) -> float:
    """
    Compute the total security penalty from a list of issues.

    Args:
        issues: List of SecurityIssue objects

    Returns:
        Sum of severity weights (always >= 0)
    """
    return sum(SEVERITY_WEIGHTS.get(i.severity, 0.0) for i in issues)


class IncrementalResult(NamedTuple):
    """
    Result of incremental security analysis.

    Attributes:
        parse_state: TRI-STATE parse outcome
        issues: List of SecurityIssue objects (empty if not PARSE_OK)
        sec_penalty: Computed security penalty (0.0 if not PARSE_OK)
    """
    parse_state: ParseState
    issues: List[SecurityIssue]
    sec_penalty: float


class SecurityCache:
    """
    LRU cache for security analysis results.

    Uses OrderedDict for O(1) get/put/eviction operations.
    Key: SHA256(code_prefix)[:16]  (16 hex chars)
    Value: IncrementalResult

    Args:
        max_size: Maximum number of entries (default: 1000)
    """

    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, IncrementalResult] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _hash(self, code: str) -> str:
        """Generate hash key for code prefix."""
        return hashlib.sha256(code.encode('utf-8', errors='replace')).hexdigest()[:16]

    def get(self, code: str) -> Optional[IncrementalResult]:
        """
        Get cached result for code, updating LRU order.

        Returns:
            IncrementalResult if found, None otherwise
        """
        key = self._hash(code)
        if key in self._cache:
            self._cache.move_to_end(key)  # O(1) LRU update
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, code: str, result: IncrementalResult) -> None:
        """
        Cache result for code, evicting oldest if at capacity.
        """
        key = self._hash(code)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = result  # Update value
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # O(1) evict oldest
            self._cache[key] = result

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / max(self._hits + self._misses, 1),
        }


def analyze_incremental(
    code: str,
    cache: Optional[SecurityCache] = None
) -> IncrementalResult:
    """
    Incremental-safe security analysis with TRI-STATE parsing.

    This function is designed for use during code generation where
    the code may be incomplete. It handles:
    - Empty/whitespace-only code → PARSE_INCOMPLETE
    - Syntax errors near end → PARSE_INCOMPLETE (no penalty)
    - Syntax errors early → PARSE_INVALID (small fixed penalty)
    - Successful parse → PARSE_OK (run full security analysis)

    Args:
        code: Python source code string (may be incomplete)
        cache: Optional SecurityCache for LRU caching

    Returns:
        IncrementalResult with parse_state, issues, and sec_penalty
    """
    # Handle empty code
    if not code or not code.strip():
        return IncrementalResult(ParseState.PARSE_INCOMPLETE, [], 0.0)

    # Check cache first
    if cache:
        cached = cache.get(code)
        if cached is not None:
            return cached

    # Try to parse
    try:
        tree = ast.parse(code)
        parse_state = ParseState.PARSE_OK
    except SyntaxError as e:
        # Classify the syntax error
        parse_state = classify_parse_error(code, e)

        if parse_state == ParseState.PARSE_INCOMPLETE:
            # Incomplete code - no penalty
            result = IncrementalResult(ParseState.PARSE_INCOMPLETE, [], 0.0)
        else:
            # PARSE_INVALID - small fixed penalty for early errors
            result = IncrementalResult(ParseState.PARSE_INVALID, [], 2.0)

        if cache:
            cache.put(code, result)
        return result

    # Parse succeeded - run security analysis
    visitor = SecurityVisitor()
    visitor.visit(tree)
    issues = visitor.result.issues
    penalty = compute_sec_penalty(issues)

    result = IncrementalResult(ParseState.PARSE_OK, issues, penalty)
    if cache:
        cache.put(code, result)
    return result
