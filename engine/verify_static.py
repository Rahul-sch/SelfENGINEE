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
from typing import List, Set, Optional, Tuple, Dict
from enum import Enum, auto
import ast


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
