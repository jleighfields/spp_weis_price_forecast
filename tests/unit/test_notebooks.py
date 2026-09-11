"""Guard that every notebook stays visible in marimo's editor file browser.

marimo decides whether a ``.py`` file is a notebook by reading only its
**first 512 bytes** and looking for both ``import marimo`` and ``marimo.App``.
A module header long enough to push ``app = marimo.App()`` past that window
makes the notebook vanish from the editor's workspace list — silently, with no
error anywhere. It still runs fine as a script and under Modal, so nothing else
in the test suite or the deploy would catch it.

That is easy to reintroduce: any future expansion of a header comment can cross
the limit. Hence this test. If it fails, move the extra header prose *below*
the ``app = marimo.App(...)`` line rather than deleting it.
"""

import glob
import os
import re

import pytest

# marimo's rule, restated rather than imported: this is the contract the editor
# applies, and asserting it directly keeps the test working if marimo moves the
# private module that implements it. TestMatchesMarimosOwnCheck below confirms
# the two still agree.
READ_LIMIT = 512

# Deliberately stricter than marimo's own check. marimo looks for the bare
# substrings `import marimo` and `marimo.App`, which a *comment* mentioning them
# satisfies just as well as real code — so a notebook whose actual app
# declaration sits past the limit could still be listed, by accident, because
# its header happens to name them. Requiring the real `import marimo` statement
# and the real `app = marimo.App(...)` assignment inside the window asserts the
# structure we actually depend on, and implies marimo's weaker rule.
REQUIRED_MARKERS = (
    re.compile(rb'^import marimo$', re.M),
    re.compile(rb'^app = marimo\.App\(', re.M),
)

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


def _notebooks():
    """Every notebook file under notebooks/, as repo-relative paths."""
    pattern = os.path.join(_REPO_ROOT, 'notebooks', '**', '*.py')
    paths = [
        os.path.relpath(p, _REPO_ROOT)
        for p in glob.glob(pattern, recursive=True)
        if os.path.basename(p) != '__init__.py'
    ]
    return sorted(paths)


def _header(path):
    with open(os.path.join(_REPO_ROOT, path), 'rb') as f:
        return f.read(READ_LIMIT)


def test_notebooks_are_discovered():
    """The glob must actually find notebooks, or every test below is vacuous."""
    found = _notebooks()
    assert len(found) >= 8, f'expected the notebook set, found {found}'


@pytest.mark.parametrize('notebook', _notebooks())
def test_marimo_declaration_within_read_limit(notebook):
    header = _header(notebook)
    missing = [m for m in REQUIRED_MARKERS if not m.search(header)]
    if not missing:
        return

    # Locate each marker in the full file so the failure says how far over it is.
    with open(os.path.join(_REPO_ROOT, notebook), 'rb') as f:
        full = f.read()
    detail = []
    for marker in REQUIRED_MARKERS:
        found = marker.search(full)
        detail.append(
            f'{marker.pattern.decode()} at byte {found.start()}' if found else
            f'{marker.pattern.decode()} NOT PRESENT'
        )
    pytest.fail(
        f'{notebook} risks being hidden from marimo\'s file browser: its app '
        f'declaration is not in the first {READ_LIMIT} bytes '
        f'({"; ".join(detail)}). Shorten the module header and move the detail '
        f'below the `app = marimo.App(...)` line — do not delete it.'
    )


class TestMatchesMarimosOwnCheck:
    """Cross-check the restated rule against marimo's implementation.

    Skipped rather than failed if marimo relocates the private module — the
    parametrized test above is the real guard; this only catches the rule
    itself drifting (a changed read limit or marker set).
    """

    @staticmethod
    def _is_marimo_app():
        try:
            from marimo._server.files.directory_scanner import is_marimo_app
        except ImportError:
            pytest.skip('marimo moved is_marimo_app; local rule still enforced')
        return is_marimo_app

    def test_every_notebook_passes_marimos_detection(self):
        is_marimo_app = self._is_marimo_app()
        hidden = [
            nb for nb in _notebooks()
            if not is_marimo_app(os.path.join(_REPO_ROOT, nb))
        ]
        assert not hidden, f'hidden from the marimo file browser: {hidden}'

    def test_rule_still_matches_marimo(self, tmp_path):
        """A header just over the limit must be rejected by both."""
        is_marimo_app = self._is_marimo_app()
        padding = b'# ' + b'x' * (READ_LIMIT - 2) + b'\n'
        over = tmp_path / 'over_limit.py'
        over.write_bytes(padding + b'import marimo\napp = marimo.App()\n')
        assert not is_marimo_app(str(over))
        assert not all(m.search(over.read_bytes()[:READ_LIMIT]) for m in REQUIRED_MARKERS)

        under = tmp_path / 'under_limit.py'
        under.write_bytes(b'# short\nimport marimo\napp = marimo.App()\n')
        assert is_marimo_app(str(under))
        assert all(m.search(under.read_bytes()[:READ_LIMIT]) for m in REQUIRED_MARKERS)

    def test_a_header_merely_naming_the_markers_does_not_count(self, tmp_path):
        """The reason this test is stricter than marimo's own rule.

        A header that only *mentions* the markers satisfies marimo, so the
        notebook is listed even though its real declaration is out of reach.
        Our rule must reject it, or it would pass while the header quietly
        masked a genuine regression.
        """
        is_marimo_app = self._is_marimo_app()
        mention = b'# see `import marimo` and `marimo.App` in the docs\n'
        nb = tmp_path / 'mentions_only.py'
        nb.write_bytes(
            mention + b'# ' + b'x' * READ_LIMIT + b'\n'
            + b'import marimo\napp = marimo.App()\n'
        )
        # marimo is satisfied by the mention alone...
        assert is_marimo_app(str(nb))
        # ...ours is not.
        header = nb.read_bytes()[:READ_LIMIT]
        assert not all(m.search(header) for m in REQUIRED_MARKERS)
