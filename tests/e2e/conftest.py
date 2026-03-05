"""Shared fixtures for E2E tests."""

from shiny.pytest import create_app_fixture
from pathlib import Path

app = create_app_fixture(Path(__file__).parent / "app_for_test.py", scope="module")
