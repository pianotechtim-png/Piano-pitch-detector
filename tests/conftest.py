"""Shared fixtures for the test suite."""
import pytest

import app as app_module


@pytest.fixture
def client():
    """Flask test client with exceptions propagated as JSON responses."""
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c
