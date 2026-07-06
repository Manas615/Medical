# This file has been refactored to use pytest.
# You can run tests using: pytest test_project.py
# Running this file directly will execute pytest on the test suite.

import sys
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["test_project.py"]))
