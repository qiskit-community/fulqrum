# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# conftest.py
import time
import numpy
import scipy
import pytest
import fulqrum
from packaging.version import parse


# function executed right after test items collected but before test run
def pytest_collection_modifyitems(config, items):
    if not config.getoption("-m"):
        skip_me = pytest.mark.skip(reason="use `-m long` to run this test")
        for item in items:
            if "long" in item.keywords:
                item.add_marker(skip_me)


def pytest_benchmark_update_json(config, benchmarks, output_json):
    """Adds custom sections to the pytest-benchmark report"""
    reporter = config.pluginmanager.get_plugin("terminalreporter")

    # 8.4.0 changed the internal attribute and type that session start
    # is recorded at. There does not seem to be a public api for it on
    # the terminal reporter so this lets us support both 8.4.0 or older
    # versions
    pytest_version = parse(pytest.__version__)
    if pytest_version.release >= (8, 4, 0):
        output_json["total_duration"] = time.time() - reporter._session_start.time
    else:
        output_json["total_duration"] = time.time() - reporter._sessionstarttime

    output_json["env_info"] = {
        "fulqrum": str(fulqrum.__version__),
        "numpy": str(numpy.__version__),
        "scipy": str(scipy.__version__),
    }
