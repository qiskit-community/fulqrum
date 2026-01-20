# Fulqrum benchmarks

A collection of scripts for tracking Fulqrum performance.


## Running benchmarks

The standard set of benchmarks can be executed from the top-level Fulqrum directory using:

```bash
pytest benchmarks
```

To save the results to a file use:

```bash
pytest --benchmark-save=FILE_NAME benchmarks
```


## Long-running tests

A test that is long-running can be marked in the following manner:

```python
@pytest.mark.long
```

Long-running tests are not executed by default.  To run them one must call

```bash
pytest -m long benchmarks
```

from the top-level Fulqrum directory.


## Configuration options

Configuration options can be set in the `default.conf` file.
