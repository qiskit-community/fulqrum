# Fulqrum benchmarks


## Long-running tests

A test that is long-running can be marked in the following manner:

```python
@pytest.mark.long
```

Long-running tests are not executed by default.  To run them one must call

```bash
pytest -m long benchmarks
```

from the top-level fulqrum directory.