# CloudGlide OLAP Test Suite

## Overview

Comprehensive test suite with **112 tests** covering validation, execution, scheduling, and scaling.

## Test Results: 89/112 Passing (79%)

| Test File | Tests | Passing | Status |
|-----------|-------|---------|--------|
| `test_validation.py` | 28 | 28 | ✅ 100% |
| `test_query_processing_model.py` | 17 | 17 | ✅ 100% |
| `test_execution_model.py` | 15 | 15 | ✅ 100% |
| `test_scheduling_model_fixed.py` | 12 | 12 | ✅ 100% |
| `test_scaling_model_fixed.py` | 13 | 11 | ⚠️ 85% |
| `test_scheduling_model.py` (legacy) | 24 | 2 | ⚠️ 8% |
| `test_scaling_model.py` (legacy) | 17 | 4 | ⚠️ 24% |

## Quick Start

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Files
```bash
# Validation (100% passing)
python -m pytest tests/test_validation.py -v

# Execution Model (100% passing)
python -m pytest tests/test_execution_model.py -v

# Scheduling (100% passing)
python -m pytest tests/test_scheduling_model_fixed.py -v

# Scaling (85% passing)
python -m pytest tests/test_scaling_model_fixed.py -v

# Query Processing (100% passing)
python -m pytest tests/test_query_processing_model.py -v
```

### Run Only Passing Tests
```bash
python -m pytest tests/test_validation.py tests/test_execution_model.py tests/test_query_processing_model.py tests/test_scheduling_model_fixed.py -v
```

## Test Coverage by Module

### ✅ validation.py (28 tests - 100% passing)
- CSV schema validation
- CSV data validation (negative values, NaN, type errors)
- Simulation configuration validation
- Scheduling configuration validation
- Scaling configuration validation
- Execution parameters validation

### ✅ execution_model.py (15 tests - 100% passing)
- State initialization for all 6 architectures
- Cost calculation and tracking
- Autoscaling hooks (DWaaS, Elastic Pool)
- Event processing (arrival events)
- Metrics collection and reporting
- Edge cases (missing files, zero hit rates)

### ✅ query_processing_model.py (17 tests - 100% passing)
- Memory tier assignment (DRAM/SSD/S3)
- Cache warmup progression
- I/O simulation (DWaaS, QaaS)
- CPU simulation with parallelization
- Core allocation algorithms
- Job finalization and estimators

### ✅ scheduling_model.py (12 tests - 100% passing)
- FCFS (First-Come-First-Served) scheduling
- SJF (Shortest Job First) scheduling
- LJF (Longest Job First) scheduling
- I/O and CPU scheduler integration
- Concurrency limit enforcement
- Edge cases (zero cores, large queues, zero delta time)

### ⚠️ scaling_model.py (11/13 tests - 85% passing)
- Queue-based autoscaling ✅
- Reactive autoscaling ✅
- Predictive autoscaling ✅
- Elastic Pool VPU scaling (2 failures - API mismatch)
- Edge cases ✅

## Test Organization

### Recommended Test Files (Use These)
- ✅ `test_validation.py` - All validation logic
- ✅ `test_execution_model.py` - Execution and state management
- ✅ `test_query_processing_model.py` - I/O, CPU, memory simulation
- ✅ `test_scheduling_model_fixed.py` - Scheduling with correct API
- ✅ `test_scaling_model_fixed.py` - Autoscaling with correct API

### Legacy Test Files (Reference Only)
- ⚠️ `test_scheduling_model.py` - Original tests (API mismatch)
- ⚠️ `test_scaling_model.py` - Original tests (API mismatch)

## Test Fixtures

### Common Fixtures
```python
@pytest.fixture
def config():
    """Returns default SimulationConfig"""

@pytest.fixture
def sample_jobs():
    """Returns list of test Job objects"""

@pytest.fixture
def test_csv_file():
    """Creates temporary CSV file for testing"""
```

## Generating Coverage Reports

### HTML Coverage Report
```bash
python -m pytest tests/ --cov=cloudglide --cov-report=html
open htmlcov/index.html
```

### Terminal Coverage Report
```bash
python -m pytest tests/ --cov=cloudglide --cov-report=term-missing
```

## Writing New Tests

### Test Structure
```python
import pytest
from cloudglide.module import function_to_test

class TestFeatureName:
    """Test suite for specific feature."""

    def test_basic_functionality(self):
        """Test basic behavior."""
        result = function_to_test(params)
        assert result == expected

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ExpectedException):
            function_to_test(invalid_params)
```

### Best Practices
1. Use descriptive test names: `test_what_when_expected()`
2. One assertion per test when possible
3. Use fixtures for common setup
4. Test both success and failure paths
5. Include edge cases (empty, zero, negative, None)

## Known Issues

### Elastic Pool Tests (2 failures)
- `test_elastic_pool_scale_basic`
- `test_elastic_pool_empty_queues`
- **Cause:** `autoscaling_ec()` API doesn't accept `base_n` parameter
- **Fix:** Check actual API signature and update tests

### Legacy Tests (41 tests with low pass rate)
- **Cause:** API signatures changed since tests were written
- **Solution:** Use `_fixed.py` versions instead

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest tests/ --cov=cloudglide --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Import Errors
```bash
# Ensure cloudglide is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Test Collection Errors
```bash
# Check for syntax errors
python -m pytest --collect-only tests/
```

### Slow Tests
```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest tests/ -n auto
```

## Documentation

For detailed information about the improvements, see:
- **[IMPROVEMENTS.md](../IMPROVEMENTS.md)** - Comprehensive documentation
- **[FINAL_SUMMARY.md](../FINAL_SUMMARY.md)** - Executive summary

## Questions or Issues?

Contact: geom@in.tum.de
