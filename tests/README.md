# Test Suite for bituslabs_ds

This directory contains comprehensive tests for the bituslabs_ds AWS data science project.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and shared fixtures
├── pytest.ini                 # Pytest settings and markers
├── test_utilities.py          # Test utilities and helper functions
├── README.md                  # This file
├── unit/                      # Unit tests
│   ├── __init__.py
│   ├── test_utils.py          # Tests for utils module
│   ├── test_config.py         # Tests for config module
│   ├── test_descriptors.py    # Tests for descriptors module
│   ├── test_eda.py            # Tests for eda module
│   ├── test_s3_utils.py       # Tests for s3_utils module
│   ├── test_athena_utils.py   # Tests for athena_utils module
│   └── test_pyspark_utils.py  # Tests for pyspark_utils module
└── integration/               # Integration tests
    ├── __init__.py
    ├── test_aws_integration.py # AWS services integration tests
    └── test_ml_integration.py  # ML pipeline integration tests
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Unit Tests Only
```bash
pytest tests/unit/
```

### Run Integration Tests Only
```bash
pytest tests/integration/
```

### Run Tests by Marker
```bash
# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only unit tests
pytest -m "unit"

# Run only integration tests
pytest -m "integration"

# Run only AWS-related tests
pytest -m "aws"

# Run only Spark-related tests
pytest -m "spark"
```

### Run Specific Test Files
```bash
# Run specific test file
pytest tests/unit/test_utils.py

# Run specific test class
pytest tests/unit/test_utils.py::TestGetDataFromUrl

# Run specific test function
pytest tests/unit/test_utils.py::TestGetDataFromUrl::test_download_and_extract_success
```

### Run Tests with Coverage
```bash
# Install coverage first
pip install pytest-cov

# Run with coverage
pytest --cov=bituslabs_ds --cov-report=html --cov-report=term-missing
```

### Run Tests in Parallel
```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

## Test Categories

### Unit Tests
- **Location**: `tests/unit/`
- **Purpose**: Test individual functions and classes in isolation
- **Mocking**: Heavy use of mocks to isolate units under test
- **Speed**: Fast execution (< 1 second per test)
- **Dependencies**: Minimal external dependencies

### Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Test interactions between components and external services
- **Mocking**: Limited mocking, use real AWS services with moto
- **Speed**: Slower execution (1-10 seconds per test)
- **Dependencies**: AWS services, Spark (when applicable)

## Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.aws`: Tests that require AWS services
- `@pytest.mark.spark`: Tests that require Spark
- `@pytest.mark.smoke`: Smoke tests for basic functionality

## Fixtures

### Common Fixtures (in conftest.py)
- `temp_dir`: Temporary directory for test files
- `sample_dataframe`: Sample pandas DataFrame
- `sample_numeric_dataframe`: Sample numeric DataFrame
- `sample_time_series_data`: Sample time series data
- `mock_s3_client`: Mock S3 client using moto
- `mock_athena_client`: Mock Athena client using moto
- `mock_spark_session`: Mock Spark session
- `mock_aws_credentials`: Mock AWS credentials
- `sample_config`: Sample configuration
- `sample_transitions_data`: Sample data for GAIL testing

### Utility Fixtures (in test_utilities.py)
- `test_data_generator`: Generate various types of test data
- `test_file_manager`: Manage test files and cleanup
- `performance_timer`: Measure test execution time
- `mock_aws_s3_client`: Custom mock S3 client
- `mock_aws_athena_client`: Custom mock Athena client

## Test Data

### Sample Data Generation
The test suite includes utilities to generate various types of test data:

```python
# Generate sample DataFrame
df = TestDataGenerator.create_sample_dataframe(
    n_rows=100,
    n_numeric_cols=5,
    include_missing=True,
    include_outliers=True
)

# Generate time series data
ts_data = TestDataGenerator.create_time_series_data(
    n_points=1000,
    n_series=3,
    include_trend=True,
    include_seasonality=True
)

# Generate correlated data
corr_data = TestDataGenerator.create_correlated_data(
    n_rows=100,
    n_features=10,
    correlation_strength=0.8
)
```

## Mocking Strategy

### AWS Services
- Use `moto` library for mocking AWS services
- Mock S3, Athena, and other AWS services
- Provide realistic responses for testing

### External Dependencies
- Mock Spark sessions for PySpark tests
- Mock file system operations
- Mock network requests

### Data Processing
- Use small, controlled datasets for unit tests
- Use larger, realistic datasets for integration tests
- Generate synthetic data for edge cases

## Performance Testing

### Timing Tests
```python
def test_performance():
    with PerformanceTimer() as timer:
        # Code to time
        result = expensive_operation()
    
    assert timer.elapsed_time < 1.0  # Should complete in under 1 second
```

### Memory Testing
```python
def test_memory_usage():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Run memory-intensive operation
    result = memory_intensive_operation()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
```

## Continuous Integration

### GitHub Actions
The test suite is designed to work with GitHub Actions:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist
      - name: Run tests
        run: pytest --cov=bituslabs_ds --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Local Development
For local development, use:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests with coverage
pytest --cov=bituslabs_ds --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests during development
```

## Best Practices

### Writing Tests
1. **Test one thing at a time**: Each test should verify a single behavior
2. **Use descriptive names**: Test names should clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Use fixtures**: Reuse common test data and setup through fixtures
5. **Mock external dependencies**: Isolate units under test

### Test Data
1. **Use realistic data**: Test with data that resembles production data
2. **Test edge cases**: Include tests for boundary conditions and error cases
3. **Use small datasets**: Keep unit test data small for fast execution
4. **Generate data programmatically**: Use factories and generators for test data

### Assertions
1. **Be specific**: Use specific assertions rather than generic ones
2. **Test behavior, not implementation**: Focus on what the code does, not how
3. **Use custom assertions**: Create helper functions for complex assertions
4. **Include error messages**: Provide clear error messages in assertions

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Make sure the package is installed in development mode
pip install -e .

# Or add the project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### AWS Credentials
```bash
# For integration tests, ensure AWS credentials are available
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2
```

#### Spark Issues
```bash
# Ensure Java is installed for Spark
java -version

# Set JAVA_HOME if needed
export JAVA_HOME=/path/to/java
```

#### Memory Issues
```bash
# Run tests with memory limits
pytest --maxfail=1 -x  # Stop on first failure
pytest -k "not slow"   # Skip slow tests
```

### Debugging Tests
```bash
# Run with verbose output
pytest -v -s

# Run with debugger
pytest --pdb

# Run specific test with debugger
pytest tests/unit/test_utils.py::TestGetDataFromUrl::test_download_and_extract_success --pdb
```

## Contributing

When adding new tests:

1. **Follow naming conventions**: Use descriptive test names
2. **Add appropriate markers**: Mark tests with relevant categories
3. **Update fixtures**: Add new fixtures to conftest.py if needed
4. **Document new utilities**: Add documentation for new test utilities
5. **Maintain coverage**: Ensure new code is covered by tests
6. **Update this README**: Document any new test patterns or utilities

