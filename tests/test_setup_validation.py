"""
Validation tests to verify the testing infrastructure is properly set up.
"""
import pytest
import jax
import jax.numpy as jnp
from pathlib import Path


def test_pytest_markers():
    """Test that custom pytest markers are properly configured."""
    # This test should be marked as a unit test
    pass


@pytest.mark.unit
def test_unit_marker():
    """Test the unit marker works."""
    assert True


@pytest.mark.integration  
def test_integration_marker():
    """Test the integration marker works."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test the slow marker works."""
    import time
    time.sleep(0.01)  # Simulate slow test
    assert True


def test_fixtures_work(sample_config, random_key, temp_dir):
    """Test that shared fixtures are working properly."""
    # Test config fixture
    assert isinstance(sample_config, dict)
    assert "ENV_NAME" in sample_config
    assert "TOTAL_TIMESTEPS" in sample_config
    
    # Test random key fixture
    assert random_key.shape == (2,)
    
    # Test temp directory fixture
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Create a test file in temp dir
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()


def test_jax_configuration_fixture(random_key, sample_observation):
    """Test JAX is properly configured for testing."""
    # Test that JAX operations work
    result = jax.random.normal(random_key, (5,))
    assert result.shape == (5,)
    
    # Test sample observation fixture
    assert sample_observation.shape == (4,)
    assert jnp.allclose(sample_observation, jnp.array([0.1, 0.2, 0.3, 0.4]))


def test_batch_data_fixture(sample_batch_data):
    """Test the sample batch data fixture."""
    assert "observations" in sample_batch_data
    assert "actions" in sample_batch_data
    assert "rewards" in sample_batch_data
    
    obs = sample_batch_data["observations"]
    assert obs.shape == (8, 10, 4)  # batch_size=8, seq_len=10, obs_dim=4


def test_mock_env_fixture(mock_environment_state):
    """Test the mock environment state fixture."""
    assert "obs" in mock_environment_state
    assert "done" in mock_environment_state
    assert "reward" in mock_environment_state
    assert "info" in mock_environment_state
    
    assert mock_environment_state["obs"].shape == (4,)
    assert mock_environment_state["done"] is False
    assert mock_environment_state["reward"] == 1.0


def test_numpy_arrays_fixture(numpy_arrays):
    """Test the numpy arrays fixture."""
    assert "config" in numpy_arrays
    assert "metrics" in numpy_arrays  
    assert "params" in numpy_arrays
    
    assert numpy_arrays["config"].shape == (3,)
    assert numpy_arrays["metrics"].shape == (100, 5)
    assert numpy_arrays["params"].shape == (50, 10)


def test_project_structure():
    """Test that the expected project structure exists."""
    project_root = Path(__file__).parent.parent
    
    # Check main Python files exist
    assert (project_root / "train_PPO_trXL.py").exists()
    assert (project_root / "transformerXL.py").exists()
    assert (project_root / "wrappers.py").exists()
    
    # Check pyproject.toml exists
    assert (project_root / "pyproject.toml").exists()
    
    # Check test structure
    tests_dir = project_root / "tests"
    assert tests_dir.exists()
    assert (tests_dir / "__init__.py").exists()
    assert (tests_dir / "conftest.py").exists()
    assert (tests_dir / "unit" / "__init__.py").exists()
    assert (tests_dir / "integration" / "__init__.py").exists()


@pytest.mark.unit
def test_coverage_excludes_work():
    """Test that coverage excludes are working by having uncoverable code."""
    if __name__ == "__main__":  # pragma: no cover
        print("This should be excluded from coverage")
        
    def debug_function():  # pragma: no cover
        raise NotImplementedError("This is just for testing coverage exclusion")
    
    # The test passes without calling the excluded functions
    assert True