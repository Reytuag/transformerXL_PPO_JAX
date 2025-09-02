import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import numpy as np
import jax
import jax.numpy as jnp


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration dictionary for testing."""
    return {
        "ENV_NAME": "CartPole-v1",
        "TOTAL_TIMESTEPS": 1000000,
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": False,
        "MEMORY_SIZE": 128,
        "NUM_LAYERS": 2,
        "FFN_SIZE_MULTIPLIER": 4
    }


@pytest.fixture
def random_key():
    """Provide a JAX random key for testing."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_observation():
    """Provide a sample observation array."""
    return jnp.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def sample_action():
    """Provide a sample action."""
    return jnp.array(1)


@pytest.fixture
def sample_batch_data():
    """Provide sample batch data for testing."""
    batch_size = 8
    seq_len = 10
    obs_dim = 4
    
    return {
        "observations": jnp.ones((batch_size, seq_len, obs_dim)),
        "actions": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "rewards": jnp.ones((batch_size, seq_len)),
        "dones": jnp.zeros((batch_size, seq_len), dtype=bool),
        "values": jnp.ones((batch_size, seq_len)),
        "log_probs": jnp.ones((batch_size, seq_len))
    }


@pytest.fixture
def mock_environment_state():
    """Provide a mock environment state for testing."""
    return {
        "obs": jnp.array([0.0, 0.1, 0.2, 0.3]),
        "done": False,
        "reward": 1.0,
        "info": {}
    }


@pytest.fixture(autouse=True)
def setup_jax():
    """Configure JAX for testing (runs automatically for all tests)."""
    # Set JAX to use CPU for testing to avoid GPU memory issues
    jax.config.update('jax_platform_name', 'cpu')
    # Disable JIT compilation for easier debugging during tests
    jax.config.update('jax_disable_jit', True)
    yield
    # Reset to default after tests
    jax.config.update('jax_disable_jit', False)


@pytest.fixture
def numpy_arrays():
    """Provide sample numpy arrays for testing data loading."""
    return {
        "config": np.array([1, 2, 3]),
        "metrics": np.random.randn(100, 5),
        "params": np.random.randn(50, 10)
    }