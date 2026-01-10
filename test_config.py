#!/usr/bin/env python3
"""
Test script for configuration management system.

This script demonstrates:
1. Loading configuration from .env file
2. Loading configuration with defaults (no .env)
3. Validation with helpful error messages
4. Accessing configuration values
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import load_config, ConfigValidationError


def test_default_config():
    """Test loading configuration with defaults (no .env file)."""
    print("=" * 60)
    print("TEST 1: Loading configuration with defaults (no .env)")
    print("=" * 60)

    try:
        config = load_config()
        print("\n✓ Configuration loaded successfully!\n")

        print("Model Configuration:")
        print(f"  - Repository: {config.model.repo}")
        print(f"  - Cache Dir: {config.model.cache_dir or 'Using HuggingFace default'}")

        print("\nNetwork Configuration:")
        print(f"  - API Host: {config.network.api_host}")
        print(f"  - API Port: {config.network.api_port}")
        print(f"  - Worker Hosts: {', '.join(config.network.worker_hosts)}")
        print(f"  - Worker SSH: {', '.join(config.network.worker_ssh)}")

        print("\nKV-Cache Configuration:")
        print(f"  - Max Size: {config.kv_cache.max_size or 'No limit'}")
        print(f"  - Reserved Memory: {config.kv_cache.reserved_memory_mb} MB")
        print(f"  - Max Sequence Length: {config.kv_cache.max_sequence_length}")

        print("\nPerformance Configuration:")
        print(f"  - Max Prompt Length: {config.performance.max_prompt_len_bytes} bytes")
        print(f"  - Request Timeout: {config.performance.request_timeout_seconds} seconds")
        print(f"  - Poll Interval: {config.performance.poll_interval_seconds} seconds")
        print(f"  - Default Max Tokens: {config.performance.default_max_tokens}")

        print("\nDistributed Configuration:")
        print(f"  - Number of Devices: {config.distributed.num_devices}")
        print(f"  - Backend: {config.distributed.backend}")

        print("\nFile Paths Configuration:")
        print(f"  - Request File: {config.file_paths.request_file_path}")
        print(f"  - Response File: {config.file_paths.response_file_path}")
        print(f"  - Server Log: {config.file_paths.server_log_path}")
        print(f"  - API Log: {config.file_paths.api_log_path}")

        print("\nSystem Configuration:")
        print(
            f"  - File Descriptor Limits: ({config.system.file_descriptor_soft_limit}, {config.system.file_descriptor_hard_limit})"
        )
        print(f"  - Model Load Wait: {config.system.model_load_wait_seconds} seconds")
        print(f"  - Log Level: {config.system.log_level}")

        return True
    except ConfigValidationError as e:
        print(f"\n✗ Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def test_env_override():
    """Test loading configuration with environment variable overrides."""
    print("\n" + "=" * 60)
    print("TEST 2: Testing environment variable overrides")
    print("=" * 60)

    # Set some environment variables
    os.environ["API_PORT"] = "9000"
    os.environ["NUM_DEVICES"] = "4"
    os.environ["LOG_LEVEL"] = "DEBUG"

    try:
        config = load_config()
        print("\n✓ Configuration loaded with overrides!\n")

        print("Overridden values:")
        print(f"  - API Port: {config.network.api_port} (expected: 9000)")
        print(f"  - Num Devices: {config.distributed.num_devices} (expected: 4)")
        print(f"  - Log Level: {config.system.log_level} (expected: DEBUG)")

        # Verify overrides worked
        assert config.network.api_port == 9000, "API_PORT override failed"
        assert config.distributed.num_devices == 4, "NUM_DEVICES override failed"
        assert config.system.log_level == "DEBUG", "LOG_LEVEL override failed"

        print("\n✓ All overrides working correctly!")

        # Clean up
        del os.environ["API_PORT"]
        del os.environ["NUM_DEVICES"]
        del os.environ["LOG_LEVEL"]

        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_validation():
    """Test configuration validation with invalid values."""
    print("\n" + "=" * 60)
    print("TEST 3: Testing validation with invalid values")
    print("=" * 60)

    test_cases = [
        ("API_PORT", "99999", "Port must be between 1 and 65535"),
        ("API_PORT", "abc", "Must be an integer"),
        ("NUM_DEVICES", "0", "Must be positive"),
        ("POLL_INTERVAL_SECONDS", "15", "Must be between 0 and 10"),
        ("LOG_LEVEL", "INVALID", "Must be valid log level"),
    ]

    all_passed = True
    for var_name, var_value, expected_error in test_cases:
        # Set invalid value
        os.environ[var_name] = var_value

        try:
            config = load_config()
            print(f"\n✗ {var_name}={var_value}: Should have failed but didn't!")
            all_passed = False
        except ConfigValidationError as e:
            print(f"\n✓ {var_name}={var_value}: Correctly rejected")
            print(f"   Error: {e}")
        except Exception as e:
            print(f"\n✓ {var_name}={var_value}: Caught error")
            print(f"   Error: {e}")
        finally:
            # Clean up
            del os.environ[var_name]

    return all_passed


def test_dotenv_loading():
    """Test loading configuration from .env file."""
    print("\n" + "=" * 60)
    print("TEST 4: Testing .env file loading")
    print("=" * 60)

    # Create a test .env file
    test_env_file = project_root / ".env.test"
    test_env_content = """
# Test .env file
API_PORT=7777
MODEL_REPO=test-model/test-repo
NUM_DEVICES=3
LOG_LEVEL=WARNING
"""

    test_env_file.write_text(test_env_content)

    try:
        config = load_config(str(test_env_file))
        print("\n✓ Configuration loaded from .env.test!\n")

        print("Values from .env.test:")
        print(f"  - API Port: {config.network.api_port} (expected: 7777)")
        print(f"  - Model Repo: {config.model.repo} (expected: test-model/test-repo)")
        print(f"  - Num Devices: {config.distributed.num_devices} (expected: 3)")
        print(f"  - Log Level: {config.system.log_level} (expected: WARNING)")

        # Verify values
        assert config.network.api_port == 7777, "API_PORT from .env failed"
        assert config.model.repo == "test-model/test-repo", "MODEL_REPO from .env failed"
        assert config.distributed.num_devices == 3, "NUM_DEVICES from .env failed"
        assert config.system.log_level == "WARNING", "LOG_LEVEL from .env failed"

        print("\n✓ All values from .env file loaded correctly!")

        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False
    finally:
        # Clean up
        if test_env_file.exists():
            test_env_file.unlink()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Configuration Management System Tests" + " " * 11 + "║")
    print("╚" + "═" * 58 + "╝")

    results = []

    # Run tests
    results.append(("Default Configuration", test_default_config()))
    results.append(("Environment Overrides", test_env_override()))
    results.append(("Validation", test_validation()))
    results.append(("dotenv Loading", test_dotenv_loading()))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n✓ All tests passed! Configuration system is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
