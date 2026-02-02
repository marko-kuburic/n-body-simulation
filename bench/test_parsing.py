#!/usr/bin/env python3
"""
Test/validate the parsing functions on sample log files.
Creates mock log files and tests parsing functionality.
"""

import tempfile
import shutil
from pathlib import Path
import sys

# Add bench directory to path
sys.path.insert(0, str(Path(__file__).parent))

from parse_and_plot import parse_cpu_log, parse_cuda_log, extract_params_from_filename


def create_mock_cpu_log(content: str) -> Path:
    """Create a temporary CPU log file."""
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def create_mock_cuda_log(content: str) -> Path:
    """Create a temporary CUDA log file."""
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def test_cpu_parsing():
    """Test CPU log parsing."""
    print("Testing CPU log parsing...")
    
    sample_log = """
N-body Simulation Started
N = 6000, timesteps = 10

Timing Results:
1. Sequential Non-Symmetric:  45.234 seconds
2. Sequential Symmetric:      22.567 seconds
3. OpenMP Non-Symmetric:      2.345 seconds (threads=32)
4. OpenMP Symmetric:          3.456 seconds (threads=32)

Simulation complete.
"""
    
    logfile = create_mock_cpu_log(sample_log)
    
    try:
        times = parse_cpu_log(logfile)
        
        expected = {
            'seq_nonsym': 45.234,
            'seq_sym': 22.567,
            'omp_nonsym': 2.345,
            'omp_sym': 3.456
        }
        
        success = True
        for variant, expected_time in expected.items():
            if variant not in times:
                print(f"  ✗ FAIL: {variant} not found in parsed times")
                success = False
            elif abs(times[variant] - expected_time) > 0.001:
                print(f"  ✗ FAIL: {variant} time mismatch: {times[variant]} != {expected_time}")
                success = False
            else:
                print(f"  ✓ {variant}: {times[variant]}s")
        
        if success:
            print("  ✓ CPU parsing PASSED")
            return True
        else:
            print("  ✗ CPU parsing FAILED")
            return False
            
    finally:
        logfile.unlink()


def test_cuda_parsing():
    """Test CUDA log parsing."""
    print("\nTesting CUDA log parsing...")
    
    sample_log = """
CUDA N-body Simulation
N = 6000, timesteps = 10

Running CUDA kernel...

Timing Results:
5. CUDA Non-Symmetric (kernel):    0.234 seconds
   CUDA Non-Symmetric (total):     0.456 seconds

Simulation complete.
"""
    
    logfile = create_mock_cuda_log(sample_log)
    
    try:
        times = parse_cuda_log(logfile)
        
        expected = {
            'kernel': 0.234,
            'total': 0.456
        }
        
        success = True
        for time_type, expected_time in expected.items():
            if time_type not in times:
                print(f"  ✗ FAIL: {time_type} not found in parsed times")
                success = False
            elif abs(times[time_type] - expected_time) > 0.001:
                print(f"  ✗ FAIL: {time_type} time mismatch: {times[time_type]} != {expected_time}")
                success = False
            else:
                print(f"  ✓ {time_type}: {times[time_type]}s")
        
        if success:
            print("  ✓ CUDA parsing PASSED")
            return True
        else:
            print("  ✗ CUDA parsing FAILED")
            return False
            
    finally:
        logfile.unlink()


def test_filename_parsing():
    """Test parameter extraction from filenames."""
    print("\nTesting filename parsing...")
    
    test_cases = [
        ("cpu_strong_T32_N6000_S10_run3.log", {
            'threads': 32, 'N': 6000, 'steps': 10, 'run': 3,
            'experiment': 'strong_scaling'
        }),
        ("cpu_size_N8000_run5.log", {
            'N': 8000, 'run': 5, 'experiment': 'size_sweep'
        }),
        ("cuda_tiled_N6000_S10_run2.log", {
            'N': 6000, 'steps': 10, 'run': 2,
            'experiment': 'cuda', 'cuda_variant': 'tiled'
        }),
        ("final_cpu_N4000_S5_run7.log", {
            'N': 4000, 'steps': 5, 'run': 7,
            'experiment': 'final_cpu'
        }),
    ]
    
    all_passed = True
    
    for filename, expected_params in test_cases:
        params = extract_params_from_filename(filename)
        
        match = True
        for key, expected_value in expected_params.items():
            if params.get(key) != expected_value:
                print(f"  ✗ FAIL: {filename}")
                print(f"    Expected {key}={expected_value}, got {params.get(key)}")
                match = False
                all_passed = False
                break
        
        if match:
            print(f"  ✓ {filename}")
    
    if all_passed:
        print("  ✓ Filename parsing PASSED")
    else:
        print("  ✗ Filename parsing FAILED")
    
    return all_passed


def main():
    """Run all tests."""
    print("=" * 80)
    print("Validating Benchmark Parsing Functions")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(test_cpu_parsing())
    results.append(test_cuda_parsing())
    results.append(test_filename_parsing())
    
    print()
    print("=" * 80)
    
    if all(results):
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
