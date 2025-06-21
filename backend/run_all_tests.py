#!/usr/bin/env python3
"""
Comprehensive Test Runner for Naarad AI Assistant Backend

This script runs all test suites in the correct order:
1. Configuration tests
2. API functionality tests
3. Integration tests
4. Performance tests

Usage:
    python run_all_tests.py [--config-only] [--api-only] [--verbose]
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)

def print_section(title: str):
    """Print a section header."""
    print(f"\n📋 {title}")
    print("-" * 60)

def check_server_running() -> bool:
    """Check if the server is running."""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def run_configuration_tests() -> bool:
    """Run configuration tests."""
    print_section("Running Configuration Tests")
    
    try:
        result = subprocess.run([
            sys.executable, "test_configuration.py"
        ], capture_output=True, text=True, cwd=backend_dir)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running configuration tests: {e}")
        return False

def run_api_tests() -> bool:
    """Run API functionality tests."""
    print_section("Running API Functionality Tests")
    
    try:
        result = subprocess.run([
            sys.executable, "test_comprehensive_suite.py"
        ], capture_output=True, text=True, cwd=backend_dir)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running API tests: {e}")
        return False

def run_existing_tests() -> bool:
    """Run existing test files."""
    print_section("Running Existing Test Files")
    
    test_files = [
        "test_chat_api.py",
        "test_agent_responses_v2.py",
        "test_agents.py",
        "test_domain_agents.py",
        "test_enhanced_router.py"
    ]
    
    success = True
    for test_file in test_files:
        test_path = Path(backend_dir) / "tests" / test_file
        if test_path.exists():
            print(f"\n🔍 Running {test_file}...")
            try:
                result = subprocess.run([
                    sys.executable, str(test_path)
                ], capture_output=True, text=True, cwd=backend_dir)
                
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                
                if result.returncode != 0:
                    success = False
                    print(f"❌ {test_file} failed")
                else:
                    print(f"✅ {test_file} passed")
            except Exception as e:
                print(f"❌ Error running {test_file}: {e}")
                success = False
        else:
            print(f"⚠️  {test_file} not found, skipping")
    
    return success

def run_pytest_tests() -> bool:
    """Run tests using pytest."""
    print_section("Running Pytest Tests")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
        ], capture_output=True, text=True, cwd=backend_dir)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running pytest tests: {e}")
        return False

def run_performance_tests() -> bool:
    """Run performance tests."""
    print_section("Running Performance Tests")
    
    try:
        import requests
        import time
        
        # Test response time
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/api/chat",
            json={"message": "Hello", "conversation_id": "perf_test"}
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        print(f"✅ Single request response time: {response_time:.2f}s")
        
        # Test concurrent requests
        import threading
        import concurrent.futures
        
        def make_request():
            try:
                response = requests.post(
                    "http://localhost:8000/api/chat",
                    json={"message": "Test", "conversation_id": "concurrent_test"}
                )
                return response.status_code
            except Exception as e:
                return f"Error: {e}"
        
        print("🔄 Testing concurrent requests...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        success_count = sum(1 for r in results if r == 200)
        print(f"✅ Concurrent requests: {success_count}/5 successful")
        
        return success_count >= 3  # At least 60% success rate
    except Exception as e:
        print(f"❌ Error running performance tests: {e}")
        return False

def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    print_section("Checking Dependencies")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "requests",
        "pydantic",
        "langchain",
        "groq",
        "dotenv"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    return True

def generate_test_report(results: Dict[str, bool]) -> None:
    """Generate a test report."""
    print_banner("TEST REPORT")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    print(f"📊 Total Test Suites: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\n📋 Detailed Results:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    if failed_tests > 0:
        print("\n🔧 Recommendations:")
        print("   1. Check your .env file configuration")
        print("   2. Ensure the server is running (uvicorn main:app --reload)")
        print("   3. Verify API keys are valid")
        print("   4. Check network connectivity")
        print("   5. Review error logs for specific issues")

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run comprehensive tests for Naarad AI Assistant")
    parser.add_argument("--config-only", action="store_true", help="Run only configuration tests")
    parser.add_argument("--api-only", action="store_true", help="Run only API tests")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--skip-server-check", action="store_true", help="Skip server availability check")
    
    args = parser.parse_args()
    
    print_banner("Naarad AI Assistant - Comprehensive Test Suite")
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Check if server is running (unless skipped)
    if not args.skip_server_check:
        print_section("Checking Server Status")
        if check_server_running():
            print("✅ Server is running on http://localhost:8000")
        else:
            print("⚠️  Server is not running")
            print("   Start the server with: uvicorn main:app --reload")
            if not args.config_only:
                print("   Some tests may fail without the server running")
    
    # Initialize results
    results = {}
    
    # Run configuration tests
    if not args.api_only:
        results["Configuration Tests"] = run_configuration_tests()
    
    # Run API tests
    if not args.config_only:
        results["API Functionality Tests"] = run_api_tests()
        results["Existing Test Files"] = run_existing_tests()
        results["Pytest Tests"] = run_pytest_tests()
        results["Performance Tests"] = run_performance_tests()
    
    # Generate report
    generate_test_report(results)
    
    # Exit with appropriate code
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please review the report above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 