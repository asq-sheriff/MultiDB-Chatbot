
"""
Quick test script to verify the API fix works.
Run this after starting the API to ensure endpoints are responding.
"""

import requests
import json
import sys

def test_endpoint(name, method, url, data=None, headers=None):
    """Test a single endpoint and return result"""
    try:
        print(f"🧪 Testing {name}...")

        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        else:
            print(f"❌ Unsupported method: {method}")
            return False

        if response.status_code == 200:
            print(f"✅ {name} - SUCCESS (Status: {response.status_code})")
            return True
        else:
            print(f"❌ {name} - FAILED (Status: {response.status_code})")
            print(f"   Response: {response.text[:200]}...")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ {name} - CONNECTION ERROR (Is the API running?)")
        return False
    except Exception as e:
        print(f"❌ {name} - ERROR: {str(e)}")
        return False

def main():
    base_url = "http://localhost:8000"

    print("🚀 Quick API Test")
    print("=" * 40)

    # Test basic endpoints
    tests = [
        ("Root Endpoint", "GET", f"{base_url}/"),
        ("Health Check", "GET", f"{base_url}/health"),
        ("API Docs", "GET", f"{base_url}/docs"),
    ]

    # Test chat endpoint (anonymous)
    tests.append((
        "Anonymous Chat",
        "POST",
        f"{base_url}/api/v1/chat/message",
        {"message": "Hello, what is Redis?"}
    ))

    # Run tests
    passed = 0
    total = len(tests)

    for test in tests:
        if test_endpoint(*test):
            passed += 1

    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the API setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())