#!/usr/bin/env python3
"""
Unified System Management Tool
==============================
A single CLI for all management operations.
"""
import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.database import DatabaseInitializer
from scripts.health_checks.health_checker import HealthChecker

async def main():
    """Main entry point for the system manager"""
    parser = argparse.ArgumentParser(
        description="Unified System Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to execute', required=True)

    # Init-db command
    init_db_parser = subparsers.add_parser('init-db', help='Initialize databases')
    init_db_parser.add_argument('--with-data', action='store_true', help='Seed initial data')

    # Health-check command
    health_parser = subparsers.add_parser('health-check', help='Run health checks')
    health_parser.add_argument('--detailed', action='store_true', help='Show detailed health info')

    args = parser.parse_args()

    if args.command == 'init-db':
        initializer = DatabaseInitializer()
        # The 'initialize_all' function returns a dictionary of results
        results = await initializer.initialize_all(with_data=args.with_data)
        # Check if all values in the dictionary are True
        success = all(results.values())
        if not success:
            print("❌ Database initialization failed for one or more services.")
            sys.exit(1)
        print("✅ All databases initialized successfully.")

    elif args.command == 'health-check':
        checker = HealthChecker()
        health_status = await checker.check_all(detailed=args.detailed)
        checker.print_summary(health_status)
        if not health_status['healthy']:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())