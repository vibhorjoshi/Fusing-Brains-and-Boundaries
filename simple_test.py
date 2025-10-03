#!/usr/bin/env python3
"""
Simple test script for GeoAI Research project
Prints basic information and creates a test file
"""

import os
import sys
import datetime

def main():
    """Main function"""
    print("GeoAI Research - Simple Test Script")
    print("-----------------------------------")
    
    # Print system information
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Current time: {datetime.datetime.now()}")
    
    # Create a test output file
    output_file = "test_output.txt"
    try:
        with open(output_file, "w") as f:
            f.write("GeoAI Research Test Output\n")
            f.write(f"Generated on: {datetime.datetime.now()}\n")
            f.write(f"Python version: {sys.version}\n")
            f.write("Test completed successfully!\n")
        
        print(f"Created test output file: {output_file}")
        return 0
    except Exception as e:
        print(f"Error creating test file: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())