# Publish all changes to GitHub
# This script adds, commits, and pushes all unpublished files to the GitHub repository

import os
import subprocess
import argparse

def run_command(command, verbose=True):
    """Run a command and print the output if verbose"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    return result

def main():
    parser = argparse.ArgumentParser(description="Publish changes to GitHub repository")
    parser.add_argument("--message", "-m", default="Update frontend with enhanced visualization components", 
                        help="Commit message")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    args = parser.parse_args()
    
    # Check if we're in a git repository
    print("Checking git repository status...")
    result = run_command("git rev-parse --is-inside-work-tree", args.verbose)
    if result.returncode != 0:
        print("Error: Not in a git repository. Please run this script from the root of your git repository.")
        return
    
    # Check for unpublished files
    print("Checking for unpublished files...")
    result = run_command("git status --porcelain", args.verbose)
    
    if not result.stdout:
        print("No changes to publish. Everything is up to date.")
        return
    
    # Print the files that will be published
    print("\nThe following changes will be published:")
    unpublished_files = result.stdout.strip().split("\n")
    for file in unpublished_files:
        status = file[:2]
        filename = file[3:]
        
        if status == "??":
            print(f" - New file: {filename}")
        elif status == " M" or status == "M ":
            print(f" - Modified: {filename}")
        elif status == " D" or status == "D ":
            print(f" - Deleted: {filename}")
        else:
            print(f" - {status}: {filename}")
    
    # Confirm with the user
    confirmation = input("\nDo you want to publish these changes? (y/n): ")
    if confirmation.lower() != "y" and confirmation.lower() != "yes":
        print("Operation cancelled.")
        return
    
    # Add all changes
    print("\nAdding changes...")
    run_command("git add .", args.verbose)
    
    # Commit changes
    print("\nCommitting changes...")
    run_command(f'git commit -m "{args.message}"', args.verbose)
    
    # Push changes
    print("\nPushing changes to GitHub...")
    run_command("git push", args.verbose)
    
    print("\nAll changes have been published to GitHub successfully!")
    
    # Display repository URL
    result = run_command("git config --get remote.origin.url", False)
    if result.returncode == 0:
        repo_url = result.stdout.strip()
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]
        if repo_url.startswith("git@github.com:"):
            repo_url = "https://github.com/" + repo_url[15:]
        print(f"Repository URL: {repo_url}")

if __name__ == "__main__":
    main()