#!/usr/bin/env python3
"""
Security check utility for GeoAI Research project
Scans Python dependencies, code, and Docker images for security vulnerabilities
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Any, Optional, Tuple


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(
        description="Security check utility for GeoAI Research project"
    )
    parser.add_argument(
        "--deps", "-d", action="store_true", help="Check Python dependencies"
    )
    parser.add_argument("--code", "-c", action="store_true", help="Check Python code")
    parser.add_argument(
        "--docker", "-i", action="store_true", help="Check Docker images"
    )
    parser.add_argument(
        "--all", "-a", action="store_true", help="Run all security checks"
    )
    parser.add_argument(
        "--output", "-o", default="security-report", help="Output directory for reports"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser


def check_dependencies() -> Tuple[bool, Dict]:
    """Check Python dependencies using safety"""
    print("Checking Python dependencies...")
    
    # Ensure safety is installed
    try:
        subprocess.run(["safety", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Installing safety...")
        subprocess.run([sys.executable, "-m", "pip", "install", "safety"], check=True)
    
    # Run safety check
    try:
        result = subprocess.run(
            ["safety", "check", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        
        # safety returns exit code 0 when no vulnerabilities are found
        # and exit code 1 when vulnerabilities are found
        if result.returncode > 1:
            print(f"Error running safety: {result.stderr}")
            return False, {"error": result.stderr}
        
        # Parse JSON output
        try:
            vulnerabilities = json.loads(result.stdout)
            vulnerability_count = len(vulnerabilities)
            print(f"Found {vulnerability_count} vulnerabilities in dependencies")
            return True, {"vulnerabilities": vulnerabilities}
        except json.JSONDecodeError:
            print(f"Error parsing safety output: {result.stdout}")
            return False, {"error": "Invalid JSON output from safety"}
    except Exception as e:
        print(f"Error running safety: {e}")
        return False, {"error": str(e)}


def check_code() -> Tuple[bool, Dict]:
    """Check Python code using bandit"""
    print("Checking Python code...")
    
    # Ensure bandit is installed
    try:
        subprocess.run(["bandit", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Installing bandit...")
        subprocess.run([sys.executable, "-m", "pip", "install", "bandit"], check=True)
    
    # Run bandit
    try:
        result = subprocess.run(
            ["bandit", "-r", "src", "backend", "-f", "json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        
        # bandit returns exit code 0 when no issues are found
        # and non-zero when issues are found or errors occur
        if result.returncode > 1:
            print(f"Error running bandit: {result.stderr}")
            return False, {"error": result.stderr}
        
        # Parse JSON output
        try:
            report = json.loads(result.stdout)
            issue_count = len(report.get("results", []))
            print(f"Found {issue_count} security issues in code")
            return True, {"report": report}
        except json.JSONDecodeError:
            print(f"Error parsing bandit output: {result.stdout}")
            return False, {"error": "Invalid JSON output from bandit"}
    except Exception as e:
        print(f"Error running bandit: {e}")
        return False, {"error": str(e)}


def check_docker_image() -> Tuple[bool, Dict]:
    """Check Docker image using trivy"""
    print("Checking Docker image...")
    
    # Ensure trivy is installed or download it
    try:
        result = subprocess.run(["trivy", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            raise FileNotFoundError("Trivy not found")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Installing trivy...")
        try:
            # For Linux
            subprocess.run(
                ["curl", "-sfL", "https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh"],
                stdout=subprocess.PIPE,
                check=True
            )
            subprocess.run(["sh", "-s", "--", "-b", "/usr/local/bin", "v0.20.2"], check=True)
        except:
            print("Could not install Trivy automatically. Please install manually: https://aquasecurity.github.io/trivy/latest/getting-started/installation/")
            return False, {"error": "Trivy installation failed"}
    
    # Build the Docker image
    print("Building Docker image...")
    try:
        subprocess.run(
            ["docker", "build", "-t", "geoai-security-scan:latest", "-f", "Dockerfile.backend", "."],
            check=True,
        )
    except subprocess.SubprocessError as e:
        print(f"Error building Docker image: {e}")
        return False, {"error": f"Docker build failed: {e}"}
    
    # Run trivy
    print("Scanning Docker image with Trivy...")
    try:
        result = subprocess.run(
            ["trivy", "image", "--format", "json", "geoai-security-scan:latest"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        
        # Parse JSON output
        try:
            report = json.loads(result.stdout)
            
            # Count vulnerabilities
            vulnerability_count = 0
            for res in report.get("Results", []):
                vulnerability_count += len(res.get("Vulnerabilities", []))
            
            print(f"Found {vulnerability_count} vulnerabilities in Docker image")
            return True, {"report": report}
        except json.JSONDecodeError:
            print(f"Error parsing trivy output")
            return False, {"error": "Invalid JSON output from trivy"}
    except Exception as e:
        print(f"Error running trivy: {e}")
        return False, {"error": str(e)}


def generate_report(
    output_dir: str, deps_result: Dict, code_result: Dict, docker_result: Dict
) -> bool:
    """Generate security report"""
    print(f"Generating security report in {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create JSON reports
    if "vulnerabilities" in deps_result:
        with open(os.path.join(output_dir, "deps-report.json"), "w") as f:
            json.dump(deps_result["vulnerabilities"], f, indent=2)
    
    if "report" in code_result:
        with open(os.path.join(output_dir, "code-report.json"), "w") as f:
            json.dump(code_result["report"], f, indent=2)
    
    if "report" in docker_result:
        with open(os.path.join(output_dir, "docker-report.json"), "w") as f:
            json.dump(docker_result["report"], f, indent=2)
    
    # Create summary report
    with open(os.path.join(output_dir, "summary-report.md"), "w") as f:
        f.write("# Security Scan Summary\n\n")
        
        # Dependencies section
        f.write("## Python Dependencies\n\n")
        if "vulnerabilities" in deps_result:
            vulnerabilities = deps_result["vulnerabilities"]
            if vulnerabilities:
                f.write(f"⚠️ Found {len(vulnerabilities)} vulnerabilities\n\n")
                f.write("| Package | Vulnerable Version | Advisory |\n")
                f.write("|---------|-------------------|----------|\n")
                
                for vuln in vulnerabilities[:10]:  # Show top 10
                    package = vuln[0] if len(vuln) > 0 else "Unknown"
                    version = vuln[1] if len(vuln) > 1 else "Unknown"
                    advisory = vuln[4] if len(vuln) > 4 else "Unknown"
                    f.write(f"| {package} | {version} | {advisory} |\n")
                
                if len(vulnerabilities) > 10:
                    f.write(f"\n... and {len(vulnerabilities) - 10} more vulnerabilities.\n")
            else:
                f.write("✅ No vulnerabilities found\n")
        else:
            f.write("❌ Error running dependency check\n")
            if "error" in deps_result:
                f.write(f"Error: {deps_result['error']}\n")
        
        # Code section
        f.write("\n## Python Code\n\n")
        if "report" in code_result:
            report = code_result["report"]
            results = report.get("results", [])
            if results:
                f.write(f"⚠️ Found {len(results)} security issues\n\n")
                
                # Group by severity
                severity_count = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                for result in results:
                    severity = result.get("issue_severity", "").upper()
                    if severity in severity_count:
                        severity_count[severity] += 1
                
                f.write("| Severity | Count |\n")
                f.write("|----------|-------|\n")
                for severity, count in severity_count.items():
                    f.write(f"| {severity} | {count} |\n")
                
                f.write("\n### Top Issues\n\n")
                
                # Sort by severity
                sorted_results = sorted(
                    results, 
                    key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x.get("issue_severity", "").upper(), 3)
                )
                
                for i, result in enumerate(sorted_results[:5]):  # Show top 5
                    severity = result.get("issue_severity", "")
                    issue = result.get("issue_text", "")
                    filename = result.get("filename", "")
                    line = result.get("line_number", "")
                    
                    f.write(f"**{i+1}. [{severity}] {issue}**  \n")
                    f.write(f"File: {filename}:{line}\n\n")
            else:
                f.write("✅ No security issues found\n")
        else:
            f.write("❌ Error running code security check\n")
            if "error" in code_result:
                f.write(f"Error: {code_result['error']}\n")
        
        # Docker section
        f.write("\n## Docker Image\n\n")
        if "report" in docker_result:
            report = docker_result["report"]
            results = report.get("Results", [])
            
            # Count vulnerabilities
            vulnerability_count = 0
            severity_count = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
            
            for result in results:
                vulns = result.get("Vulnerabilities", [])
                vulnerability_count += len(vulns)
                
                for vuln in vulns:
                    severity = vuln.get("Severity", "").upper()
                    if severity in severity_count:
                        severity_count[severity] += 1
            
            if vulnerability_count > 0:
                f.write(f"⚠️ Found {vulnerability_count} vulnerabilities\n\n")
                
                f.write("| Severity | Count |\n")
                f.write("|----------|-------|\n")
                for severity, count in severity_count.items():
                    if count > 0:
                        f.write(f"| {severity} | {count} |\n")
                
                f.write("\n### Critical Vulnerabilities\n\n")
                
                # Find critical vulnerabilities
                critical_vulns = []
                for result in results:
                    for vuln in result.get("Vulnerabilities", []):
                        if vuln.get("Severity", "").upper() == "CRITICAL":
                            critical_vulns.append(vuln)
                
                if critical_vulns:
                    f.write("| CVE ID | Package | Fixed Version |\n")
                    f.write("|--------|---------|---------------|\n")
                    
                    for vuln in critical_vulns[:10]:  # Show top 10
                        cve_id = vuln.get("VulnerabilityID", "")
                        package = vuln.get("PkgName", "")
                        fixed = vuln.get("FixedVersion", "Not available")
                        
                        f.write(f"| {cve_id} | {package} | {fixed} |\n")
                    
                    if len(critical_vulns) > 10:
                        f.write(f"\n... and {len(critical_vulns) - 10} more critical vulnerabilities.\n")
                else:
                    f.write("No critical vulnerabilities found.\n")
            else:
                f.write("✅ No vulnerabilities found\n")
        else:
            f.write("❌ Error running Docker image security check\n")
            if "error" in docker_result:
                f.write(f"Error: {docker_result['error']}\n")
        
        # Recommendations section
        f.write("\n## Recommendations\n\n")
        f.write("1. Update vulnerable dependencies to their latest secure versions\n")
        f.write("2. Review and fix code security issues, focusing on high severity issues first\n")
        f.write("3. Consider using a more secure base image for Docker or minimize installed packages\n")
        f.write("4. Run security scans regularly as part of your development workflow\n")
    
    print(f"Security report generated: {os.path.join(output_dir, 'summary-report.md')}")
    return True


def main() -> int:
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Default to --all if no specific checks are requested
    if not (args.deps or args.code or args.docker):
        args.all = True
    
    # Results
    deps_result: Dict = {}
    code_result: Dict = {}
    docker_result: Dict = {}
    
    # Run checks
    if args.all or args.deps:
        deps_success, deps_result = check_dependencies()
        if not deps_success and args.verbose:
            print("Dependency check failed")
    
    if args.all or args.code:
        code_success, code_result = check_code()
        if not code_success and args.verbose:
            print("Code check failed")
    
    if args.all or args.docker:
        docker_success, docker_result = check_docker_image()
        if not docker_success and args.verbose:
            print("Docker image check failed")
    
    # Generate report
    generate_report(args.output, deps_result, code_result, docker_result)
    
    # Determine exit code (0 for success)
    has_vulnerabilities = (
        "vulnerabilities" in deps_result and deps_result["vulnerabilities"]
        or "report" in code_result and code_result["report"].get("results", [])
        or "report" in docker_result and any(
            len(res.get("Vulnerabilities", [])) > 0 for res in docker_result["report"].get("Results", [])
        )
    )
    
    return 1 if has_vulnerabilities else 0


if __name__ == "__main__":
    sys.exit(main())