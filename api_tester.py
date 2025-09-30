"""
Comprehensive API Endpoint Tester
Tests all Building Footprint AI endpoints systematically
"""

import requests
import json
import time
from datetime import datetime

class APITester:
    def __init__(self, base_url="http://127.0.0.1:8002"):
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None
        self.api_key = None
        
    def test_endpoint(self, method, endpoint, data=None, headers=None, files=None):
        """Generic endpoint tester"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers)
            elif method.upper() == "POST":
                if files:
                    response = self.session.post(url, data=data, files=files, headers=headers)
                else:
                    response = self.session.post(url, json=data, headers=headers)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, headers=headers)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": response.json() if response.content else None,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "status_code": None,
                "data": None,
                "error": str(e)
            }
    
    def print_result(self, title, result):
        """Print test result"""
        status_icon = "✅" if result["success"] and result["status_code"] < 400 else "❌"
        status_code = result["status_code"] if result["status_code"] else "ERROR"
        
        print(f"{status_icon} {title}: {status_code}")
        
        if result["error"]:
            print(f"   Error: {result['error']}")
        elif result["data"]:
            # Print key information from response
            if isinstance(result["data"], dict):
                if "message" in result["data"]:
                    print(f"   Message: {result['data']['message']}")
                if "status" in result["data"]:
                    print(f"   Status: {result['data']['status']}")
                if "token_type" in result["data"]:
                    print(f"   Token Type: {result['data']['token_type']}")
        print()
    
    def run_all_tests(self):
        """Run comprehensive API tests"""
        print("🏢 BUILDING FOOTPRINT AI - COMPREHENSIVE API TEST")
        print("=" * 60)
        print(f"📡 Testing Server: {self.base_url}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 1. Basic Endpoints
        print("\n1️⃣ BASIC ENDPOINTS")
        print("-" * 30)
        
        result = self.test_endpoint("GET", "/")
        self.print_result("Root Endpoint", result)
        
        result = self.test_endpoint("GET", "/health")
        self.print_result("Health Check", result)
        
        # 2. Authentication Endpoints  
        print("\n2️⃣ AUTHENTICATION ENDPOINTS")
        print("-" * 30)
        
        # Register new user
        user_data = {
            "username": "testuser123",
            "email": "test@example.com",
            "password": "testpass123",
            "role": "USER"
        }
        
        result = self.test_endpoint("POST", "/api/v1/auth/register", user_data)
        self.print_result("User Registration", result)
        
        # Login user
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        
        result = self.test_endpoint("POST", "/api/v1/auth/login", login_data)
        self.print_result("User Login (Admin)", result)
        
        if result["success"] and result["data"]:
            self.token = result["data"].get("access_token")
            print(f"   🔑 Token obtained: {self.token[:20]}...")
        
        # Test with token header
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            
            result = self.test_endpoint("POST", "/api/v1/auth/refresh-token", headers=headers)
            self.print_result("Token Refresh", result)
            
            result = self.test_endpoint("POST", "/api/v1/auth/generate-api-key", headers=headers)
            self.print_result("API Key Generation", result)
            
            if result["success"] and result["data"]:
                self.api_key = result["data"].get("api_key")
                print(f"   🗝️ API Key obtained: {self.api_key[:20]}...")
        
        # 3. ML Processing Endpoints
        print("\n3️⃣ ML PROCESSING ENDPOINTS") 
        print("-" * 30)
        
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            
            # Extract buildings
            ml_data = {
                "image_url": "https://example.com/satellite.jpg",
                "coordinates": [-122.4, 37.8, -122.3, 37.9],
                "model_type": "mask_rcnn",
                "apply_regularization": True
            }
            
            result = self.test_endpoint("POST", "/api/v1/ml-processing/extract-buildings", ml_data, headers)
            self.print_result("Extract Buildings", result)
            
            job_id = None
            if result["success"] and result["data"]:
                job_id = result["data"].get("job_id")
                print(f"   📋 Job ID: {job_id}")
            
            # Process state (using form data)
            import requests
            try:
                state_response = requests.post(
                    f"{self.base_url}/api/v1/ml-processing/process-state",
                    data={"state_name": "California"},
                    headers=headers
                )
                state_result = {
                    "success": True,
                    "status_code": state_response.status_code,
                    "data": state_response.json() if state_response.content else None,
                    "error": None
                }
            except Exception as e:
                state_result = {
                    "success": False,
                    "status_code": None,
                    "data": None,
                    "error": str(e)
                }
            
            self.print_result("Process State", state_result)
            
            # Check task status
            if job_id:
                result = self.test_endpoint("GET", f"/api/v1/ml-processing/task-status/{job_id}")
                self.print_result("Task Status Check", result)
        
        # 4. Building Data Endpoints
        print("\n4️⃣ BUILDING DATA ENDPOINTS")
        print("-" * 30)
        
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            
            result = self.test_endpoint("GET", "/api/v1/buildings", headers=headers)
            self.print_result("List Buildings", result)
            
            building_id = None
            if result["success"] and result["data"] and result["data"].get("buildings"):
                building_id = result["data"]["buildings"][0]["id"]
                print(f"   🏗️ Testing with building ID: {building_id}")
                
                # Get specific building
                result = self.test_endpoint("GET", f"/api/v1/buildings/{building_id}", headers=headers)
                self.print_result("Get Building Details", result)
                
                # Update building
                update_data = {"confidence": 0.95, "notes": "Updated via API test"}
                result = self.test_endpoint("PUT", f"/api/v1/buildings/{building_id}", update_data, headers)
                self.print_result("Update Building", result)
            
            # Building statistics
            result = self.test_endpoint("GET", "/api/v1/buildings/statistics/overview", headers=headers)
            self.print_result("Building Statistics", result)
        
        # 5. Job Management Endpoints
        print("\n5️⃣ JOB MANAGEMENT ENDPOINTS")
        print("-" * 30)
        
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            
            result = self.test_endpoint("GET", "/api/v1/jobs", headers=headers)
            self.print_result("List Jobs", result)
            
            job_id = None
            if result["success"] and result["data"] and result["data"].get("jobs"):
                job_id = result["data"]["jobs"][0]["id"]
                print(f"   📋 Testing with job ID: {job_id}")
                
                # Get job details
                result = self.test_endpoint("GET", f"/api/v1/jobs/{job_id}", headers=headers)
                self.print_result("Get Job Details", result)
            
            # Job statistics
            result = self.test_endpoint("GET", "/api/v1/jobs/statistics", headers=headers)
            self.print_result("Job Statistics", result)
        
        # 6. File Management Endpoints
        print("\n6️⃣ FILE MANAGEMENT ENDPOINTS")
        print("-" * 30)
        
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            
            result = self.test_endpoint("GET", "/api/v1/files", headers=headers)
            self.print_result("List Files", result)
            
            file_id = None
            if result["success"] and result["data"] and result["data"].get("files"):
                file_id = result["data"]["files"][0]["id"]
                print(f"   📁 Testing with file ID: {file_id}")
                
                # Download file
                result = self.test_endpoint("GET", f"/api/v1/files/{file_id}/download", headers=headers)
                self.print_result("Download File", result)
        
        # 7. Admin Endpoints (Admin only)
        print("\n7️⃣ ADMIN ENDPOINTS")
        print("-" * 30)
        
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            
            result = self.test_endpoint("GET", "/api/v1/admin/dashboard", headers=headers)
            self.print_result("Admin Dashboard", result)
            
            result = self.test_endpoint("GET", "/api/v1/admin/users", headers=headers)
            self.print_result("List All Users", result)
            
            result = self.test_endpoint("GET", "/api/v1/admin/system/health", headers=headers)
            self.print_result("Detailed Health Check", result)
            
            result = self.test_endpoint("POST", "/api/v1/admin/cleanup", headers=headers)
            self.print_result("System Cleanup", result)
        
        print("\n" + "=" * 60)
        print("🎉 API TESTING COMPLETED!")
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

if __name__ == "__main__":
    # Wait for server to be ready
    print("⏳ Waiting for server to start...")
    time.sleep(5)
    
    # Run tests
    tester = APITester()
    tester.run_all_tests()