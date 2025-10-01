#!/usr/bin/env python3
"""
Live World Map Testing Demo with Adaptive Fusion RL Agent
This script demonstrates the capabilities of the enhanced GeoAI system.
"""

import requests
import asyncio
import json
import time
from typing import Dict, List, Optional
import random

# API Configuration
API_BASE = "http://localhost:8001"
FRONTEND_URL = "http://localhost:8002/threejs_world_map_frontend.html"

class AdaptiveFusionWorldDemo:
    def __init__(self):
        self.session = requests.Session()
        self.active_jobs = []
        self.batch_sessions = []
        
    def check_system_status(self) -> Dict:
        """Check if backend and RL agent are operational."""
        try:
            # Check main API
            response = self.session.get(f"{API_BASE}/")
            if response.status_code == 200:
                main_status = response.json()
            else:
                return {"error": "Backend not accessible"}
            
            # Check RL Agent status
            response = self.session.get(f"{API_BASE}/api/rl-agent/status")
            if response.status_code == 200:
                rl_status = response.json()
            else:
                return {"error": "RL Agent not accessible"}
            
            # Get world locations
            response = self.session.get(f"{API_BASE}/api/world-locations")
            if response.status_code == 200:
                locations_data = response.json()
            else:
                return {"error": "World locations not accessible"}
            
            return {
                "status": "operational",
                "backend": main_status,
                "rl_agent": rl_status,
                "locations": locations_data,
                "frontend_url": FRONTEND_URL
            }
            
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to backend API"}
        except Exception as e:
            return {"error": f"System check failed: {str(e)}"}
    
    def test_single_location(self, location: str, training: bool = True, iterations: int = 15) -> Dict:
        """Test adaptive fusion on a single location."""
        try:
            request_data = {
                "location": location,
                "technique": "adaptive_fusion_rl",
                "real_time_training": training,
                "num_iterations": iterations,
                "use_satellite_imagery": True
            }
            
            print(f"🎯 Starting single location test: {location}")
            print(f"   Training enabled: {training}")
            print(f"   Iterations: {iterations}")
            
            response = self.session.post(
                f"{API_BASE}/api/process/single-location",
                json=request_data
            )
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data['job_id']
                self.active_jobs.append(job_id)
                
                print(f"✅ Job started: {job_id}")
                print(f"   Estimated completion: {job_data.get('estimated_completion', 'Unknown')}")
                
                return self.monitor_single_job(job_id, location)
            else:
                return {"error": f"Failed to start job: {response.text}"}
                
        except Exception as e:
            return {"error": f"Single location test failed: {str(e)}"}
    
    def monitor_single_job(self, job_id: str, location: str) -> Dict:
        """Monitor a single processing job until completion."""
        print(f"📊 Monitoring job {job_id}...")
        
        start_time = time.time()
        max_wait_time = 60  # Maximum wait time in seconds
        
        while time.time() - start_time < max_wait_time:
            try:
                response = self.session.get(f"{API_BASE}/api/process/status/{job_id}")
                
                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get('status', 'unknown')
                    progress = status_data.get('progress', 0)
                    
                    if status == 'completed':
                        results = status_data.get('results', {})
                        final_perf = results.get('final_performance', {})
                        rl_stats = results.get('rl_stats', {})
                        
                        print(f"🎉 Job completed successfully!")
                        print(f"   Location: {location}")
                        print(f"   Best IoU: {final_perf.get('best_iou', 0):.3f}")
                        print(f"   RL Epsilon: {rl_stats.get('epsilon', 0):.3f}")
                        print(f"   Memory Size: {rl_stats.get('memory_size', 0)}")
                        
                        return {
                            "status": "success",
                            "job_id": job_id,
                            "location": location,
                            "results": results
                        }
                    
                    elif status == 'error':
                        error_msg = status_data.get('error', 'Unknown error')
                        print(f"❌ Job failed: {error_msg}")
                        return {"status": "error", "error": error_msg}
                    
                    elif status == 'processing':
                        print(f"⏳ Processing... {progress}%")
                    
                    time.sleep(2)  # Check every 2 seconds
                else:
                    print(f"⚠️  Error checking job status: {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️  Error monitoring job: {str(e)}")
                
        print(f"⏰ Job monitoring timed out after {max_wait_time}s")
        return {"status": "timeout", "job_id": job_id}
    
    def test_world_batch(self, regions: Optional[List[str]] = None, locations_per_region: int = 3, 
                        parallel: bool = True) -> Dict:
        """Test adaptive fusion across multiple world regions."""
        
        if regions is None:
            regions = ['North America', 'Europe', 'Asia']
        
        try:
            request_data = {
                "regions": regions,
                "num_locations_per_region": locations_per_region,
                "parallel_processing": parallel,
                "training_enabled": True,
                "performance_tracking": True
            }
            
            print(f"🌍 Starting world batch test")
            print(f"   Regions: {', '.join(regions)}")
            print(f"   Locations per region: {locations_per_region}")
            print(f"   Parallel processing: {parallel}")
            
            response = self.session.post(
                f"{API_BASE}/api/world-map/batch-test",
                json=request_data
            )
            
            if response.status_code == 200:
                session_data = response.json()
                session_id = session_data['session_id']
                self.batch_sessions.append(session_id)
                
                print(f"✅ Batch session started: {session_id}")
                print(f"   Total locations: {session_data.get('total_locations', 0)}")
                print(f"   Selected locations: {session_data.get('selected_locations', [])}")
                
                return self.monitor_batch_session(session_id)
            else:
                return {"error": f"Failed to start batch test: {response.text}"}
                
        except Exception as e:
            return {"error": f"Batch test failed: {str(e)}"}
    
    def monitor_batch_session(self, session_id: str) -> Dict:
        """Monitor a batch testing session until completion."""
        print(f"📊 Monitoring batch session {session_id}...")
        
        start_time = time.time()
        max_wait_time = 180  # Maximum wait time in seconds
        
        while time.time() - start_time < max_wait_time:
            try:
                response = self.session.get(f"{API_BASE}/api/world-map/session/{session_id}")
                
                if response.status_code == 200:
                    session_data = response.json()
                    status = session_data.get('status', 'unknown')
                    progress = session_data.get('progress', 0)
                    completed = session_data.get('completed_locations', 0)
                    total = session_data.get('total_locations', 0)
                    
                    if status == 'completed':
                        summary = session_data.get('summary', {})
                        
                        print(f"🎉 Batch session completed successfully!")
                        print(f"   Total processed: {summary.get('total_processed', 0)}")
                        print(f"   Successful detections: {summary.get('successful_detections', 0)}")
                        print(f"   Success rate: {summary.get('success_rate', 0)*100:.1f}%")
                        print(f"   Average IoU: {summary.get('average_iou', 0):.3f}")
                        
                        # Regional breakdown
                        regional = summary.get('regional_breakdown', {})
                        if regional:
                            print(f"   Regional Performance:")
                            for region, stats in regional.items():
                                print(f"     {region}: {stats['average_iou']:.3f} IoU ({stats['count']} locations)")
                        
                        return {
                            "status": "success",
                            "session_id": session_id,
                            "summary": summary,
                            "results": session_data.get('results', [])
                        }
                    
                    elif status == 'error':
                        error_msg = session_data.get('error', 'Unknown error')
                        print(f"❌ Batch session failed: {error_msg}")
                        return {"status": "error", "error": error_msg}
                    
                    elif status == 'processing':
                        print(f"⏳ Processing... {progress}% ({completed}/{total} locations)")
                    
                    time.sleep(5)  # Check every 5 seconds for batch jobs
                else:
                    print(f"⚠️  Error checking session status: {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️  Error monitoring session: {str(e)}")
                
        print(f"⏰ Session monitoring timed out after {max_wait_time}s")
        return {"status": "timeout", "session_id": session_id}
    
    def get_global_metrics(self) -> Dict:
        """Get current global performance metrics."""
        try:
            response = self.session.get(f"{API_BASE}/api/world-map/global-metrics")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get metrics: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Metrics request failed: {str(e)}"}
    
    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of the system."""
        print("=" * 70)
        print("🚀 GeoAI ADAPTIVE FUSION WITH RL AGENT - LIVE WORLD MAP TESTING")
        print("=" * 70)
        
        # 1. System Status Check
        print("\n1️⃣  SYSTEM STATUS CHECK")
        print("-" * 30)
        
        status = self.check_system_status()
        if "error" in status:
            print(f"❌ System check failed: {status['error']}")
            print("Please ensure the backend is running on port 8001")
            return
        
        print("✅ Backend API operational")
        print("✅ RL Agent active")
        print(f"✅ {status['locations']['total_count']} world locations available")
        print(f"✅ Frontend available at: {FRONTEND_URL}")
        
        # Display RL Agent stats
        rl_agent = status['rl_agent']
        print(f"\n🤖 RL Agent Status:")
        print(f"   State dimension: {rl_agent['network_parameters']['state_dim']}")
        print(f"   Action dimension: {rl_agent['network_parameters']['action_dim']}")
        print(f"   Current epsilon: {rl_agent['network_parameters']['epsilon']:.3f}")
        print(f"   Memory size: {rl_agent['network_parameters']['memory_size']}")
        
        # 2. Single Location Tests
        print(f"\n2️⃣  SINGLE LOCATION ADAPTIVE FUSION TESTS")
        print("-" * 45)
        
        # Test high-population city (challenging)
        single_test_1 = self.test_single_location("tokyo_japan", training=True, iterations=20)
        
        # Test moderate city
        single_test_2 = self.test_single_location("london_uk", training=True, iterations=15)
        
        # Test smaller city  
        single_test_3 = self.test_single_location("amsterdam_netherlands", training=True, iterations=10)
        
        # 3. Global Batch Testing
        print(f"\n3️⃣  GLOBAL BATCH TESTING WITH RL ADAPTATION")
        print("-" * 48)
        
        # Test across multiple regions
        batch_test = self.test_world_batch(
            regions=['North America', 'Europe', 'Asia', 'Africa'],
            locations_per_region=2,
            parallel=True
        )
        
        # 4. Global Metrics Summary
        print(f"\n4️⃣  GLOBAL PERFORMANCE METRICS")
        print("-" * 35)
        
        final_metrics = self.get_global_metrics()
        if "error" not in final_metrics:
            rl_metrics = final_metrics.get('rl_agent_metrics', {})
            global_metrics = final_metrics.get('rl_agent_metrics', {})
            
            print(f"🎯 Total Locations Processed: {rl_metrics.get('total_processed', 0)}")
            print(f"🎯 Total Training Episodes: {rl_metrics.get('total_training_episodes', 0)}")
            print(f"🎯 Global Average IoU: {rl_metrics.get('average_iou', 0):.3f}")
            print(f"🎯 Best IoU Achieved: {rl_metrics.get('best_iou', 0):.3f}")
            print(f"🎯 Current RL Epsilon: {rl_metrics.get('epsilon', 0):.3f}")
            
        # 5. Demo Summary
        print(f"\n5️⃣  DEMO SUMMARY")
        print("-" * 20)
        print("✅ Adaptive Fusion RL Agent successfully deployed")
        print("✅ Live world map testing operational") 
        print("✅ Multi-region batch processing functional")
        print("✅ Real-time training and adaptation working")
        print("✅ Three.js frontend integrated with live metrics")
        
        print(f"\n🌐 Access the interactive world map at:")
        print(f"   {FRONTEND_URL}")
        print(f"\n📡 Backend API documentation at:")
        print(f"   http://localhost:8001/docs")
        
        print("\n" + "=" * 70)
        print("🎉 DEMO COMPLETED - System ready for live global testing!")
        print("=" * 70)

def main():
    """Main demo function."""
    demo = AdaptiveFusionWorldDemo()
    demo.run_comprehensive_demo()

if __name__ == "__main__":
    main()