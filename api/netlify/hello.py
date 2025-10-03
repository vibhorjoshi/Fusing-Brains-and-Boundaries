from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests to the Netlify Function"""
        try:
            # Set response headers
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Return a basic API response
            response = {
                "status": "healthy",
                "service": "geoai-api-netlify-function",
                "version": "1.0.0",
                "message": "GeoAI API is running on Netlify Functions",
                "path": self.path,
                "limitations": "Limited functionality due to Netlify Functions constraints"
            }
            
            # Write the response as JSON
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return
            
        except Exception as e:
            # Handle errors
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "message": str(e)
            }
            
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
            return
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return