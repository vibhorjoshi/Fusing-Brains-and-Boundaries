import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8081
FRONTEND_DIR = Path('d:/geo ai research paper/frontend')

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        kwargs['directory'] = str(FRONTEND_DIR)
        super().__init__(*args, **kwargs)
        
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_GET(self):
        # Let Next.js handle routing
        super().do_GET()

# Make sure the frontend directory exists
if not FRONTEND_DIR.exists():
    print(f"Error: Frontend directory {FRONTEND_DIR} does not exist.")
    exit(1)

Handler = MyHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"🌐 Serving GeoAI Frontend at port {PORT}")
    print(f"📍 Access the frontend: http://localhost:{PORT}")
    print(f"📡 API: http://localhost:8000")
    print(f"�️ Streamlit Dashboard: http://localhost:8502")
    
    # Open browser automatically
    webbrowser.open(f'http://localhost:{PORT}')
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()