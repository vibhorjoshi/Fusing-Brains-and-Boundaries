import http.server
import socketserver
import webbrowser
import os

PORT = 8081

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_GET(self):
        # Redirect root to the enhanced USA dashboard
        if self.path == '/' or self.path == '/index.html':
            self.path = '/enhanced_usa_agricultural_dashboard.html'
        elif self.path == '/dashboard' or self.path == '/live':
            self.path = '/enhanced_usa_agricultural_dashboard.html'
        
        super().do_GET()

# Change to the directory containing the HTML file
os.chdir('d:/geo ai research paper')

Handler = MyHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"🌐 Serving Real USA Agricultural Dashboard at port {PORT}")
    print(f"📍 Enhanced Dashboard: http://localhost:{PORT}/enhanced_usa_agricultural_dashboard.html")
    print(f"� Enhanced API: http://localhost:8007")
    print(f"🐳 Redis Storage: localhost:6379")
    
    # Open browser automatically
    webbrowser.open(f'http://localhost:{PORT}/enhanced_usa_agricultural_dashboard.html')
    
    httpd.serve_forever()