#!/usr/bin/env python3
"""
MaskRCNN Service - Handles 3D mask creation for agricultural detection
This service connects to Redis and processes images to create 3D masks
"""

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
import redis
import cv2
from flask import Flask, request, jsonify
import threading
from datetime import datetime
import io
import base64

# Configure TensorFlow to use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU for processing.")

# Initialize Flask app
app = Flask(__name__)

# Initialize Redis client
redis_host = os.environ.get('REDIS_HOST', 'localhost')
redis_port = int(os.environ.get('REDIS_PORT', 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port)
print(f"Connected to Redis at {redis_host}:{redis_port}")

class MaskRCNNService:
    """Handles 3D mask creation using MaskRCNN"""
    
    def __init__(self):
        print("MaskRCNN service initializing...")
        # In a production environment, we would load the actual MaskRCNN model here
        # For this example, we'll use a simulated model
        time.sleep(2)  # Simulate model loading time
        print("MaskRCNN service initialized and ready")
        
    def create_masks(self, image):
        """Create 3D masks for agricultural detection"""
        try:
            if image is None:
                return None
                
            # For production, this would use the actual MaskRCNN model
            # Here we'll simulate mask generation
            height, width = image.shape[:2]
            
            # Generate random binary masks (would be MaskRCNN output in production)
            num_masks = np.random.randint(3, 8)
            masks = []
            
            for i in range(num_masks):
                # Create a random polygon-like mask
                mask = np.zeros((height, width), dtype=np.uint8)
                
                # Generate random polygon points
                num_points = np.random.randint(6, 15)
                points = np.random.randint(0, min(height, width), (num_points, 2))
                
                # Draw filled polygon
                cv2.fillPoly(mask, [points], 1)
                
                masks.append(mask)
            
            # Stack masks into 3D array
            masks_3d = np.stack(masks, axis=2)
            print(f"Created {num_masks} masks with shape {masks_3d.shape}")
            
            return masks_3d
            
        except Exception as e:
            print(f"Error in MaskRCNN processing: {e}")
            return None

# Initialize the MaskRCNN service
maskrcnn_service = MaskRCNNService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "maskrcnn"}), 200

@app.route('/process', methods=['POST'])
def process_image():
    """Process an image to generate masks"""
    try:
        # Get image data from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image with MaskRCNN
        masks = maskrcnn_service.create_masks(image)
        
        if masks is None:
            return jsonify({"error": "Failed to generate masks"}), 500
        
        # Encode masks as base64
        # First convert to uint8 if not already
        if masks.dtype != np.uint8:
            masks = (masks * 255).astype(np.uint8)
        
        # Compress and encode masks
        success, encoded_masks = cv2.imencode('.png', masks)
        if not success:
            return jsonify({"error": "Failed to encode masks"}), 500
        
        masks_base64 = base64.b64encode(encoded_masks).decode('utf-8')
        
        # Store in Redis if requested
        if 'job_id' in data:
            job_id = data['job_id']
            redis_client.setex(
                f"maskrcnn:result:{job_id}",
                3600,  # 1 hour expiration
                masks_base64
            )
            return jsonify({
                "status": "success", 
                "job_id": job_id,
                "message": "Masks generated and stored in Redis"
            }), 200
        
        # Return masks directly if no job_id
        return jsonify({
            "status": "success",
            "masks": masks_base64,
            "shape": masks.shape
        }), 200
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

def redis_listener():
    """Listen for processing jobs in Redis"""
    pubsub = redis_client.pubsub()
    pubsub.subscribe('maskrcnn:jobs')
    
    print("Redis listener started, waiting for jobs...")
    
    for message in pubsub.listen():
        if message['type'] != 'message':
            continue
        
        try:
            job_data = json.loads(message['data'])
            job_id = job_data.get('job_id')
            image_key = job_data.get('image_key')
            
            if not job_id or not image_key:
                continue
                
            print(f"Processing job {job_id}")
            
            # Get image data from Redis
            image_data = redis_client.get(image_key)
            if not image_data:
                print(f"Image data not found for key: {image_key}")
                continue
                
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process with MaskRCNN
            masks = maskrcnn_service.create_masks(image)
            
            if masks is None:
                redis_client.setex(
                    f"maskrcnn:result:{job_id}",
                    3600,
                    json.dumps({"status": "error", "message": "Failed to generate masks"})
                )
                continue
            
            # Compress and encode masks
            success, encoded_masks = cv2.imencode('.png', masks)
            if not success:
                redis_client.setex(
                    f"maskrcnn:result:{job_id}",
                    3600,
                    json.dumps({"status": "error", "message": "Failed to encode masks"})
                )
                continue
                
            masks_base64 = base64.b64encode(encoded_masks).decode('utf-8')
            
            # Store result in Redis
            redis_client.setex(
                f"maskrcnn:result:{job_id}",
                3600,  # 1 hour expiration
                json.dumps({
                    "status": "success",
                    "masks": masks_base64,
                    "shape": list(masks.shape)
                })
            )
            
            print(f"Job {job_id} completed successfully")
            
        except Exception as e:
            print(f"Error processing job: {e}")

if __name__ == "__main__":
    # Start Redis listener in a background thread
    listener_thread = threading.Thread(target=redis_listener)
    listener_thread.daemon = True
    listener_thread.start()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)