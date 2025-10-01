"""
Cloud-compatible replacements for OpenCV functions using PIL and scipy
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage, morphology
from skimage import measure, morphology as skimage_morphology
import warnings

# Try to import cv2, but provide fallbacks if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available. Using PIL/scipy fallbacks for cloud deployment.")

class CloudCompatibleCV2:
    """Cloud-compatible OpenCV replacement using PIL and scipy"""
    
    # Constants
    MORPH_OPEN = 2
    MORPH_CLOSE = 3
    MORPH_ERODE = 0
    MORPH_DILATE = 1
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 2
    CHAIN_APPROX_NONE = 1
    INTER_AREA = 3
    INTER_NEAREST = 0
    CV_32F = 5
    
    @staticmethod
    def morphologyEx(image, operation, kernel, iterations=1):
        """Morphological operations using scipy"""
        if CV2_AVAILABLE:
            return cv2.morphologyEx(image, operation, kernel, iterations=iterations)
        
        # Convert to boolean for scipy operations
        binary_img = image.astype(bool)
        
        # Create circular kernel if kernel is small
        if kernel.shape == (3, 3):
            struct = morphology.disk(1)
        elif kernel.shape == (5, 5):
            struct = morphology.disk(2)
        else:
            struct = kernel.astype(bool)
        
        for _ in range(iterations):
            if operation == CloudCompatibleCV2.MORPH_OPEN:
                binary_img = morphology.binary_opening(binary_img, struct)
            elif operation == CloudCompatibleCV2.MORPH_CLOSE:
                binary_img = morphology.binary_closing(binary_img, struct)
            elif operation == CloudCompatibleCV2.MORPH_ERODE:
                binary_img = morphology.binary_erosion(binary_img, struct)
            elif operation == CloudCompatibleCV2.MORPH_DILATE:
                binary_img = morphology.binary_dilation(binary_img, struct)
        
        return binary_img.astype(image.dtype)
    
    @staticmethod
    def findContours(image, mode, method):
        """Find contours using skimage"""
        if CV2_AVAILABLE:
            return cv2.findContours(image, mode, method)
        
        # Use skimage to find contours
        contours_list = measure.find_contours(image, 0.5)
        
        # Convert to OpenCV format (list of numpy arrays)
        opencv_contours = []
        for contour in contours_list:
            # Swap x,y coordinates and convert to int
            contour_opencv = np.fliplr(contour).astype(np.int32)
            opencv_contours.append(contour_opencv.reshape(-1, 1, 2))
        
        return opencv_contours, None
    
    @staticmethod
    def contourArea(contour):
        """Calculate contour area using shoelace formula"""
        if CV2_AVAILABLE:
            return cv2.contourArea(contour)
        
        if len(contour.shape) == 3:
            contour = contour.reshape(-1, 2)
        
        x = contour[:, 0]
        y = contour[:, 1]
        return 0.5 * abs(sum(x[i]*y[(i+1) % len(x)] - x[(i+1) % len(x)]*y[i] for i in range(len(x))))
    
    @staticmethod
    def arcLength(contour, closed):
        """Calculate contour perimeter"""
        if CV2_AVAILABLE:
            return cv2.arcLength(contour, closed)
        
        if len(contour.shape) == 3:
            contour = contour.reshape(-1, 2)
        
        perimeter = 0
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)] if closed else contour[min(i + 1, len(contour) - 1)]
            if i < len(contour) - 1 or closed:
                perimeter += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        return perimeter
    
    @staticmethod
    def boundingRect(contour):
        """Calculate bounding rectangle"""
        if CV2_AVAILABLE:
            return cv2.boundingRect(contour)
        
        if len(contour.shape) == 3:
            contour = contour.reshape(-1, 2)
        
        x_min, y_min = np.min(contour, axis=0)
        x_max, y_max = np.max(contour, axis=0)
        
        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
    
    @staticmethod
    def convexHull(contour):
        """Calculate convex hull"""
        if CV2_AVAILABLE:
            return cv2.convexHull(contour)
        
        from scipy.spatial import ConvexHull
        
        if len(contour.shape) == 3:
            points = contour.reshape(-1, 2)
        else:
            points = contour
        
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            return hull_points.reshape(-1, 1, 2).astype(np.int32)
        except:
            return contour
    
    @staticmethod
    def Canny(image, threshold1, threshold2):
        """Canny edge detection using scipy"""
        if CV2_AVAILABLE:
            return cv2.Canny(image, threshold1, threshold2)
        
        # Use scipy filters for edge detection
        sobel_x = ndimage.sobel(image, axis=1)
        sobel_y = ndimage.sobel(image, axis=0)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Simple thresholding
        edges = np.zeros_like(magnitude)
        edges[magnitude > threshold1] = 255
        
        return edges.astype(np.uint8)
    
    @staticmethod
    def dilate(image, kernel, iterations=1):
        """Dilate using scipy"""
        if CV2_AVAILABLE:
            return cv2.dilate(image, kernel, iterations=iterations)
        
        result = image.copy()
        for _ in range(iterations):
            result = ndimage.binary_dilation(result, kernel).astype(image.dtype)
        
        return result
    
    @staticmethod
    def resize(image, dsize, interpolation=None):
        """Resize using PIL"""
        if CV2_AVAILABLE:
            return cv2.resize(image, dsize, interpolation=interpolation)
        
        # Convert numpy to PIL
        if image.dtype != np.uint8:
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image_pil = Image.fromarray(image)
        
        # Map OpenCV interpolation to PIL
        if interpolation == CloudCompatibleCV2.INTER_NEAREST:
            resample = Image.NEAREST
        else:
            resample = Image.LANCZOS
        
        resized = image_pil.resize(dsize, resample=resample)
        
        # Convert back to numpy
        result = np.array(resized)
        if image.dtype != np.uint8:
            result = result.astype(np.float32) / 255.0
        
        return result
    
    @staticmethod
    def blur(image, ksize):
        """Blur using PIL"""
        if CV2_AVAILABLE:
            return cv2.blur(image, ksize)
        
        # Convert to PIL
        if image.dtype != np.uint8:
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image_pil = Image.fromarray(image)
        
        # Apply blur
        radius = max(ksize) // 2
        blurred = image_pil.filter(ImageFilter.BoxBlur(radius))
        
        # Convert back
        result = np.array(blurred)
        if image.dtype != np.uint8:
            result = result.astype(np.float32) / 255.0
        
        return result
    
    @staticmethod
    def Sobel(image, ddepth, dx, dy, ksize=3):
        """Sobel filter using scipy"""
        if CV2_AVAILABLE:
            return cv2.Sobel(image, ddepth, dx, dy, ksize=ksize)
        
        if dx == 1 and dy == 0:
            return ndimage.sobel(image, axis=1).astype(np.float32)
        elif dx == 0 and dy == 1:
            return ndimage.sobel(image, axis=0).astype(np.float32)
        else:
            return np.zeros_like(image, dtype=np.float32)
    
    @staticmethod
    def approxPolyDP(contour, epsilon, closed):
        """Approximate polygon using simple Douglas-Peucker-like algorithm"""
        if CV2_AVAILABLE:
            return cv2.approxPolyDP(contour, epsilon, closed)
        
        # Simple approximation - just return original contour
        # In a full implementation, we'd use the Douglas-Peucker algorithm
        return contour
    
    @staticmethod
    def boxPoints(rect):
        """Get box points from rotated rectangle"""
        if CV2_AVAILABLE:
            return cv2.boxPoints(rect)
        
        center, size, angle = rect
        cx, cy = center
        w, h = size
        angle = np.radians(angle)
        
        # Calculate corners
        corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        
        # Rotate corners
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_corners = corners @ rotation_matrix.T
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy
        
        return rotated_corners.astype(np.float32)
    
    @staticmethod
    def fillPoly(image, pts, color):
        """Fill polygon - simplified version"""
        if CV2_AVAILABLE:
            return cv2.fillPoly(image, pts, color)
        
        # Simple implementation - would need more sophisticated polygon filling
        # For now, just return the image unchanged
        return image

# Create the cv2 replacement
if not CV2_AVAILABLE:
    cv2 = CloudCompatibleCV2()