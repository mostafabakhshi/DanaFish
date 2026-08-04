import cv2
import numpy as np
import os
import logging
from rotation_model import ZebraFishRotationModel
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageRotationCorrector:
    """
    Step 2: Image Orientation Correction System
    Uses landmarks from Step 1 to automatically flip zebrafish images 
    to standard orientation using horizontal and vertical flips only.
    Target: head (h) on left of body (b), tail (t) below body (b)
    """
    
    def __init__(self, rotation_model_path="runs/detect/rotation_model_v1/weights/best.pt", target_size=(840, 840)):
        """Initialize the rotation corrector with trained landmark model."""
        self.rotation_model = ZebraFishRotationModel(rotation_model_path)
        self.target_size = target_size  # (width, height) for resizing
        logger.info(f"Image Rotation Corrector initialized with target size: {target_size}")
    
    def resize_image_with_aspect_ratio(self, image, target_size=(840, 840)):
        """
        Resize image to target size while maintaining aspect ratio and padding with black.
        
        Args:
            image: Input image
            target_size: Target (width, height) tuple
            
        Returns:
            resized_image: Resized image with padding
            scale_factor: Scale factor used for resizing
            offset_x, offset_y: Padding offsets
        """
        
        target_width, target_height = target_size
        original_height, original_width = image.shape[:2]
        
        # Calculate scale factor to fit image in target size while maintaining aspect ratio
        scale_width = target_width / original_width
        scale_height = target_height / original_height
        scale_factor = min(scale_width, scale_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate padding offsets to center the image
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        
        # Place resized image on canvas
        canvas[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = resized
        
        logger.info(f"Resized image from {original_width}x{original_height} to {target_width}x{target_height} "
                   f"(scale: {scale_factor:.3f}, offset: {offset_x},{offset_y})")
        
        return canvas, scale_factor, offset_x, offset_y
    
    def transform_landmarks_for_resize(self, landmarks, scale_factor, offset_x, offset_y):
        """
        Transform landmark coordinates after resizing and padding.
        
        Args:
            landmarks: Original landmarks dictionary
            scale_factor: Scale factor used for resizing
            offset_x, offset_y: Padding offsets
            
        Returns:
            transformed_landmarks: Landmarks with updated coordinates for resized image
        """
        
        if not landmarks:
            return landmarks
        
        def transform_point(point):
            """Transform a single point for resize."""
            x, y = point
            new_x = int(x * scale_factor + offset_x)
            new_y = int(y * scale_factor + offset_y)
            return [new_x, new_y]
        
        def transform_bbox(bbox):
            """Transform a bounding box for resize."""
            x1, y1, x2, y2 = bbox
            
            # Transform corners
            corner1 = transform_point([x1, y1])
            corner2 = transform_point([x2, y2])
            
            # Ensure proper bbox format
            new_x1 = min(corner1[0], corner2[0])
            new_x2 = max(corner1[0], corner2[0])
            new_y1 = min(corner1[1], corner2[1])
            new_y2 = max(corner1[1], corner2[1])
            
            return [new_x1, new_y1, new_x2, new_y2]
        
        # Transform landmarks
        transformed_landmarks = {}
        
        for class_name, landmark in landmarks.items():
            if class_name in ['head', 'tail', 'body'] and landmark:
                transformed_landmarks[class_name] = {
                    'bbox': transform_bbox(landmark['bbox']),
                    'center': transform_point(landmark['center']),
                    'confidence': landmark['confidence']
                }
            elif class_name in ['head', 'tail', 'body']:
                transformed_landmarks[class_name] = None
        
        return transformed_landmarks
    
    def calculate_orientation_flips(self, landmarks):
        """
        Calculate the flips needed to achieve standard orientation using only horizontal and vertical flips.
        Target: Head (h) to the LEFT of body (b), Tail (t) BELOW body (b).
        
        Conditions:
        1. Head X-coordinate < Body X-coordinate (head left of body)
        2. Tail Y-coordinate > Body Y-coordinate (tail below body)
        
        Args:
            landmarks: Dictionary with head, tail, body landmarks from Step 1
            
        Returns:
            needs_horizontal_flip: Whether to flip horizontally
            needs_vertical_flip: Whether to flip vertically
        """
        
        if not landmarks['head'] or not landmarks['tail'] or not landmarks['body']:
            logger.warning("Missing required landmarks for orientation calculation")
            return False, False
        
        # Get landmark centers
        head_center = np.array(landmarks['head']['center'])
        tail_center = np.array(landmarks['tail']['center'])
        body_center = np.array(landmarks['body']['center'])
        
        logger.info(f"Original positions: Head={head_center}, Body={body_center}, Tail={tail_center}")
        
        # Check current conditions
        head_is_left_of_body = head_center[0] < body_center[0]  # X coordinate comparison
        tail_is_below_body = tail_center[1] > body_center[1]    # Y coordinate comparison
        
        logger.info(f"Current orientation: Head left of body={head_is_left_of_body}, Tail below body={tail_is_below_body}")
        
        # Determine flips needed to satisfy both conditions
        needs_horizontal_flip = False
        needs_vertical_flip = False
        
        # If head is not left of body, we need horizontal flip
        if not head_is_left_of_body:
            needs_horizontal_flip = True
            logger.info("Head is not left of body - horizontal flip needed")
        
        # If tail is not below body, we need vertical flip
        if not tail_is_below_body:
            needs_vertical_flip = True
            logger.info("Tail is not below body - vertical flip needed")
        
        # Test all 4 possible combinations to find the best solution
        test_cases = [
            (False, False, "no_flip"),
            (True, False, "horizontal_only"),
            (False, True, "vertical_only"),
            (True, True, "both_flips")
        ]
        
        best_h_flip = False
        best_v_flip = False
        best_score = 0  # Number of conditions satisfied
        best_strategy = "no_flip"
        
        for h_flip, v_flip, strategy in test_cases:
            # Simulate the flips
            test_head_center = head_center.copy()
            test_tail_center = tail_center.copy()
            test_body_center = body_center.copy()
            
            # Apply horizontal flip simulation (flip X coordinates around image center)
            if h_flip:
                # For simulation, assume image width (we'll use actual width in real transformation)
                # This is just for testing the logic
                test_head_center[0] = -test_head_center[0] + 2 * body_center[0]
                test_tail_center[0] = -test_tail_center[0] + 2 * body_center[0]
            
            # Apply vertical flip simulation (flip Y coordinates around image center)
            if v_flip:
                test_head_center[1] = -test_head_center[1] + 2 * body_center[1]
                test_tail_center[1] = -test_tail_center[1] + 2 * body_center[1]
            
            # Check conditions after simulated flips
            test_head_left = test_head_center[0] < test_body_center[0]
            test_tail_below = test_tail_center[1] > test_body_center[1]
            
            # Calculate score (number of conditions satisfied)
            score = int(test_head_left) + int(test_tail_below)
            
            logger.info(f"Test {strategy}: h_flip={h_flip}, v_flip={v_flip}, "
                       f"head_left={test_head_left}, tail_below={test_tail_below}, score={score}")
            
            # Choose the solution with the highest score
            if score > best_score:
                best_score = score
                best_h_flip = h_flip
                best_v_flip = v_flip
                best_strategy = strategy
        
        logger.info(f"Best flip strategy: {best_strategy} with score: {best_score}/2")
        logger.info(f"Final flip decision: h_flip={best_h_flip}, v_flip={best_v_flip}")
        
        if best_score == 2:
            logger.info("✅ Perfect orientation will be achieved!")
        elif best_score == 1:
            logger.info("⚠️ Partial orientation improvement (1/2 conditions satisfied)")
        else:
            logger.info("❌ No improvement possible with current landmarks")
        
        return best_h_flip, best_v_flip
    
    def rotate_image(self, image, angle, center=None):
        """
        Rotate image by specified angle around center point.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = clockwise)
            center: Rotation center (default: image center)
            
        Returns:
            rotated_image: Rotated image with same dimensions
        """
        
        if center is None:
            center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0))
        
        return rotated
    
    def flip_image(self, image, horizontal=False, vertical=False):
        """
        Flip image horizontally and/or vertically.
        
        Args:
            image: Input image
            horizontal: Whether to flip horizontally
            vertical: Whether to flip vertically
            
        Returns:
            flipped_image: Flipped image
        """
        
        result = image.copy()
        
        if horizontal:
            result = cv2.flip(result, 1)  # Horizontal flip
        
        if vertical:
            result = cv2.flip(result, 0)  # Vertical flip
        
        return result
    
    def transform_landmarks(self, landmarks, image_shape, h_flip=False, v_flip=False):
        """
        Transform landmark coordinates after flipping (no rotation).
        
        Args:
            landmarks: Original landmarks dictionary
            image_shape: (height, width) of the image
            h_flip: Whether horizontal flip was applied
            v_flip: Whether vertical flip was applied
            
        Returns:
            transformed_landmarks: Landmarks with updated coordinates
        """
        
        if not landmarks:
            return landmarks
        
        height, width = image_shape[:2]
        
        def transform_point(point):
            """Transform a single point through flips only."""
            x, y = point
            
            # Apply flips
            if h_flip:
                x = width - 1 - x
            if v_flip:
                y = height - 1 - y
            
            return [int(x), int(y)]
        
        def transform_bbox(bbox):
            """Transform a bounding box through flips."""
            x1, y1, x2, y2 = bbox
            
            # Transform corners
            corner1 = transform_point([x1, y1])
            corner2 = transform_point([x2, y2])
            
            # Find new bounding box (min/max of transformed corners)
            new_x1 = min(corner1[0], corner2[0])
            new_x2 = max(corner1[0], corner2[0])
            new_y1 = min(corner1[1], corner2[1])
            new_y2 = max(corner1[1], corner2[1])
            
            # Ensure bounds are within image
            new_x1 = max(0, min(new_x1, width - 1))
            new_x2 = max(0, min(new_x2, width - 1))
            new_y1 = max(0, min(new_y1, height - 1))
            new_y2 = max(0, min(new_y2, height - 1))
            
            return [new_x1, new_y1, new_x2, new_y2]
        
        # Transform landmarks (excluding neurons per project specification)
        transformed_landmarks = {}
        
        for class_name, landmark in landmarks.items():
            if class_name in ['head', 'tail', 'body'] and landmark:  # Use full names
                transformed_landmarks[class_name] = {
                    'bbox': transform_bbox(landmark['bbox']),
                    'center': transform_point(landmark['center']),
                    'confidence': landmark['confidence']
                }
            elif class_name in ['head', 'tail', 'body']:  # Handle None values
                transformed_landmarks[class_name] = None
        
        return transformed_landmarks
    
    def get_spinal_region(self, corrected_image, corrected_landmarks, padding=20):
        """
        Extract the body (b) region as the spinal region for next-step analysis.
        This region will be used for spinal cord detection and neuron analysis.
        
        Args:
            corrected_image: The orientation-corrected image
            corrected_landmarks: Transformed landmarks after correction
            padding: Additional padding around the body bounding box
            
        Returns:
            spinal_region_info: Dictionary containing:
                - region_image: Cropped image of the spinal region
                - region_bbox: Coordinates of the region in the full image
                - body_center_in_region: Body center relative to cropped region
                - success: Whether extraction was successful
        """
        
        if not corrected_landmarks or not corrected_landmarks.get('body'):
            logger.warning("No body landmark found for spinal region extraction")
            return {
                'region_image': None,
                'region_bbox': None,
                'body_center_in_region': None,
                'success': False
            }
        
        body_landmark = corrected_landmarks['body']
        body_bbox = body_landmark['bbox']
        body_center = body_landmark['center']
        
        # Expand bounding box with padding
        x1, y1, x2, y2 = body_bbox
        height, width = corrected_image.shape[:2]
        
        # Add padding and ensure bounds
        padded_x1 = max(0, x1 - padding)
        padded_y1 = max(0, y1 - padding)
        padded_x2 = min(width, x2 + padding)
        padded_y2 = min(height, y2 + padding)
        
        # Extract region
        region_image = corrected_image[padded_y1:padded_y2, padded_x1:padded_x2]
        
        # Calculate body center relative to the cropped region
        body_center_in_region = [
            body_center[0] - padded_x1,
            body_center[1] - padded_y1
        ]
        
        logger.info(f"Extracted spinal region: {region_image.shape[:2]} from body bbox {body_bbox}")
        logger.info(f"Body center in region: {body_center_in_region}")
        
        return {
            'region_image': region_image,
            'region_bbox': [padded_x1, padded_y1, padded_x2, padded_y2],
            'body_center_in_region': body_center_in_region,
            'body_confidence': body_landmark['confidence'],
            'success': True
        }
    
    def visualize_landmarks(self, image, landmarks, title="Landmarks"):
        """
        Draw bounding boxes and labels for detected landmarks.
        
        Args:
            image: Input image
            landmarks: Dictionary with detected landmarks
            title: Title for visualization
            
        Returns:
            annotated_image: Image with bounding boxes and labels
        """
        
        annotated = image.copy()
        
        # Class colors (BGR format) - Standardized colors per project specification
        colors = {
            'head': (0, 255, 0),    # Green for head
            'tail': (0, 0, 255),    # Red for tail
            'body': (255, 0, 0),    # Blue for body
        }
        
        # Draw landmarks (excluding neurons as per project specification)
        for class_name, landmark in landmarks.items():
            if landmark and class_name in ['head', 'tail', 'body']:  # Only process orientation landmarks
                bbox = landmark['bbox']
                confidence = landmark['confidence']
                center = landmark['center']
                
                # Draw bounding box
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            colors.get(class_name, (255, 255, 255)), 2)
                
                # Draw center point
                cv2.circle(annotated, tuple(center), 3, colors.get(class_name, (255, 255, 255)), -1)
                
                # Draw label with short name for display
                short_name = {'head': 'H', 'tail': 'T', 'body': 'B'}.get(class_name, class_name.upper())
                label = f"{short_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated, (bbox[0], bbox[1] - label_size[1] - 5), 
                            (bbox[0] + label_size[0], bbox[1]), 
                            colors.get(class_name, (255, 255, 255)), -1)
                cv2.putText(annotated, label, (bbox[0], bbox[1] - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
    
    def create_comparison_image(self, original_image, corrected_image, original_landmarks, corrected_landmarks, title="Before vs After"):
        """
        Create a side-by-side comparison of original and corrected images with annotations.
        
        Args:
            original_image: Original image
            corrected_image: Corrected image
            original_landmarks: Original landmarks
            corrected_landmarks: Transformed landmarks
            title: Title for the comparison
            
        Returns:
            comparison_image: Side-by-side comparison
        """
        
        # Create annotated versions
        original_annotated = self.visualize_landmarks(original_image, original_landmarks, "Original")
        corrected_annotated = self.visualize_landmarks(corrected_image, corrected_landmarks, "Corrected")
        
        # Ensure both images have the same height
        height = max(original_annotated.shape[0], corrected_annotated.shape[0])
        
        # Resize if needed
        if original_annotated.shape[0] != height:
            original_annotated = cv2.resize(original_annotated, (original_annotated.shape[1], height))
        if corrected_annotated.shape[0] != height:
            corrected_annotated = cv2.resize(corrected_annotated, (corrected_annotated.shape[1], height))
        
        # Create side-by-side comparison
        comparison = np.hstack([original_annotated, corrected_annotated])
        
        # Add title
        title_height = 50
        title_image = np.ones((title_height, comparison.shape[1], 3), dtype=np.uint8) * 255
        
        # Add text
        font_scale = 1.0
        font_thickness = 2
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (comparison.shape[1] - text_size[0]) // 2
        text_y = (title_height + text_size[1]) // 2
        
        cv2.putText(title_image, title, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        # Add separator line
        separator_x = original_annotated.shape[1]
        cv2.line(comparison, (separator_x, 0), (separator_x, height), (255, 255, 255), 3)
        
        # Combine title and comparison
        final_comparison = np.vstack([title_image, comparison])
        
        return final_comparison
    
    def correct_image_orientation(self, image_path, output_path=None, save_annotated=True):
        """
        Correct the orientation of a single image with automatic resizing to 840x840.
        
        Args:
            image_path: Path to input image
            output_path: Path to save corrected image (optional)
            save_annotated: Whether to save annotated version with bounding boxes
            
        Returns:
            corrected_image: Orientation-corrected and resized image
            correction_info: Dictionary with correction details
        """
        
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        logger.info(f"Correcting orientation for: {os.path.basename(image_path)}")
        
        # Step 1: Resize image to 840x840 with aspect ratio preservation
        resized_image, scale_factor, offset_x, offset_y = self.resize_image_with_aspect_ratio(
            original_image, self.target_size
        )
        
        # Step 2: Get landmarks from the ORIGINAL image (before resize)
        landmarks = self.rotation_model.get_orientation_landmarks(image_path)
        
        # Step 3: Transform landmarks to match the resized image
        if landmarks:
            resized_landmarks = self.transform_landmarks_for_resize(
                landmarks, scale_factor, offset_x, offset_y
            )
            logger.info("Transformed landmarks for resized image")
        else:
            resized_landmarks = landmarks
        
        # Step 4: Calculate required flips based on resized landmarks
        h_flip, v_flip = self.calculate_orientation_flips(resized_landmarks)
        
        # Step 5: Apply corrections to resized image
        corrected_image = resized_image.copy()
        
        # Apply flips
        if h_flip or v_flip:
            corrected_image = self.flip_image(corrected_image, h_flip, v_flip)
            logger.info(f"Applied flips: horizontal={h_flip}, vertical={v_flip}")
        else:
            logger.info("No flips needed - image already properly oriented")
        
        # Step 6: Transform resized landmarks to match final corrected image
        if resized_landmarks:
            final_landmarks = self.transform_landmarks(
                resized_landmarks, 
                resized_image.shape, 
                h_flip, 
                v_flip
            )
            logger.info("Transformed landmarks to match final corrected image")
        else:
            final_landmarks = resized_landmarks
        
        # Step 7: Save corrected images if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save plain corrected image (resized and flipped)
            cv2.imwrite(output_path, corrected_image)
            logger.info(f"Saved corrected image (840x840): {output_path}")
            
            # Save annotated version if requested
            if save_annotated and final_landmarks:
                # Create annotated image with landmarks
                annotated_image = self.visualize_landmarks(
                    corrected_image, final_landmarks, 
                    f"Corrected: {os.path.basename(image_path)} (840x840)"
                )
                
                # Generate annotated filename
                base_name, ext = os.path.splitext(output_path)
                annotated_path = f"{base_name}_annotated{ext}"
                cv2.imwrite(annotated_path, annotated_image)
                logger.info(f"Saved annotated image: {annotated_path}")
                
                # Create and save comparison image (original vs final)
                # Note: For comparison, we'll show original vs final corrected
                comparison_image = self.create_comparison_image(
                    resized_image, corrected_image, resized_landmarks, final_landmarks,
                    f"Resize + Flip Correction: {os.path.basename(image_path)}"
                )
                comparison_path = f"{base_name}_comparison{ext}"
                cv2.imwrite(comparison_path, comparison_image)
                logger.info(f"Saved comparison image: {comparison_path}")
        
        # Step 8: Prepare correction info
        correction_info = {
            'original_path': image_path,
            'output_path': output_path,
            'original_size': original_image.shape[:2],  # (height, width)
            'resized_to': self.target_size,  # (width, height)
            'scale_factor': scale_factor,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'horizontal_flip': h_flip,
            'vertical_flip': v_flip,
            'landmarks_detected': {
                'head': landmarks['head'] is not None if landmarks else False,
                'tail': landmarks['tail'] is not None if landmarks else False,
                'body': landmarks['body'] is not None if landmarks else False
            },
            'correction_applied': h_flip or v_flip,
            'resize_applied': True,
            'correction_method': 'resize_then_flip',  # Updated method indicator
            'original_landmarks': landmarks,  # Store original landmarks
            'resized_landmarks': resized_landmarks,  # Store resized landmarks
            'corrected_landmarks': final_landmarks  # Store final corrected landmarks
        }
        
        return corrected_image, correction_info
    
    def correct_batch_images(self, input_dir, output_dir, save_annotated=True):
        """
        Correct orientation for all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save corrected images
            save_annotated: Whether to save annotated versions with bounding boxes
            
        Returns:
            results: List of correction results for each image
        """
        
        logger.info(f"Starting batch orientation correction...")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Save annotated: {save_annotated}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = []
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        successful_corrections = 0
        
        for i, image_path in enumerate(image_files, 1):
            try:
                # Generate output path maintaining directory structure
                rel_path = os.path.relpath(image_path, input_dir)
                output_path = os.path.join(output_dir, f"corrected_{rel_path}")
                output_path = output_path.replace('\\', '/')  # Normalize path separators
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                logger.info(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
                
                # Correct orientation
                corrected_image, correction_info = self.correct_image_orientation(
                    image_path, output_path, save_annotated
                )
                
                results.append(correction_info)
                
                if correction_info['correction_applied']:
                    successful_corrections += 1
                    logger.info(f"✓ Corrected: "
                               f"h_flip={correction_info['horizontal_flip']}, "
                               f"v_flip={correction_info['vertical_flip']}")
                    if save_annotated:
                        logger.info(f"  ✓ Saved plain, annotated, and comparison versions")
                else:
                    logger.info("✓ No correction needed (already properly oriented)")
                    if save_annotated:
                        logger.info(f"  ✓ Saved annotated version showing current landmarks")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {image_path}: {str(e)}")
                results.append({
                    'original_path': image_path,
                    'error': str(e),
                    'correction_applied': False
                })
        
        # Summary
        logger.info(f"\nBatch correction completed:")
        logger.info(f"  Total images: {len(image_files)}")
        logger.info(f"  Successfully processed: {len([r for r in results if 'error' not in r])}")
        logger.info(f"  Corrections applied: {successful_corrections}")
        logger.info(f"  Already properly oriented: {len([r for r in results if 'error' not in r and not r['correction_applied']])}")
        logger.info(f"  Errors: {len([r for r in results if 'error' in r])}")
        if save_annotated:
            logger.info(f"  Annotated and comparison images saved with bounding boxes for all classes")
        
        return results

if __name__ == "__main__":
    # Test the rotation corrector
    corrector = ImageRotationCorrector()
    
    # Test on a single image first
    test_image_dir = "datasets/fish13.v4-bnhtsize.yolov11/test/images"
    test_images = [f for f in os.listdir(test_image_dir) if f.lower().endswith('.jpg')]
    
    if test_images:
        test_image_path = os.path.join(test_image_dir, test_images[0])
        output_path = f"test_results/step2_rotation/corrected_{test_images[0]}"
        
        logger.info("Testing Step 2: Image Rotation Correction")
        corrected_image, info = corrector.correct_image_orientation(test_image_path, output_path)
        logger.info(f"Test completed. Correction info: {info}")
    else:
        logger.error("No test images found")