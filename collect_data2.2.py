import carla
import time
import numpy as np
import random
import cv2
import os
import traceback
from threading import Event

# Create directory for saving data if it doesn't exist
os.makedirs('carla_data_all', exist_ok=True)

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(50.0)
world = client.get_world()

# Set simulation settings for 5 FPS
settings = world.get_settings()
settings.fixed_delta_seconds = 1.0 / 3
settings.synchronous_mode = True
world.apply_settings(settings)

# Get the blueprint library and select vehicle and camera blueprints
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1920')
camera_bp.set_attribute('image_size_y', '1080')
camera_bp.set_attribute('fov', '160')
camera_transform = carla.Transform(carla.Location(x=0, y=0, z=10), carla.Rotation(pitch=-90))

# Get all available spawn points
all_spawn_points = world.get_map().get_spawn_points()

# Initialize traffic manager
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

def preprocess_frame(frame):
    """
    Resize the input frame to (300,300), convert to grayscale, normalize, and compress using JPEG quality=20.
    
    Parameters:
      frame (np.array): Input image array with shape (H, W, 4) or (H, W, 3)
    
    Returns:
      compressed_img (np.array): 1D numpy array (uint8) containing the JPEG-compressed data.
    """
    try:
        # Remove alpha channel if present
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Resize the image to (300,300)
        resized_frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_AREA)
        
        # Convert resized image to grayscale
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
        
        # Normalize the grayscale image to the range [0,255]
        min_val = gray.min()
        max_val = gray.max()
        if max_val > min_val:
            norm_gray = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            norm_gray = gray.astype(np.uint8)
        
        # Compress the normalized grayscale image using JPEG encoding with quality=20
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        ret, compressed_img = cv2.imencode('.jpg', norm_gray, encode_param)
        if not ret:
            raise ValueError("Image compression failed")
        
        return compressed_img
    except Exception as e:
        print(f"Error in preprocessing frame: {e}")
        # Return a simple compressed placeholder in case of error
        dummy = np.zeros((100, 100), dtype=np.uint8)
        ret, comp = cv2.imencode('.jpg', dummy)
        return comp

def clean_up_actors(vehicle=None, camera=None):
    """Clean up specific actors or all if none specified"""
    try:
        if camera and camera.is_alive:
            camera.stop()
            camera.destroy()
            print("Camera destroyed")
        
        if vehicle and vehicle.is_alive:
            vehicle.set_autopilot(False)
            vehicle.destroy()
            print("Vehicle destroyed")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")

def emergency_cleanup():
    """Clean up all actors in case of an emergency"""
    print("Performing emergency cleanup...")
    try:
        for actor in world.get_actors():
            if actor.is_alive and (actor.type_id.startswith('vehicle.') or actor.type_id.startswith('sensor.')):
                actor.destroy()
        print("Emergency cleanup completed")
    except Exception as e:
        print(f"Error during emergency cleanup: {e}")

# Storage for all rollouts
all_rollouts_data = []

# Set the number of steps per rollout (reduced from 1000)
STEPS_PER_ROLLOUT = 500

# Main try-except block for the entire simulation
try:
    # Run multiple rollouts (default is 10)
    MAX_ROLLOUTS = 250
    for rollout in range(MAX_ROLLOUTS):
        print(f"Starting rollout {rollout+1}/{MAX_ROLLOUTS}")
        
        # Variables for the current rollout
        vehicle = None
        camera = None
        observations = []  # Each entry is the compressed image data
        actions = []       # Each entry is a dictionary of vehicle control commands
        
        # Use an Event for synchronization instead of a boolean flag
        image_ready_event = Event()
        
        try:
            # Select a random spawn point and spawn the vehicle
            spawn_point = random.choice(all_spawn_points)
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Vehicle spawned at {spawn_point}")
            
            # Create a shared data structure for camera callback
            class SharedData:
                def __init__(self):
                    self.vehicle = vehicle
                    self.observations = observations
                    self.actions = actions
                    self.image_ready_event = image_ready_event
            
            shared_data = SharedData()
            
            def process_image(image, data):
                try:
                    # Convert raw data to a NumPy array
                    img = np.array(image.raw_data).reshape((image.height, image.width, 4))
                    # Compress the frame
                    compressed = preprocess_frame(img)
                    data.observations.append(compressed)
                    
                    if data.vehicle and data.vehicle.is_alive:
                        control = data.vehicle.get_control()
                        control_dict = {
                            'throttle': float(control.throttle),
                            'steer': float(control.steer),
                            'brake': float(control.brake),
                            'hand_brake': bool(control.hand_brake),
                            'reverse': bool(control.reverse)
                        }
                        data.actions.append(control_dict)
                    else:
                        print("Vehicle no longer valid in camera callback")
                    
                    # Signal that the image processing is complete
                    data.image_ready_event.set()
                except Exception as e:
                    print(f"Error in camera callback: {e}")
                    # Still signal completion to avoid deadlocks
                    data.image_ready_event.set()
            
            # Attach and start the camera with a lambda that includes our shared data
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            camera.listen(lambda image: process_image(image, shared_data))
            print("Camera attached and started")
            
            # Enable autopilot
            vehicle.set_autopilot(True, traffic_manager.get_port())
            print("Autopilot enabled")
            
            # Run the simulation for the specified number of steps
            for step in range(STEPS_PER_ROLLOUT):
                # Clear the event flag
                image_ready_event.clear()
                
                # Tick the world
                world.tick()
                
                # Wait for the camera callback to process the image
                # Set a timeout to prevent hanging if the callback never completes
                if not image_ready_event.wait(timeout=1.0):
                    print(f"Warning: Camera callback timed out at step {step}")
                
                if step % 20 == 0:
                    print(f"Rollout {rollout+1}, Step {step}/{STEPS_PER_ROLLOUT}")
            
            # Rollout complete, disable autopilot
            if vehicle and vehicle.is_alive:
                vehicle.set_autopilot(False)
            
            print(f"Rollout {rollout+1} completed successfully")
            
            # Save data for this rollout immediately
            rollout_filename = f'carla_data_all/rollout_{rollout+1+49}.npz'
            np.savez_compressed(
                rollout_filename,
                observations=observations,
                actions=actions
            )
            print(f"Saved rollout data to {rollout_filename}")
            
            # Store summary information
            all_rollouts_data.append({
                'rollout': rollout + 1,
                'num_frames': len(observations),
                'filename': rollout_filename
            })
            
        except Exception as e:
            print(f"Error during rollout {rollout+1}: {e}")
            print(traceback.format_exc())
        finally:
            # Clean up actors for this rollout
            clean_up_actors(vehicle, camera)
            # Force garbage collection to free memory
            import gc
            gc.collect()
            # Wait a moment to ensure cleanup is complete
            time.sleep(1)
    
    # Save the metadata for all rollouts
    meta_filename = 'carla_data_all/rollouts_metadata.npz'
    np.savez(meta_filename, rollouts=all_rollouts_data)
    print(f"Saved metadata for {len(all_rollouts_data)} rollouts to {meta_filename}")
    
except Exception as e:
    print(f"Fatal error in main simulation loop: {e}")
    print(traceback.format_exc())
    emergency_cleanup()
finally:
    # Ensure we reset the simulation settings
    try:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Reset simulation to asynchronous mode")
    except Exception as e:
        print(f"Error resetting simulation settings: {e}")
    
    print("Data collection process finished")