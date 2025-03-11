import carla
import numpy as np
import cv2
import random
import time
import torch
from PIL import Image
import io


"""
    Spawn NPC vehicles and pedestrians with enhanced functionality.
    
    Args:
        num_vehicles (int): Number of vehicles to spawn.
        num_pedestrians (int): Number of pedestrians to spawn.
        vehicle_filter (str): Filter for vehicle blueprints.
        vehicle_generation (str): Generation of vehicles ('1', '2', or 'All').
        pedestrian_filter (str): Filter for pedestrian blueprints.
        pedestrian_generation (str): Generation of pedestrians ('1', '2', or 'All').
        safe_vehicles (bool): If True, only spawn safer vehicles (cars).
        enable_vehicle_lights (bool): If True, enable automatic vehicle lights.
    """


def spawn_npcs(self, num_vehicles=30, num_pedestrians=15, vehicle_filter='vehicle.*', vehicle_generation='All', pedestrian_filter='walker.pedestrian.*', pedestrian_generation='2', safe_vehicles=False, enable_vehicle_lights=False):
   
    # Initialize lists to track spawned actors
    self.npc_vehicles = []
    self.npc_pedestrians = []
    self.npc_controllers = []
    all_walker_ids = []
    
    # Get vehicle blueprints based on filter and generation
    vehicle_blueprints = self._get_actor_blueprints(vehicle_filter, vehicle_generation)
    
    # Filter for safer vehicles if requested
    if safe_vehicles:
        vehicle_blueprints = [bp for bp in vehicle_blueprints if bp.get_attribute('base_type') == 'car']
    
    # Sort blueprints for consistency
    vehicle_blueprints = sorted(vehicle_blueprints, key=lambda bp: bp.id)
    
    # --- Spawn vehicles ---
    spawn_points = self.world.get_map().get_spawn_points()
    num_spawn_points = len(spawn_points)
    
    if num_vehicles < num_spawn_points:
        random.shuffle(spawn_points)
    elif num_vehicles > num_spawn_points:
        print(f"Warning: Requested {num_vehicles} vehicles, but only found {num_spawn_points} spawn points")
        num_vehicles = num_spawn_points
    
    # Create batch commands for vehicle spawning
    batch = []
    for i, transform in enumerate(spawn_points[:num_vehicles]):
        blueprint = random.choice(vehicle_blueprints)
        
        # Set random color if available
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        
        # Set random driver if available
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        
        # Set role name
        blueprint.set_attribute('role_name', 'autopilot')
        
        # Create spawn command
        batch.append(carla.command.SpawnActor(blueprint, transform)
                    .then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.traffic_manager.get_port())))
    
    # Execute vehicle spawn batch
    for response in self.client.apply_batch_sync(batch, True):
        if response.error:
            print(f"Error spawning vehicle: {response.error}")
        else:
            self.npc_vehicles.append(response.actor_id)
    
    # Enable vehicle lights if requested
    if enable_vehicle_lights and self.npc_vehicles:
        all_vehicle_actors = self.world.get_actors(self.npc_vehicles)
        for actor in all_vehicle_actors:
            self.traffic_manager.update_vehicle_lights(actor, True)
    
    # --- Spawn pedestrians ---
    pedestrian_blueprints = self._get_actor_blueprints(pedestrian_filter, pedestrian_generation)
    
    # 1. Get random locations for pedestrians
    spawn_points = []
    for _ in range(num_pedestrians):
        spawn_point = carla.Transform()
        loc = self.world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    
    # 2. Spawn pedestrian actors
    batch = []
    walker_speeds = []
    
    for spawn_point in spawn_points:
        walker_bp = random.choice(pedestrian_blueprints)
        
        # Make walker not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        
        # Set walking speed
        if walker_bp.has_attribute('speed'):
            # Default to walking speed
            walker_speeds.append(walker_bp.get_attribute('speed').recommended_values[1])
        else:
            walker_speeds.append(0.0)
            
        batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
    
    # Execute pedestrian spawn batch
    results = self.client.apply_batch_sync(batch, True)
    
    # Track successfully spawned pedestrians and their speeds
    walkers_list = []
    valid_speeds = []
    
    for i, result in enumerate(results):
        if result.error:
            print(f"Error spawning pedestrian: {result.error}")
        else:
            walkers_list.append({"id": result.actor_id})
            valid_speeds.append(walker_speeds[i])
            self.npc_pedestrians.append(result.actor_id)
    
    # 3. Spawn walker controllers
    batch = []
    walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
    
    for walker in walkers_list:
        batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker["id"]))
    
    # Execute controller spawn batch
    results = self.client.apply_batch_sync(batch, True)
    
    for i, result in enumerate(results):
        if result.error:
            print(f"Error spawning controller: {result.error}")
        else:
            walkers_list[i]["con"] = result.actor_id
            self.npc_controllers.append(result.actor_id)
    
    # 4. Create a list of all walker and controller IDs
    for walker in walkers_list:
        all_walker_ids.append(walker["con"])  # Controller
        all_walker_ids.append(walker["id"])   # Walker
    
    # Wait for a tick to ensure client receives the last transforms
    self.world.tick()
    
    # 5. Initialize walker controllers
    all_actors = self.world.get_actors(all_walker_ids)
    
    for i in range(0, len(all_walker_ids), 2):
        if i < len(all_actors):
            # Start walker
            all_actors[i].start()
            # Set random destination
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # Set max speed
            all_actors[i].set_max_speed(float(valid_speeds[int(i/2)]))
    
    print(f'Successfully spawned {len(self.npc_vehicles)} vehicles and {len(self.npc_pedestrians)} walkers')
    
    return len(self.npc_vehicles), len(self.npc_pedestrians)



"""
    Get actor blueprints based on filter and generation.
    
    Args:
        filter_string (str): Blueprint filter string.
        generation (str): Actor generation ('1', '2', or 'All').
        
    Returns:
        list: List of blueprints that match the filter and generation.
"""

def _get_actor_blueprints(self, filter_string, generation):
   
    bps = self.blueprint_library.filter(filter_string)
    
    if generation.lower() == "all":
        return bps
    
    # If the filter returns only one bp, return it
    if len(bps) == 1:
        return bps
    
    try:
        int_generation = int(generation)
        # Check if generation is valid
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("Warning! Actor Generation is not valid. Using all generations.")
            return bps
    except:
        print("Warning! Actor Generation is not valid. Using all generations.")
        return bps

"""
    Destroy all spawned NPC vehicles and pedestrians.
 """

def destroy_npcs(self):
   
    # Destroy vehicles
    if hasattr(self, 'npc_vehicles') and self.npc_vehicles:
        print(f'Destroying {len(self.npc_vehicles)} vehicles')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_vehicles])
        self.npc_vehicles = []
    
    # Stop and destroy walker controllers and pedestrians
    if hasattr(self, 'npc_controllers') and self.npc_controllers and hasattr(self, 'npc_pedestrians') and self.npc_pedestrians:
        # Get all walker actors
        all_walker_ids = self.npc_controllers + self.npc_pedestrians
        all_actors = self.world.get_actors(all_walker_ids)
        
        # Stop controllers first
        for i in range(0, len(self.npc_controllers)):
            controller = self.world.get_actor(self.npc_controllers[i])
            if controller:
                controller.stop()
        
        print(f'Destroying {len(self.npc_pedestrians)} walkers')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in all_walker_ids])
        self.npc_controllers = []
        self.npc_pedestrians = []
    
    time.sleep(0.5)
    print('All NPCs destroyed')


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

class CarlaEnv:
    def __init__(self, town='Town12', render_mode=None):
        # Connect to Carla server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        
        # Load desired map
        self.world = self.client.load_world(town)
        
        # Set synchronous mode
        settings = self.world.get_settings()
        self.old_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        
        # Set up traffic manager
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        
        # Set up blueprint library
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Vehicle setup
        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None
        self.lane_invasion_sensor = None
        
        # Store camera image
        self.image = None
        
        # State tracking
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.last_location = None
        self.start_location = None
        self.total_distance = 0
        self.previous_distance = 0
        self.episode_start = 0
        self.render_mode = render_mode
        
        # Reward tracking
        self.step_reward = 0
        self.last_reward = 0
        
        # Action space: [steer, throttle, brake]
        self.action_space = 3

    
    def reset(self):
        # Clean up old actors
        if self.vehicle:
            if self.camera_sensor:
                self.camera_sensor.destroy()
            if self.collision_sensor:
                self.collision_sensor.destroy()
            if self.lane_invasion_sensor:
                self.lane_invasion_sensor.destroy()
            self.vehicle.destroy()
        
        # Clear previous episode data
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.last_reward = 0
        
        # Spawn vehicle
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        self.start_location = spawn_point.location
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Set up camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '160')
        camera_transform = carla.Transform(carla.Location(x=0, y=0, z=5), carla.Rotation(pitch=-90))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(lambda image: self._process_image(image))
        
        # Set up collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        # Set up lane invasion sensor
        lane_invasion_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

    #    self.spawn_npcs(30, 15)

        # Wait for sensors to initialize
        for _ in range(10):
            self.world.tick()
            time.sleep(0.01)
        
        self.episode_start = time.time()
        self.last_location = self.vehicle.get_location()
        self.total_distance = 0
        self.previous_distance = 0
        
        # Return initial observation
        return self._get_obs(), {}
    
    def _process_image(self, image):

        try:
            # Convert raw data to a NumPy array with shape (height, width, 4)
            img_array = np.array(image.raw_data).reshape((image.height, image.width, 4))
            
            # Preprocess the frame: resize, convert to grayscale, normalize, and compress
            compressed = preprocess_frame(img_array)
            
            # Decode the JPEG-compressed image back into a grayscale image
            # cv2.IMREAD_GRAYSCALE ensures that the result is a single-channel image
            decoded = cv2.imdecode(compressed, cv2.IMREAD_GRAYSCALE)
            self.image = decoded
        except Exception as e:
            print(f"Error processing image: {e}")
            # In case of error, assign a blank grayscale image (using the expected size, e.g. 300x300)
            self.image = np.zeros((300, 300), dtype=np.uint8)

    
    def _on_collision(self, event):
        self.collision_hist.append(event)
    
    def _on_lane_invasion(self, event):
        self.lane_invasion_hist.append(event)
    
    def _get_obs(self):
        if self.image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self.image
    
    def step(self, action):
        # Apply control to vehicle
        steer, throttle, brake = action
        control = carla.VehicleControl(
            throttle=float(max(0, throttle)),
            steer=float(steer),
            brake=float(max(0, brake)),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        self.vehicle.apply_control(control)
        
        # Advance simulation
        self.world.tick()
        
        # Calculate reward
        reward = self._compute_reward()
        self.last_reward = reward
        
        # Check termination conditions
        done = False
        # Terminate on collision
        if len(self.collision_hist) > 0:
            done = True
            reward -= 100  # Large penalty for collision
        
        # Terminate if stuck or moving too slow
        current_location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Compute distance traveled in this step
        distance = current_location.distance(self.last_location)
        self.total_distance += distance
        self.last_location = current_location
        
        # Terminate if vehicle is not moving for too long
        if speed < 0.1 and time.time() - self.episode_start > 5:
            done = True
            reward -= 50  # Penalty for getting stuck
        
        # Terminate after timeout (2 minutes)
        if time.time() - self.episode_start > 120:
            done = True
        
        # Get observation
        obs = self._get_obs()
        
        # Optionally render
        if self.render_mode == 'human':
            cv2.imshow("Carla", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        
        info = {
            'speed': speed,
            'distance': self.total_distance,
            'collisions': len(self.collision_hist),
            'lane_invasions': len(self.lane_invasion_hist)
        }
        
        return obs, reward, done, False, info
    
    def _compute_reward(self):
        reward = 0
        
        # Get vehicle state
        transform = self.vehicle.get_transform()
        location = transform.location
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Reward for moving forward
        reward += speed * 0.1
        
        # Penalty for lane invasions
        current_lane_invasions = len(self.lane_invasion_hist)
        if current_lane_invasions > 0:
            reward -= 2 * current_lane_invasions
            self.lane_invasion_hist = []  # Reset after accounting for penalty
        
        # Calculate distance traveled since last step
        distance_this_step = location.distance(self.last_location)
        self.previous_distance = distance_this_step
        
        # Waypoint-based reward: check if vehicle is following the road
        waypoint = self.world.get_map().get_waypoint(location)
        if waypoint:
            # Reward for staying on the road
            if waypoint.lane_type == carla.LaneType.Driving:
                # Give higher reward if in the center of the lane
                lane_center = waypoint.transform.location
                distance_from_center = location.distance(lane_center)
                lane_width = waypoint.lane_width
                # Normalize distance_from_center to [0, 1] where 0 is center and 1 is edge
                normalized_distance = min(distance_from_center / (lane_width / 2), 1.0)
                # Reward is higher when close to the center (1 - normalized_distance)
                reward += 2.0 * (1.0 - normalized_distance)
            else:
                # Penalty for being off the driving lane
                reward -= 2.0
        
        return reward
        
    def close(self):
        # Restore original settings
        if self.old_settings:
            self.world.apply_settings(self.old_settings)
        
        # Destroy sensors and the vehicle
        if self.camera_sensor:
            self.camera_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        
        # Destroy NPC vehicles
        if hasattr(self, 'npc_vehicles'):
            for npc_id in self.npc_vehicles:
                actor = self.world.get_actor(npc_id)
                if actor:
                    actor.destroy()
        
        # Destroy NPC pedestrians
        if hasattr(self, 'npc_pedestrians'):
            for ped_id in self.npc_pedestrians:
                actor = self.world.get_actor(ped_id)
                if actor:
                    actor.destroy()
        
        # Destroy NPC controllers
        if hasattr(self, 'npc_controllers'):
            for controller_id in self.npc_controllers:
                actor = self.world.get_actor(controller_id)
                if actor:
                    actor.destroy()
        
        if self.render_mode == 'human':
            cv2.destroyAllWindows()