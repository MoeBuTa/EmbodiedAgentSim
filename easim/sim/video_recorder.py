import cv2
import numpy as np
import time
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

from easim.utils.constants import (
    DEFAULT_FPS, DEFAULT_VIDEO_RESOLUTION, DEFAULT_VIDEO_CODEC,
    ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
    SENSOR_RGB
)

# Import pygame only when needed to avoid dependency issues
try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available, interactive mode disabled")


class VideoRecorder:
    """Records simulation videos"""

    def __init__(self, output_path: str, fps: int = DEFAULT_FPS,
                 resolution: Tuple[int, int] = DEFAULT_VIDEO_RESOLUTION):
        self.output_path = Path(output_path)
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.frames = []

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def start_recording(self):
        """Initialize video writer"""
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.resolution
        )
        self.frames = []

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the video with proper RGB->BGR conversion"""
        if frame is None:
            return

        # Fix 1: Ensure proper data type (Habitat often outputs float32)
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Fix 2: Resize if needed
        if frame.shape[:2] != self.resolution[::-1]:  # OpenCV uses (height, width)
            frame = cv2.resize(frame, self.resolution)

        # Fix 3: Convert RGB to BGR (THIS IS THE KEY FIX)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Write frame
        if self.writer is not None:
            self.writer.write(frame)

        self.frames.append(frame.copy())

    def stop_recording(self):
        """Finalize and save video"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        print(f"Video saved to: {self.output_path}")

    def save_frames_as_images(self, output_dir: str):
        """Save individual frames as images"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(self.frames):
            frame_path = output_path / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)


class BaseNavigationStrategy(ABC):
    """Abstract base class for navigation strategies"""

    @abstractmethod
    def get_next_action(self, observations: Dict[str, np.ndarray],
                        step_count: int) -> str:
        """Get the next action based on observations"""
        pass

    @abstractmethod
    def is_done(self, observations: Dict[str, np.ndarray],
                step_count: int) -> bool:
        """Check if navigation is complete"""
        pass


class RandomNavigationStrategy(BaseNavigationStrategy):
    """Random navigation strategy"""

    def __init__(self, max_steps: int = 100, forward_prob: float = 0.7):
        self.max_steps = max_steps
        self.forward_prob = forward_prob
        self.actions = [ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT]

    def get_next_action(self, observations: Dict[str, np.ndarray],
                        step_count: int) -> str:
        """Get random action with forward bias"""
        if np.random.random() < self.forward_prob:
            return ACTION_MOVE_FORWARD
        else:
            return np.random.choice([ACTION_TURN_LEFT, ACTION_TURN_RIGHT])

    def is_done(self, observations: Dict[str, np.ndarray],
                step_count: int) -> bool:
        """Stop after max steps"""
        return step_count >= self.max_steps


class FixedPathStrategy(BaseNavigationStrategy):
    """Navigate using a predefined sequence of actions"""

    def __init__(self, action_sequence: List[str]):
        self.action_sequence = action_sequence
        self.current_step = 0

    def get_next_action(self, observations: Dict[str, np.ndarray],
                        step_count: int) -> str:
        """Get next action from sequence"""
        if self.current_step < len(self.action_sequence):
            action = self.action_sequence[self.current_step]
            self.current_step += 1
            return action
        return ACTION_MOVE_FORWARD  # Default action

    def is_done(self, observations: Dict[str, np.ndarray],
                step_count: int) -> bool:
        """Stop when sequence is complete"""
        return self.current_step >= len(self.action_sequence)


class SimulationRecorder:
    """Records simulation with video and automation support"""

    def __init__(self, simulator, output_dir: str):
        self.simulator = simulator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.video_recorder = None
        self.current_strategy = None

    def record_navigation(self, strategy: BaseNavigationStrategy,
                          video_filename: str = "navigation.mp4",
                          save_frames: bool = False) -> Dict[str, Any]:
        """Record navigation using given strategy"""

        # Setup video recording
        video_path = self.output_dir / video_filename
        self.video_recorder = VideoRecorder(str(video_path))
        self.video_recorder.start_recording()

        # Reset simulator
        self.simulator.reset()

        # Navigation loop
        step_count = 0
        start_time = time.time()

        while True:
            # Get observations
            observations = self.simulator.get_observations()

            # Add frame to video
            if SENSOR_RGB in observations:
                rgb_frame = observations[SENSOR_RGB]
                self.video_recorder.add_frame(rgb_frame)

            # Check if done
            if strategy.is_done(observations, step_count):
                break

            # Get next action
            action = strategy.get_next_action(observations, step_count)

            # Take action
            observations = self.simulator.step(action)
            step_count += 1

        # Finalize recording
        end_time = time.time()
        self.video_recorder.stop_recording()

        # Save frames if requested
        if save_frames:
            frames_dir = self.output_dir / f"{video_filename}_frames"
            self.video_recorder.save_frames_as_images(str(frames_dir))

        # Return statistics
        return {
            "total_steps": step_count,
            "duration": end_time - start_time,
            "video_path": str(video_path),
            "scene_info": self.simulator.get_scene_info()
        }


class PygameVisualizer:
    """Real-time visualization using pygame"""

    def __init__(self, window_size: Tuple[int, int] = (800, 600)):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for interactive visualization")

        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Habitat Simulation")
        self.clock = pygame.time.Clock()
        self.running = False

        # Font for text display
        self.font = pygame.font.Font(None, 36)

    def run_interactive_simulation(self, simulator):
        """Run interactive simulation with keyboard controls"""
        self.running = True
        simulator.reset()

        while self.running:
            # Handle events
            action = self._handle_events()

            if action:
                simulator.step(action)

            # Get observations and display
            observations = simulator.get_observations()
            self._render_observations(observations)

            # Control frame rate
            self.clock.tick(30)

        pygame.quit()

    def _handle_events(self) -> Optional[str]:
        """Handle pygame events and return action"""
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    action = ACTION_MOVE_FORWARD
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    action = ACTION_TURN_LEFT
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    action = ACTION_TURN_RIGHT
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

        return action

    def _render_observations(self, observations: Dict[str, np.ndarray]):
        """Render observations on pygame screen"""
        self.screen.fill((0, 0, 0))  # Clear screen

        # Display RGB image if available
        if SENSOR_RGB in observations:
            rgb_image = observations[SENSOR_RGB]

            # Handle different image formats
            if len(rgb_image.shape) == 3:
                # Ensure image is in correct format for pygame
                if rgb_image.dtype != np.uint8:
                    # Convert to uint8 if needed
                    if rgb_image.max() <= 1.0:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    else:
                        rgb_image = rgb_image.astype(np.uint8)

                # Handle RGBA vs RGB
                if rgb_image.shape[2] == 4:
                    # Convert RGBA to RGB by dropping alpha channel
                    rgb_image = rgb_image[:, :, :3]
                    print("Converted RGBA to RGB")
                elif rgb_image.shape[2] != 3:
                    print(f"Unexpected number of channels: {rgb_image.shape[2]}")
                    return

                try:
                    # Create pygame surface - need to transpose for pygame (width, height)
                    rgb_surface = pygame.surfarray.make_surface(
                        rgb_image.swapaxes(0, 1)
                    )

                    # Scale to fit window
                    scaled_surface = pygame.transform.scale(rgb_surface, self.window_size)
                    self.screen.blit(scaled_surface, (0, 0))

                except Exception as e:
                    print(f"Surface creation failed: {e}")
                    # Fallback: create gray screen
                    self.screen.fill((128, 128, 128))
            else:
                print(f"Unexpected image dimensions: {rgb_image.shape}")
                self.screen.fill((128, 128, 128))
        else:
            print(f"No color sensor found. Available: {list(observations.keys())}")
            self.screen.fill((128, 128, 128))

        # Add controls text with better positioning
        controls_text = [
            "Controls:",
            "W/↑ - Move Forward",
            "A/← - Turn Left",
            "D/→ - Turn Right",
            "ESC - Quit"
        ]

        # Create semi-transparent background for text
        text_bg = pygame.Surface((250, 160))
        text_bg.set_alpha(180)
        text_bg.fill((0, 0, 0))
        self.screen.blit(text_bg, (10, 10))

        y_offset = 20
        for line in controls_text:
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (20, y_offset))
            y_offset += 25

        pygame.display.flip()