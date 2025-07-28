"""
Video recording functionality for core simulator
"""
import cv2
import numpy as np
import time
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

from easim.utils.constants import (
    DEFAULT_FPS, DEFAULT_VIDEO_RESOLUTION, DEFAULT_VIDEO_CODEC
)

# Import pygame only when needed to avoid dependency issues
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class VideoRecorder:
    """Records simulation videos from RGB observations"""

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

        # Ensure proper data type
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Resize if needed
        if frame.shape[:2] != self.resolution[::-1]:  # OpenCV uses (height, width)
            frame = cv2.resize(frame, self.resolution)

        # Convert RGB to BGR for OpenCV
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
        self.actions = ["move_forward", "turn_left", "turn_right"]

    def get_next_action(self, observations: Dict[str, np.ndarray],
                        step_count: int) -> str:
        """Get random action with forward bias"""
        if np.random.random() < self.forward_prob:
            return "move_forward"
        else:
            return np.random.choice(["turn_left", "turn_right"])

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
        return "move_forward"  # Default action

    def is_done(self, observations: Dict[str, np.ndarray],
                step_count: int) -> bool:
        """Stop when sequence is complete"""
        return self.current_step >= len(self.action_sequence)


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
                    action = "move_forward"
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    action = "turn_left"
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    action = "turn_right"
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

        return action

    def _render_observations(self, observations: Dict[str, np.ndarray]):
        """Render observations on pygame screen"""
        self.screen.fill((0, 0, 0))  # Clear screen

        # Display RGB image if available
        if 'rgb' in observations:
            rgb_image = observations['rgb']

            if len(rgb_image.shape) == 3:
                # Ensure image is in correct format for pygame
                if rgb_image.dtype != np.uint8:
                    if rgb_image.max() <= 1.0:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    else:
                        rgb_image = rgb_image.astype(np.uint8)

                # Handle RGBA vs RGB
                if rgb_image.shape[2] == 4:
                    rgb_image = rgb_image[:, :, :3]
                elif rgb_image.shape[2] != 3:
                    print(f"Unexpected number of channels: {rgb_image.shape[2]}")
                    return

                try:
                    # Create pygame surface
                    rgb_surface = pygame.surfarray.make_surface(
                        rgb_image.swapaxes(0, 1)
                    )
                    scaled_surface = pygame.transform.scale(rgb_surface, self.window_size)
                    self.screen.blit(scaled_surface, (0, 0))

                except Exception as e:
                    print(f"Surface creation failed: {e}")
                    self.screen.fill((128, 128, 128))
            else:
                print(f"Unexpected image dimensions: {rgb_image.shape}")
                self.screen.fill((128, 128, 128))
        else:
            print(f"No RGB sensor found. Available: {list(observations.keys())}")
            self.screen.fill((128, 128, 128))

        # Add controls text
        controls_text = [
            "Controls:",
            "W/↑ - Move Forward",
            "A/← - Turn Left",
            "D/→ - Turn Right",
            "ESC - Quit"
        ]

        # Semi-transparent background for text
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


# Integration functions for CoreSimulator
def record_navigation(simulator,
                     strategy: BaseNavigationStrategy,
                     output_path: str,
                     fps: int = DEFAULT_FPS,
                     save_frames: bool = False) -> Dict[str, Any]:
    """Record navigation using given strategy"""

    # Setup video recording
    recorder = VideoRecorder(output_path, fps=fps)
    recorder.start_recording()

    # Reset simulator
    simulator.reset()

    # Navigation loop
    step_count = 0
    start_time = time.time()

    while True:
        # Get observations
        observations = simulator.get_observations()

        # Add frame to video
        if 'rgb' in observations:
            recorder.add_frame(observations['rgb'])

        # Check if done
        if strategy.is_done(observations, step_count):
            break

        # Get next action
        action = strategy.get_next_action(observations, step_count)

        # Take action
        simulator.step(action)
        step_count += 1

    # Finalize recording
    end_time = time.time()
    recorder.stop_recording()

    # Save frames if requested
    if save_frames:
        frames_dir = Path(output_path).parent / f"{Path(output_path).stem}_frames"
        recorder.save_frames_as_images(str(frames_dir))

    # Return statistics
    return {
        "total_steps": step_count,
        "duration": end_time - start_time,
        "video_path": str(Path(output_path).absolute()),
        "fps": fps
    }


def run_interactive_simulation(simulator):
    """Run interactive simulation with pygame visualization"""
    if not PYGAME_AVAILABLE:
        print("pygame is required for interactive mode")
        print("Install with: pip install pygame")
        return

    visualizer = PygameVisualizer(window_size=(800, 600))
    visualizer.run_interactive_simulation(simulator)