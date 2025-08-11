"""Trial runner for executing evaluation trials with recording options"""
from pathlib import Path
from typing import Dict, Any, Optional
from habitat import Agent

from easim.recorders.video import VideoRecorder
from easim.recorders.image import ImageRecorder
from easim.utils.constants import VIDEO_DIR


class TrialRunner:
    """Handles running individual evaluation trials with optional recording"""

    def __init__(self):
        self.video_recorder = VideoRecorder()
        self.image_recorder = ImageRecorder()
        self.run_name = None

    def initialize_run(self, task_name: str) -> None:
        """
        Initialize a new run for the trial runner.
        
        :param task_name: Name of the task for directory structure.
        """
        # Set up recording directory structure following images/{task_name}/run_{}/
        from easim.utils.constants import IMAGE_DIR
        
        # Check both image and video directories to find the next run number
        image_base_dir = IMAGE_DIR / task_name
        video_base_dir = VIDEO_DIR / task_name
        
        run_number = 1
        while (image_base_dir / f"run_{run_number:03d}").exists() or (video_base_dir / f"run_{run_number:03d}").exists():
            run_number += 1

        # Generate run name for proper directory structure
        self.run_name = f"run_{run_number:03d}"

    def _setup_recording(self, episode_num: int, task_name: str, scene_id: str, observations: dict) -> None:
        """
        Set up recording for video and images.
        
        :param episode_num: Episode number for naming files.
        :param task_name: Name of the task for directory structure.
        :param scene_id: Scene ID path from the episode.
        :param observations: Initial observations to check for RGB availability.
        """
        # Use existing run_name or initialize if not set
        if self.run_name is None:
            self.initialize_run(task_name)
        
        # Generate episode filename with incremental number and scene identifier
        if '/' in scene_id:
            # Extract folder name which contains the scene identifier
            # e.g., "hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb" -> "00800-TEEsavR23oF"
            scene_identifier = scene_id.split('/')[-2]
        else:
            # Fallback for direct scene names
            scene_identifier = scene_id.split('.')[0]

        # Use episode_num + 1 for incremental naming (episode_1, episode_2, etc.)
        episode_name = f"episode_{episode_num + 1}_{scene_identifier}"
        
        # Setup recording with directory structure
        self.video_recorder.set_task_run_and_episode(task_name, self.run_name, episode_name)
        self.image_recorder.set_task_run_and_episode(task_name, self.run_name, episode_name)
        
        # Start recording
        if "rgb" in observations:
            self.video_recorder.start_recording()
        self.image_recorder.reset()

    def _run_trial_with_recording(self, env, agent: "Agent", episode_num: int, task_name: str) -> Optional[Any]:
        """
        Run a single trial with video and image recording.
        
        :param env: The habitat environment.
        :param agent: The agent to evaluate.
        :param episode_num: Episode number for naming files.
        :param task_name: Name of the task for directory structure.
        :return: Dictionary containing episode metrics.
        """
        # Reset environment to get next episode from the dataset
        observations = env.reset()
        agent.reset()

        # Setup recording
        self._setup_recording(episode_num, task_name, env.current_episode.scene_id, observations)

        try:
            while not env.episode_over:
                # Record observations
                if "rgb" in observations:
                    self.video_recorder.add_frame(observations["rgb"])
                
                self.image_recorder.add_observations(observations)
                
                action = agent.act(observations)
                observations = env.step(action)
        finally:
            self.video_recorder.stop_recording()
        
        return env.get_metrics()

    @staticmethod
    def _run_trial_no_recording(env, agent: "Agent") -> Dict:
        """
        Run a single trial without any recording.
        
        :param env: The habitat environment.
        :param agent: The agent to evaluate.
        :return: Dictionary containing episode metrics.
        """
        observations = env.reset()
        agent.reset()

        while not env.episode_over:
            action = agent.act(observations)
            observations = env.step(action)

        return env.get_metrics()

    def run_trial(self, env, agent: "Agent", episode_num: int, task_name: str, 
                  enable_record: bool = False) -> Dict[str, Any]:
        """
        Run a single evaluation trial.
        
        :param env: The habitat environment.
        :param agent: The agent to evaluate.
        :param episode_num: Episode number for naming files.
        :param task_name: Name of the task for directory structure.
        :param enable_record: Whether to record video and images.
        :return: Dictionary containing episode metrics.
        """
        if enable_record:
            return self._run_trial_with_recording(env, agent, episode_num, task_name)
        else:
            return self._run_trial_no_recording(env, agent)