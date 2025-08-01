import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from easim.core.habitat_wrapper import HabitatWrapper
from easim.core.object_nav_agent import RandomObjectNavAgent, HeuristicObjectNavAgent
from easim.core.types import Observation, EpisodeInfo


class ObjectNavPipeline:
    """Pipeline for running object navigation experiments."""
    
    def __init__(self, config_path: str = "benchmark/nav/objectnav/objectnav_hm3d.yaml"):
        """
        Initialize the object navigation pipeline.
        
        Args:
            config_path: Path to habitat config file
        """
        self.env = HabitatWrapper(config_path)
        self.agent = None
        self.display_enabled = True
        self.episode_stats: List[Dict[str, Any]] = []
    
    def set_agent(self, agent_type: str = "random") -> None:
        """
        Set the navigation agent.
        
        Args:
            agent_type: Type of agent ('random' or 'heuristic')
        """
        if agent_type == "random":
            self.agent = RandomObjectNavAgent()
        elif agent_type == "heuristic":
            self.agent = HeuristicObjectNavAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def run_episode(self, max_steps: int = 500, display: bool = True) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Args:
            max_steps: Maximum number of steps per episode
            display: Whether to display RGB observations
            
        Returns:
            Episode statistics
        """
        if self.agent is None:
            raise ValueError("Agent not set. Call set_agent() first.")
        
        self.display_enabled = display
        observation = self.env.reset()
        self.agent.reset()
        
        if display:
            print(f"Episode: {self.env.current_episode_info.episode_id}")
            print(f"Scene: {self.env.current_episode_info.scene_id}")
            print(f"Target object: {self.env.current_episode_info.target_object}")
            self._display_observation(observation)
        
        step_count = 0
        done = False
        
        while not done and step_count < max_steps:
            action = self.agent.act(observation)
            observation, done, info = self.env.step(action)
            step_count += 1
            
            if display:
                print(f"Step {step_count}: Action={action}, Distance={info.get('distance_to_goal', 'N/A'):.3f}")
                self._display_observation(observation)
                
                # Wait for key press or auto-advance
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Stop episode
                    done = True
        
        # Get final episode stats
        episode_stats = self._get_episode_stats(step_count)
        self.episode_stats.append(episode_stats)
        
        if display:
            self._print_episode_summary(episode_stats)
        
        return episode_stats
    
    def run_multiple_episodes(self, num_episodes: int = 10, agent_type: str = "random", 
                            display: bool = False) -> List[Dict[str, Any]]:
        """
        Run multiple episodes and collect statistics.
        
        Args:
            num_episodes: Number of episodes to run
            agent_type: Type of agent to use
            display: Whether to display observations
            
        Returns:
            List of episode statistics
        """
        self.set_agent(agent_type)
        episode_results = []
        
        for i in range(num_episodes):
            print(f"\n--- Episode {i+1}/{num_episodes} ---")
            stats = self.run_episode(display=display)
            episode_results.append(stats)
        
        self._print_overall_stats(episode_results)
        return episode_results
    
    def _display_observation(self, observation: Observation) -> None:
        """Display RGB observation."""
        if not self.display_enabled or observation.rgb is None:
            return
        
        # Convert RGB to BGR for OpenCV
        rgb_bgr = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Object Navigation", rgb_bgr)
    
    def _get_episode_stats(self, steps: int) -> Dict[str, Any]:
        """Get statistics for the completed episode."""
        info = self.env.current_episode_info
        if info is None:
            return {"steps": steps, "success": False}
        
        return {
            "episode_id": info.episode_id,
            "scene_id": info.scene_id,
            "target_object": info.target_object,
            "steps": steps,
            "success": info.success,
            "spl": info.spl,
            "distance_to_goal": info.distance_to_goal
        }
    
    def _print_episode_summary(self, stats: Dict[str, Any]) -> None:
        """Print summary of episode results."""
        print(f"\n--- Episode Summary ---")
        print(f"Success: {stats['success']}")
        print(f"Steps: {stats['steps']}")
        print(f"SPL: {stats['spl']:.3f}")
        print(f"Final distance to goal: {stats['distance_to_goal']:.3f}")
    
    def _print_overall_stats(self, episode_results: List[Dict[str, Any]]) -> None:
        """Print overall statistics across episodes."""
        if not episode_results:
            return
        
        success_rate = sum(1 for ep in episode_results if ep['success']) / len(episode_results)
        avg_steps = np.mean([ep['steps'] for ep in episode_results])
        avg_spl = np.mean([ep['spl'] for ep in episode_results])
        
        print(f"\n=== Overall Statistics ===")
        print(f"Episodes: {len(episode_results)}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average SPL: {avg_spl:.3f}")
    
    def close(self) -> None:
        """Clean up resources."""
        self.env.close()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()