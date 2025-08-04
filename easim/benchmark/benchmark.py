from collections import defaultdict
from typing import Optional, Dict
from easim.utils.constants import BENCHMARK_CONFIG
from habitat import Benchmark, Agent
from tqdm import tqdm
from easim.utils.video_recorder import VideoRecorder


class HabitatBenchmark(Benchmark):
    """
    A class to benchmark agents in Habitat environments.

    Inherits from the Habitat Benchmark class and can be extended with additional functionality if needed.
    """

    def __init__(self, task_name=None, eval_remote=False):
        """
        Initialize the HabitatBenchmark with the given task name and remote evaluation flag.

        :param task_name: Name of the task/benchmark configuration to use.
        :param eval_remote: Boolean indicating whether to run evaluation remotely or locally.
        """
        super().__init__(BENCHMARK_CONFIG[task_name], eval_remote)
        self.task_name = task_name


    def evaluate(self, agent: "Agent", num_episodes: Optional[int] = None, record_video: bool = False) -> Dict[str, float]:
        """
        Evaluate the agent in the local environment.

        :param agent: The agent to evaluate.
        :param num_episodes: Number of episodes to run for evaluation. If None, uses the default from the environment config.
        :param record_video: Whether to record videos for each episode.
        :return: A dictionary containing evaluation metrics.
        """
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)
        
        # Set up video recording directory if needed
        video_dir = None
        if record_video:
            video_dir = VideoRecorder.setup_video_directory(self.task_name)
        
        count_episodes = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            if record_video:
                metrics = VideoRecorder.record_episode_with_video(
                    self._env, agent, count_episodes, video_dir
                )
            else:
                metrics = VideoRecorder.record_episode_no_video(self._env, agent)
            
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            
            count_episodes += 1
            pbar.update(1)

        pbar.close()
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        return avg_metrics