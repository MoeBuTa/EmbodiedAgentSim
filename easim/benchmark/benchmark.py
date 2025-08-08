from collections import defaultdict
from typing import Optional, Dict

from easim.utils.constants import BENCHMARK_CONFIG, AGENT_LIST
from easim.utils.habitat_utils import save_evaluation_results
from habitat import Benchmark, Agent
from tqdm import tqdm
from easim.benchmark.trial_runner import TrialRunner


class HabitatBenchmark(Benchmark):
    """
    A class to benchmark agents in Habitat environments.

    Inherits from the Habitat Benchmark class and can be extended with additional functionality if needed.
    """

    def __init__(self,  task_name: str, agent: "Agent", eval_remote=False):
        """
        Initialize the HabitatBenchmark with the given task name and remote evaluation flag.

        :param task_name: Name of the task/benchmark configuration to use.
        :param eval_remote: Boolean indicating whether to run evaluation remotely or locally.
        """
        super().__init__(BENCHMARK_CONFIG[task_name], eval_remote)
        self.task_name = task_name
        self.agent = self.initialize_agent(agent)
        self.trial_runner = TrialRunner()

    @staticmethod
    def initialize_agent(agent: "Agent") -> "Agent":
        if agent not in AGENT_LIST:
            raise ValueError(f"Agent '{agent}' is not supported. Available agents: {list(AGENT_LIST.keys())}")
        return AGENT_LIST[agent]()


    def evaluate(self, num_episodes: Optional[int] = None, enable_record: bool = False) -> Dict[str, float]:
        """
        Evaluate the agent across different task types.

        :param num_episodes: Number of episodes to run for evaluation. If None, uses the default from the environment config.
        :param enable_record: Whether to record videos and images for each episode.
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
        count_episodes = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            metrics = self.trial_runner.run_trial(
                self._env, self.agent, count_episodes, self.task_name, 
                enable_record=enable_record
            )
            
            self._accumulate_metrics(metrics, agg_metrics)
            
            count_episodes += 1
            pbar.update(1)

        pbar.close()
        
        return self._calculate_and_save_results(agg_metrics, count_episodes)

    @staticmethod
    def _accumulate_metrics(metrics: Dict, agg_metrics: Dict) -> None:
        """
        Accumulate metrics from a single episode into aggregated metrics.
        
        :param metrics: Metrics from a single episode.
        :param agg_metrics: Aggregated metrics dictionary to update.
        """
        for m, v in metrics.items():
            # Skip episode_info which contains non-numeric data
            if m == "episode_info":
                continue
                
            if isinstance(v, dict):
                for sub_m, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
            else:
                if isinstance(v, (int, float)):
                    agg_metrics[m] += v

    def _calculate_and_save_results(self, agg_metrics: Dict, count_episodes: int) -> Dict[str, float]:
        """
        Calculate average metrics and save evaluation results.
        
        :param agg_metrics: Aggregated metrics from all episodes.
        :param count_episodes: Total number of episodes evaluated.
        :return: Dictionary containing average evaluation metrics.
        """
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        
        # Extract agent information
        agent_name = getattr(self.agent, '__class__', type(self.agent)).__name__
        agent_model = getattr(self.agent, 'model', 'unknown')
        
        # Save evaluation results to CSV
        save_evaluation_results(self.task_name, avg_metrics, count_episodes, agent_name, agent_model)
        
        return avg_metrics