"""
MP3D Embodied Question Answering (EQA) task
"""
import numpy as np
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from collections import Counter

from easim.tasks.base.task import BaseTask, TaskConfig
from easim.tasks.base.episode import EQAEpisode, Position
from easim.tasks.base.metrics import BaseMetric, MetricResult
from easim.core.simulator import CoreSimulator


class EQAAccuracy(BaseMetric):
    """EQA Answer Accuracy"""

    def __init__(self):
        super().__init__("eqa_accuracy")

    def reset(self):
        self.correct_answers = 0
        self.total_answers = 0
        self.answer_details = []

    def update(self, episode, agent_path, success, predicted_answer=None, **kwargs):
        if not isinstance(episode, EQAEpisode) or predicted_answer is None:
            return None

        correct = self._check_answer_correctness(episode.answer, predicted_answer)
        self.total_answers += 1
        if correct:
            self.correct_answers += 1

        self.answer_details.append({
            'episode_id': episode.episode_id,
            'question': episode.question,
            'ground_truth': episode.answer,
            'predicted': predicted_answer,
            'correct': correct
        })

        return MetricResult(
            name=self.name,
            value=correct,
            info={
                'episode_correct': correct,
                'predicted_answer': predicted_answer,
                'ground_truth_answer': episode.answer
            }
        )

    def compute(self):
        accuracy = self.correct_answers / self.total_answers if self.total_answers > 0 else 0.0
        return MetricResult(
            name=self.name,
            value=accuracy,
            info={
                'accuracy': accuracy,
                'correct': self.correct_answers,
                'total': self.total_answers,
                'answer_details': self.answer_details
            }
        )

    def _check_answer_correctness(self, ground_truth: str, predicted: str) -> bool:
        """Check if predicted answer is correct"""
        # Normalize answers
        gt_normalized = self._normalize_answer(ground_truth)
        pred_normalized = self._normalize_answer(predicted)

        return gt_normalized == pred_normalized

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        return answer.lower().strip()


class EQAMeanReciprocalRank(BaseMetric):
    """Mean Reciprocal Rank for EQA"""

    def __init__(self):
        super().__init__("eqa_mrr")

    def reset(self):
        self.reciprocal_ranks = []

    def update(self, episode, agent_path, success, answer_rankings=None, **kwargs):
        if not isinstance(episode, EQAEpisode) or answer_rankings is None:
            return None

        # Find rank of correct answer
        ground_truth = episode.answer.lower().strip()
        rank = None

        for i, (answer, score) in enumerate(answer_rankings):
            if self._normalize_answer(answer) == ground_truth:
                rank = i + 1  # 1-indexed rank
                break

        if rank is not None:
            reciprocal_rank = 1.0 / rank
        else:
            reciprocal_rank = 0.0  # Correct answer not in rankings

        self.reciprocal_ranks.append(reciprocal_rank)

        return MetricResult(
            name=self.name,
            value=reciprocal_rank,
            info={'episode_rr': reciprocal_rank, 'rank': rank}
        )

    def compute(self):
        mrr = np.mean(self.reciprocal_ranks) if self.reciprocal_ranks else 0.0
        return MetricResult(
            name=self.name,
            value=mrr,
            info={'mrr': mrr, 'num_episodes': len(self.reciprocal_ranks)}
        )

    def _normalize_answer(self, answer: str) -> str:
        return answer.lower().strip()


@dataclass
class EQAConfig(TaskConfig):
    """EQA specific configuration"""
    # Answer vocabulary
    answer_vocab_size: int = 3000
    answer_vocab_file: Optional[str] = None

    # Question types
    question_types: List[str] = None

    # Navigation vs pure QA
    allow_navigation: bool = True
    max_navigation_steps: int = 100
    answer_after_navigation: bool = True

    # Rewards
    correct_answer_reward: float = 10.0
    wrong_answer_penalty: float = -1.0
    navigation_step_penalty: float = -0.01

    def __post_init__(self):
        super().__post_init__()

        if self.question_types is None:
            self.question_types = [
                "existence", "counting", "spatial", "color", "object_category",
                "room_type", "relative_position", "size", "material"
            ]

        self.action_space_type = "eqa"
        self.sensor_suite_type = "eqa"


class MP3DEQATask(BaseTask):
    """MP3D Embodied Question Answering task"""

    def __init__(self,
                 config: EQAConfig,
                 simulator: CoreSimulator,
                 episode_dataset: 'EQADataset',
                 metrics: Optional[List[BaseMetric]] = None):

        # Ensure we have EQA episodes
        if episode_dataset.episodes and not isinstance(episode_dataset.episodes[0], EQAEpisode):
            raise ValueError("MP3DEQATask requires EQAEpisode instances")

        super().__init__(config, simulator, episode_dataset, metrics)
        self.eqa_config = config

        # Load answer vocabulary
        self.answer_vocab = self._load_answer_vocabulary()
        self.answer_to_id = {ans: i for i, ans in enumerate(self.answer_vocab)}

        # Track navigation phase
        self.navigation_phase = True
        self.answer_given = False

    def _get_default_metrics(self) -> List[BaseMetric]:
        """Get EQA specific metrics"""
        return [
            EQAAccuracy(),
            EQAMeanReciprocalRank()
        ]

    def _load_answer_vocabulary(self) -> List[str]:
        """Load answer vocabulary"""
        if self.eqa_config.answer_vocab_file:
            # Load from file
            with open(self.eqa_config.answer_vocab_file, 'r') as f:
                vocab = [line.strip() for line in f.readlines()]
            return vocab[:self.eqa_config.answer_vocab_size]
        else:
            # Create default vocabulary from dataset
            return self._create_default_vocabulary()

    def _create_default_vocabulary(self) -> List[str]:
        """Create vocabulary from dataset answers"""
        all_answers = []
        for episode in self.episode_dataset.episodes:
            if isinstance(episode, EQAEpisode):
                all_answers.append(episode.answer.lower().strip())

        # Get most common answers
        answer_counts = Counter(all_answers)
        vocab = [ans for ans, count in answer_counts.most_common(self.eqa_config.answer_vocab_size)]

        return vocab

    def _check_success(self,
                       episode: EQAEpisode,
                       agent_position: Position,
                       observations: Dict[str, Any]) -> bool:
        """EQA success is answering correctly"""
        # Success is determined when answer is given
        return self.answer_given and hasattr(self.state, 'answer_correct')

    def _calculate_reward(self,
                          episode: EQAEpisode,
                          observations: Dict[str, Any],
                          action: str,
                          success: bool) -> float:
        """Calculate EQA reward"""
        reward = 0.0

        if action == "answer":
            # Answer action taken
            predicted_answer = self._extract_answer_from_action(observations)
            correct = self._check_answer_correctness(episode.answer, predicted_answer)

            if correct:
                reward += self.eqa_config.correct_answer_reward
                self.state.info['answer_correct'] = True
            else:
                reward += self.eqa_config.wrong_answer_penalty
                self.state.info['answer_correct'] = False

            self.answer_given = True
            self.state.info['predicted_answer'] = predicted_answer

        elif self.navigation_phase:
            # Navigation step
            reward += self.eqa_config.navigation_step_penalty

            # Collision penalty
            if observations.get('collisions', {}).get('is_collision', False):
                reward += self.config.collision_penalty

        return reward

    def _extract_answer_from_action(self, observations: Dict[str, Any]) -> str:
        """Extract answer from action/observations"""
        # This would extract the answer from agent's output
        # For now, return a placeholder
        if 'answer_prediction' in observations:
            return observations['answer_prediction']
        else:
            # Random answer for demo purposes
            return np.random.choice(self.answer_vocab)

    def _check_answer_correctness(self, ground_truth: str, predicted: str) -> bool:
        """Check if answer is correct"""
        gt_normalized = ground_truth.lower().strip()
        pred_normalized = predicted.lower().strip()
        return gt_normalized == pred_normalized

    def _get_episode_info(self, episode: EQAEpisode) -> Dict[str, Any]:
        """Get EQA episode info"""
        info = super()._get_episode_info(episode)

        # Add EQA specific information
        info.update({
            'question': episode.question,
            'question_tokens': self._tokenize_question(episode.question),
            'question_type': episode.question_type,
            'answer_type': episode.answer_type,
            'answer_vocab_size': len(self.answer_vocab),
            'navigation_phase': self.navigation_phase
        })

        return info

    def _tokenize_question(self, question: str) -> List[int]:
        """Tokenize question (placeholder)"""
        # This would use actual tokenizer
        words = question.lower().split()
        return list(range(len(words)))  # Dummy token IDs

    def reset(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset EQA task"""
        self.navigation_phase = self.eqa_config.allow_navigation
        self.answer_given = False
        return super().reset(episode_id)

    def step(self, action) -> tuple:
        """EQA step with answer handling"""
        # Handle answer action specially
        if action == "answer":
            self.navigation_phase = False

        return super().step(action)

    def get_answer_vocabulary(self) -> List[str]:
        """Get answer vocabulary"""
        return self.answer_vocab

    def get_answer_id(self, answer: str) -> int:
        """Get answer ID"""
        return self.answer_to_id.get(answer.lower().strip(), -1)

    def get_question_types(self) -> List[str]:
        """Get supported question types"""
        return self.eqa_config.question_types


class EQAEvaluator:
    """Evaluator for EQA task"""

    def __init__(self, task: MP3DEQATask):
        self.task = task

    def evaluate_agent(self,
                       agent_policy: callable,
                       num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate EQA agent"""
        from ...base.task import TaskRunner

        runner = TaskRunner(self.task)

        if num_episodes is None:
            num_episodes = len(self.task.episode_dataset)

        # Run evaluation
        results = runner.run_evaluation(
            num_episodes=num_episodes,
            action_policy=agent_policy
        )

        # Calculate EQA-specific metrics
        eqa_metrics = self._calculate_eqa_metrics(results['episode_results'])
        results['eqa_metrics'] = eqa_metrics

        return results

    def _calculate_eqa_metrics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate EQA-specific metrics"""
        # Extract answers and correctness
        correct_answers = []
        total_episodes = len(episode_results)

        question_type_accuracy = {}
        answer_type_accuracy = {}

        for result in episode_results:
            # Basic accuracy
            correct = result.get('success', False)
            correct_answers.append(correct)

            # Type-specific accuracy (would need episode metadata)
            # This is a placeholder for type-specific analysis

        overall_accuracy = np.mean(correct_answers) if correct_answers else 0.0

        return {
            'accuracy': overall_accuracy,
            'num_episodes': total_episodes,
            'correct_answers': sum(correct_answers)
        }


# Factory functions
def create_mp3d_eqa_task(split: str = "val",
                         max_episodes: Optional[int] = None,
                         **kwargs) -> MP3DEQATask:
    """Create MP3D EQA task"""
    from ....core.simulator import TaskSimulator
    from ....datasets.mp3d.eqa import MP3DEQADataset

    # Create simulator
    simulator = TaskSimulator(
        task_type="eqa",
        dataset_type="MP3D",
        **kwargs
    )

    # Load dataset
    dataset = MP3DEQADataset(split=split, max_episodes=max_episodes)

    # Create config
    config = EQAConfig(
        split=split,
        max_episodes=max_episodes,
        **kwargs
    )

    return MP3DEQATask(config, simulator, dataset)


def evaluate_eqa_agent(agent_policy: callable,
                       split: str = "val",
                       num_episodes: int = 100,
                       **kwargs) -> Dict[str, Any]:
    """Evaluate EQA agent"""
    # Create task
    task = create_mp3d_eqa_task(
        split=split,
        max_episodes=num_episodes,
        **kwargs
    )

    # Create evaluator and run evaluation
    evaluator = EQAEvaluator(task)
    return evaluator.evaluate_agent(agent_policy, num_episodes)


# Example agent policies
def random_eqa_agent(observations: Dict[str, Any], step: int) -> str:
    """Random EQA agent"""
    if step > 50:  # Answer after some navigation
        return "answer"

    actions = ["move_forward", "turn_left", "turn_right", "look_up", "look_down"]
    return np.random.choice(actions)


def no_nav_eqa_agent(observations: Dict[str, Any], step: int) -> str:
    """EQA agent that answers immediately without navigation"""
    return "answer"


# Demo function
def demo_eqa_task():
    """Demonstrate EQA task usage"""
    print("=== MP3D EQA Task Demo ===")

    try:
        # Create task
        task = create_mp3d_eqa_task(split="val", max_episodes=3)

        print(f"Created EQA task with {len(task.episode_dataset)} episodes")
        print(f"Answer vocabulary size: {len(task.get_answer_vocabulary())}")

        # Run evaluation
        evaluator = EQAEvaluator(task)
        results = evaluator.evaluate_agent(
            agent_policy=random_eqa_agent,
            num_episodes=3
        )

        print("\nEvaluation Results:")
        for metric, value in results['eqa_metrics'].items():
            print(f"  {metric}: {value:.4f}")

    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure MP3D EQA dataset is available")


if __name__ == "__main__":
    demo_eqa_task()