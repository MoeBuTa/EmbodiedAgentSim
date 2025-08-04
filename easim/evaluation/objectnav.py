#!/usr/bin/env python3

import argparse

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from easim.benchmark.benchmark import HabitatBenchmark
from easim.utils import setup_habitat_lab_env
from easim.utils.constants import BENCHMARK_CONFIG

setup_habitat_lab_env()

class ForwardOnlyAgent(habitat.Agent):
    def reset(self):
        pass

    def act(self, observations):
        action = HabitatSimActions.move_forward
        return {"action": action}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name",
        type=str,
        default="objectnav_hm3d",
    )
    args = parser.parse_args()

    agent = ForwardOnlyAgent()
    benchmark = HabitatBenchmark(args.task_name)
    metrics = benchmark.evaluate(agent, num_episodes=10, record_video=True)

    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
