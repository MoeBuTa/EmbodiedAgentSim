import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions


class SampleAgent(habitat.Agent):
    def reset(self):
        pass

    def act(self, observations):
        action = HabitatSimActions.turn_left  # Example action, can be changed to any valid action
        return {"action": action}
