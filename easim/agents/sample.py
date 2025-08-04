import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions


class SampleAgent(habitat.Agent):
    def reset(self):
        pass

    def act(self, observations):
        action = HabitatSimActions.move_forward
        return {"action": action}
