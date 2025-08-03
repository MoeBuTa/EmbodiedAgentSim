from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import numpy as np

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def transform_semantic_for_display(semantic):
    """Convert semantic segmentation to displayable format"""

    # Normalize to 0-255 range for display
    semantic_display = ((semantic.astype(np.float32) / semantic.max()) * 255).astype(np.uint8)
    return semantic_display


def example():
    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    )
    print("Environment creation successful")
    observations = env.reset()
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
    cv2.imshow("DEPTH", observations["depth"])


    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1
        print(observations)

        cv2.imshow("RGB", observations["rgb"])
        cv2.imshow("DEPTH", observations["depth"])



    print("Episode finished after {} steps.".format(count_steps))

    if action == HabitatSimActions.stop:
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()