from serpent.game_agent import GameAgent

from serpent.enums import InputControlTypes

from serpent.frame_grabber import FrameGrabber
from serpent.input_controller import KeyboardKey

from serpent.machine_learning.reinforcement_learning.agents.random_agent import RandomAgent
from serpent.machine_learning.reinforcement_learning.agents.rainbow_dqn_agent import RainbowDQNAgent
from serpent.machine_learning.reinforcement_learning.agents.ppo_agent import PPOAgent

from serpent.config import config

from serpent.logger import Loggers

import serpent.cv

import time
import random


class SerpentHorizonChaseTurboGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_pause_callbacks["PLAY"] = self.handle_play_pause

    def setup_play(self):
        self.environment = self.game.environments["RACE"](
            game_api=self.game.api,
            input_controller=self.input_controller,
            episodes_per_race_track=100000000000
        )

        self.game_inputs = [
            {
                "name": "STEERING",
                "control_type": InputControlTypes.DISCRETE,
                "inputs": self.game.api.combine_game_inputs(["STEERING"])
            }
        ]

        # self.agent = RandomAgent(
        #     "horAIzon",
        #     game_inputs=self.game_inputs,
        #     callbacks=dict(
        #         after_observe=self.after_agent_observe
        #     )
        # )

        self.agent = RainbowDQNAgent(
            "horAIzon",
            game_inputs=self.game_inputs,
            callbacks=dict(
                after_observe=self.after_agent_observe,
                before_update=self.before_agent_update,
                after_update=self.after_agent_update
            ),
            rainbow_kwargs=dict(
                replay_memory_capacity=250000,
                observe_steps=10000,
                hidden_size=512,
                conv_layers=3,
                discount=0.9,
                max_steps=2000000,
                noisy_std=0.1
            ),
            logger=Loggers.COMET_ML,
            logger_kwargs=dict(
                api_key=config["comet_ml_api_key"],
                project_name="serpent-ai-hct",
                reward_func=self.reward
            )
        )

        # self.agent = PPOAgent(
        #     "horAIzon",
        #     game_inputs=self.game_inputs,
        #     callbacks=dict(
        #         after_observe=self.after_agent_observe,
        #         before_update=self.before_agent_update,
        #         after_update=self.after_agent_update
        #     ),
        #     input_shape=(100, 100),
        #     ppo_kwargs=dict(
        #         memory_capacity=5120,
        #         discount=0.9,
        #         epochs=10,
        #         batch_size=64,
        #         entropy_regularization_coefficient=0.001,
        #         epsilon=0.2
        #     ),
        #     logger=Loggers.COMET_ML,
        #     logger_kwargs=dict(
        #         api_key=config["comet_ml_api_key"],
        #         project_name="serpent-ai-hct",
        #         reward_func=self.reward
        #     )
        # )

        self.agent.logger.experiment.log_other("race_track", "Midnight")

        self.analytics_client.track(event_key="GAME_NAME", data={"name": "Horizon Chase Turbo"})

        self.environment.new_episode(maximum_steps=2400)  # 5 minutes

    def handle_play(self, game_frame, game_frame_pipeline):
        self.paused_at = None

        valid_game_state = self.environment.update_game_state(game_frame)

        if not valid_game_state:
            return None

        reward = self.reward(self.environment.game_state)

        terminal = (
            self.environment.game_state["is_too_slow"] or
            self.environment.game_state["is_out_of_fuel"] or
            self.environment.game_state["is_race_over"] or
            self.environment.episode_over
        )

        self.agent.observe(reward=reward, terminal=terminal)

        if not terminal:
            game_frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
            agent_actions = self.agent.generate_actions(game_frame_buffer)

            self.environment.perform_input(agent_actions)
        else:
            self.environment.clear_input()
            self.agent.reset()

            if self.environment.game_state["is_race_over"]:
                time.sleep(5)
                self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                time.sleep(11)

                if (self.environment.episode + 1) % self.environment.episodes_per_race_track == 0:
                    self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
                    time.sleep(8)

                    self.game.api.select_random_region_track(self.input_controller)
            else:
                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
                time.sleep(1)

                if (self.environment.episode + 1) % self.environment.episodes_per_race_track == 0:
                    for _ in range(3):
                        self.input_controller.tap_key(KeyboardKey.KEY_S)
                        time.sleep(0.1)

                    self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                    time.sleep(1)
                    self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                    time.sleep(8)
                    self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
                    time.sleep(1)
            
                    self.game.api.select_random_region_track(self.input_controller)

            self.environment.end_episode()
            self.environment.new_episode(maximum_steps=2400)

    def handle_play_pause(self):
        self.input_controller.handle_keys([])

    def reward(self, game_state):
        value = game_state["current_speed"] ** 1.5

        if value > 5200:
            value = 5200

        reward = serpent.cv.normalize(value, 0, 5200)

        if game_state["fuel_levels"][0] > game_state["fuel_levels"][1]:
            reward += 0.5

        time_penalty = 0.1
        reward -= time_penalty

        if game_state["is_race_over"]:
            reward = 1

        if reward > 1:
            reward = 1

        return reward

    def after_agent_observe(self):
        self.environment.episode_step()

    def before_agent_update(self):
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
        time.sleep(1)

    def after_agent_update(self):
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
        time.sleep(3)
