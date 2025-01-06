from gymnasium.envs.registration import register


register(
     id="acoustic_levitation_environment_v2/GlobalTrain-v0",
     entry_point="acoustic_levitation_environment_v2.envs:GlobalTrain",
     max_episode_steps=20,
)

register(
     id="acoustic_levitation_environment_v2/GlobalEval-v0",
     entry_point="acoustic_levitation_environment_v2.envs:GlobalEval",
     max_episode_steps=20,
)

register(
     id="acoustic_levitation_environment_v2/GlobalPlanner-v0",
     entry_point="acoustic_levitation_environment_v2.envs:GlobalPlanner",
     max_episode_steps=20,
)

register(
     id="acoustic_levitation_environment_v2/GlobalRePlanner-v0",
     entry_point="acoustic_levitation_environment_v2.envs:GlobalRePlanner",
     max_episode_steps=20,
)

register(
     id="acoustic_levitation_environment_v2/GlobalPlannerAPF-v0",
     entry_point="acoustic_levitation_environment_v2.envs:GlobalPlannerAPF",
     max_episode_steps=20,
)

register(
     id="acoustic_levitation_environment_v2/GlobalRePlannerAPF-v0",
     entry_point="acoustic_levitation_environment_v2.envs:GlobalRePlannerAPF",
     max_episode_steps=20,
)