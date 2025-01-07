from gymnasium.envs.registration import register


register(
     id="acousticlevitationenvironment/GlobalTrain-v0",
     entry_point="acousticlevitationenvironment.envs:GlobalTrain",
     max_episode_steps=20,
)

register(
     id="acousticlevitationenvironment/GlobalEval-v0",
     entry_point="acousticlevitationenvironment.envs:GlobalEval",
     max_episode_steps=20,
)

register(
     id="acousticlevitationenvironment/GlobalPlanner-v0",
     entry_point="acousticlevitationenvironment.envs:GlobalPlanner",
     max_episode_steps=20,
)

register(
     id="acousticlevitationenvironment/GlobalRePlanner-v0",
     entry_point="acousticlevitationenvironment.envs:GlobalRePlanner",
     max_episode_steps=20,
)

register(
     id="acousticlevitationenvironment/GlobalPlannerAPF-v0",
     entry_point="acousticlevitationenvironment.envs:GlobalPlannerAPF",
     max_episode_steps=20,
)

register(
     id="acousticlevitationenvironment/GlobalRePlannerAPF-v0",
     entry_point="acousticlevitationenvironment.envs:GlobalRePlannerAPF",
     max_episode_steps=20,
)