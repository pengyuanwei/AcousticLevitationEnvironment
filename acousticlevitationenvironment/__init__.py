from gymnasium.envs.registration import register


register(
     id="acousticlevitationenvironment/TrainEnv-v0",
     entry_point="acousticlevitationenvironment.envs:TrainEnv",
     max_episode_steps=20,
)

register(
     id="acousticlevitationenvironment/EvalEnv-v0",
     entry_point="acousticlevitationenvironment.envs:EvalEnv",
     max_episode_steps=20,
)

register(
     id="acousticlevitationenvironment/Planner-v0",
     entry_point="acousticlevitationenvironment.envs:Planner",
     max_episode_steps=20,
)

register(
     id="acousticlevitationenvironment/RePlanner-v0",
     entry_point="acousticlevitationenvironment.envs:RePlanner",
     max_episode_steps=20,
)

register(
     id="acousticlevitationenvironment/PlannerAPF-v0",
     entry_point="acousticlevitationenvironment.envs:PlannerAPF",
     max_episode_steps=20,
)

register(
     id="acousticlevitationenvironment/RePlannerAPF-v0",
     entry_point="acousticlevitationenvironment.envs:RePlannerAPF",
     max_episode_steps=20,
)