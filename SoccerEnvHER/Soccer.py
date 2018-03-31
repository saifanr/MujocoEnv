import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SoccerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self._striked = False
        self._min_strike_dist = np.inf
        self.strike_threshold = 0.1
        mujoco_env.MujocoEnv.__init__(self, 'Soccer.xml', 5)

    def step(self, a):
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        d = np.linalg.norm(vec_2)

        reward = -(d > 0.05).astype(np.float32)

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()

        g_dist = np.linalg.norm(self.achieved_goal - self.desired_goal)
        is_success = (g_dist < 0.05).astype(np.float32)

        done = False
        info = {
            'is_success': is_success,
        }

        return obs, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        self._min_strike_dist = np.inf
        self._striked = False
        self._strike_pos = None

        qpos = self.init_qpos

        self.ball = np.array([0.0, 0.16])
        while True:
            self.goal = np.concatenate([
                    self.np_random.uniform(low=-0.8, high=0.8, size=1),
                    self.np_random.uniform(low=0.5, high=1.5, size=1)])
            if np.linalg.norm(self.ball - self.goal) > 0.17:
                break

        qpos[-9:-7] = [self.ball[1], self.ball[0]]
        qpos[-7:-5] = self.goal
        diff = self.ball - self.goal
        angle = -np.arctan(diff[0] / (diff[1] + 1e-8))
        qpos[-1] = angle / 3.14
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1,
                size=self.model.nv)
        qvel[6:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat[:6],
            self.sim.data.qvel.flat[:6],
            self.get_body_com("tips_foot"),
            self.get_body_com("object"),
            self.get_body_com("goal"), ])

        self.achieved_goal = np.squeeze(self.get_body_com("object"))
        self.desired_goal = np.squeeze(self.get_body_com("goal"))

        return {
            'observation': obs.copy(),
            'achieved_goal': self.achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy(),
        }

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = np.linalg.norm(achieved_goal - goal, axis=-1)
        return -(d > 0.05).astype(np.float32)
