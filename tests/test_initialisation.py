import os
import psutil
import time

from ruamel.yaml import YAML, dump, RoundTripDumper
import numpy as np

from raisim_gym_torch.env.bin.amp import RaisimGymEnv
from raisim_gym_torch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv

# directories
home_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../')
task_path = home_path + "/gym_envs/amp"

# config
cfg = YAML().load(open(task_path + "/cfg_amp.yaml", 'r'))
if cfg['environment']['num_threads'] == 'auto':
    cfg['environment']['num_threads'] = psutil.cpu_count(logical=False)

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/resources", dump(cfg['environment'], Dumper=RoundTripDumper)))

env.turn_on_visualization()
env.reset()


def test_initialisation():
    for _ in range(1):

        env.reset()
        frame_start = time.time()

        obs = env.observe(False)
        print("real foot positions: ", obs[:, 33:45])

        # act = np.zeros((240, 12)).astype('float32')
        # env.step(act)

        frame_end = time.time()

        # wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        # if wait_time > 0.:
        #     time.sleep(wait_time)
        time.sleep(1)


test_initialisation()
