import os
from ruamel.yaml import YAML, dump, RoundTripDumper
import torch
import numpy as np

from raisim_gym_torch.env.bin.amp import RaisimGymEnv
from raisim_gym_torch.env.bin.amp import NormalSampler
from raisim_gym_torch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv

from modules import get_discriminator_from_config

# directories
home_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../')
task_path = home_path + "/gym_envs/amp"

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/resources", dump(cfg['environment'], Dumper=RoundTripDumper)))

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_discriminator():
    disc = get_discriminator_from_config(cfg, env, device)

    obs = torch.zeros((1, 66)).to(device)

    assert disc.obs_shape == [obs.size()[1]]

    disc.reset()
    assert disc._evaluate_count == 0
    assert not disc._jacobian

    with torch.no_grad():

        assert disc.evaluate(obs)
        assert disc.predict(obs)
        assert not disc.loss()


test_discriminator()
