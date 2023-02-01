import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import deque


class AMP:
    def __init__(self,
                 discriminator,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 mini_batch_size,
                 expert_dataset,
                 writer,
                 learning_rate=1e-5,
                 weight_decay=0.005,
                 logit_reg_weight=0.05,
                 device='cpu'):

        # discriminator
        self.ob_dim = discriminator.obs_shape[0] // 2
        self.discriminator = discriminator

        # torch
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam([*self.discriminator.parameters()], lr=learning_rate, weight_decay=weight_decay)
        self.device = device

        # log
        self.writer = writer
        self.tot_timesteps = 0

        # expert dataset
        self.expert_tc = self.process_expert_dataset(expert_dataset)
        self.expert_tc_norm = torch.zeros_like(self.expert_tc)
        self.expert_size = self.expert_tc.size()[0]

        # normalisation
        self.env_mean = np.zeros(self.ob_dim)
        self.env_var = np.ones(self.ob_dim)
        self.trans_mean = torch.zeros(self.ob_dim * 2).to(self.device)
        self.trans_var = torch.ones(self.ob_dim * 2).to(self.device)

        # env parameters
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env

        # learning parameters
        self.num_learning_epochs = num_learning_epochs
        self.mini_batch_size = mini_batch_size
        self.grad_pen_weight = 10
        self.logit_reg_weight = logit_reg_weight

        self.buffer = deque(maxlen=1000000)

    # -----------------------------------------------------------------------------------------------------------------
    # core methods
    # -----------------------------------------------------------------------------------------------------------------

    def step(self, obs, obs_2, mean, var):
        # trim extra states
        obs = obs[:, :self.ob_dim]
        obs_2 = obs_2[:, :self.ob_dim]
        mean = mean[:self.ob_dim]
        var = var[:self.ob_dim]

        # unnormalise
        obs_un_norm = self.unnormalise(obs, mean, var)
        obs_2_un_norm = self.unnormalise(obs_2, mean, var)

        transitions = self.generate_transition(obs_un_norm, obs_2_un_norm)

        # store in buffer
        for t in transitions:
            self.buffer.append(t)

        with torch.no_grad():
            # renormalise obs for inference
            obs_re_norm = self.normalise(obs_un_norm, self.env_mean, self.env_var)
            obs_2_re_norm = self.normalise(obs_2_un_norm, self.env_mean, self.env_var)

            transitions_re_norm = self.generate_transition(obs_re_norm, obs_2_re_norm)

            predictions = self.discriminator.predict(transitions_re_norm)

        return self.get_disc_reward(predictions)

    def update(self, log_this_iteration, update, env_mean, env_var):
        # store normalising stats for expert data
        self.env_mean = env_mean[:self.ob_dim]
        self.env_var = env_var[:self.ob_dim]

        # train network using supervised learning
        disc_info = self.train_step(log_this_iteration, update)

        if log_this_iteration:
            self.log({**disc_info, 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        # self.writer.add_scalar('AMP/pred_loss', variables['mean_pred_loss'], variables['it'])
        self.writer.add_scalars('AMP/accuracy', {
            'expert_acc': variables['expert_acc'],
            'agent_acc': variables['agent_acc']
        }, variables['it'])

    def train_step(self, log_this_iteration, it):
        mean_pred_loss = 0
        expert_acc_tot, agent_acc_tot = 0, 0
        backward_count = 0

        # convert agent obs to a pytorch dataloader
        agent_dl = self.agent_to_dataloader()

        self.normalise_expert_data()

        for epoch in range(self.num_learning_epochs):
            for agent_sample, agent_targets in agent_dl:
                # get samples and process
                expert_sample, expert_targets = self.sample_expert_dataset()

                transitions = torch.concatenate((expert_sample, agent_sample))
                targets = torch.concatenate((expert_targets, agent_targets))

                predictions = self.discriminator.evaluate(transitions)

                # Calculate loss
                pred_loss = self.calculate_prediction_loss(predictions, targets)
                grad_pen_loss = self.grad_pen_weight / 2 * self.calculate_gradient_penalty(expert_sample)
                logit_loss = self.logit_reg_weight * torch.norm(predictions)
                output = pred_loss + grad_pen_loss + logit_loss

                # compute jacobian
                # self.discriminator.loss(
                #     writer=self.writer if log_this_iteration and backward_count == 0 else None, it=it)

                self.optimizer.zero_grad()
                output.backward()
                self.optimizer.step()

                backward_count += 1

                if log_this_iteration:
                    mean_pred_loss += pred_loss.item()

                    # calculate classification error
                    expert_acc = predictions[:256, :] > 0
                    expert_acc_tot += torch.mean(expert_acc.float()).item()
                    agent_acc = predictions[256:, :] < 0
                    agent_acc_tot += torch.mean(agent_acc.float()).item()

        if log_this_iteration:
            num_mini_batches = len(agent_dl)
            num_updates = self.num_learning_epochs * num_mini_batches
            mean_pred_loss /= num_updates
            expert_acc_tot /= num_updates
            agent_acc_tot /= num_updates

        # for logging
        disc_info = {
            'mean_pred_loss': mean_pred_loss,
            'expert_acc': expert_acc_tot,
            'agent_acc': agent_acc_tot,
            'learning_rate': self.learning_rate
        }

        return disc_info

    # -----------------------------------------------------------------------------------------------------------------
    # data processing
    # -----------------------------------------------------------------------------------------------------------------

    def generate_transition(self, obs, obs_2):
        transition = np.concatenate((obs[:, :self.ob_dim], obs_2[:, :self.ob_dim]), axis=1)
        transition = torch.from_numpy(transition).to(self.device).type(torch.float32)

        return transition

    def process_expert_dataset(self, dataset):
        expert = np.concatenate((dataset[:-1, :], dataset[1:, :]), axis=1)
        expert_tc = torch.from_numpy(expert).to(self.device).type(torch.float32)

        return expert_tc

    def sample_expert_dataset(self):
        idx = np.random.randint(0, self.expert_size, self.mini_batch_size)
        expert_sample = self.expert_tc_norm[idx]
        expert_targets = torch.ones((self.mini_batch_size, 1)).to(self.device)

        return expert_sample, expert_targets

    def normalise_expert_data(self):
        self.expert_tc_norm = self.normalise(self.expert_tc, self.trans_mean, self.trans_var)

    def agent_to_dataloader(self):
        # convert the agent obs into a pytorch dataloader
        transitions = torch.stack(list(self.buffer))

        self.trans_mean = torch.from_numpy(
            np.concatenate([self.env_mean, self.env_mean])).to(self.device)
        self.trans_var = torch.from_numpy(
            np.concatenate([self.env_var, self.env_var])).to(self.device)
        transitions_norm = self.normalise(transitions, self.trans_mean, self.trans_var)

        targets = -torch.ones((transitions_norm.size()[0], 1)).to(self.device)
        dataset = torch.utils.data.TensorDataset(transitions_norm, targets)
        dl = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)

        return dl

    # -----------------------------------------------------------------------------------------------------------------
    # training
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def calculate_prediction_loss(predictions, targets):
        loss = torch.nn.MSELoss()
        return loss(predictions, targets)

    def calculate_gradient_penalty(self, expert_sample):
        expert_pred = self.discriminator.evaluate(expert_sample.requires_grad_())
        grad = torch.autograd.grad(inputs=expert_sample,
                                   outputs=expert_pred,
                                   grad_outputs=torch.ones_like(expert_pred).to(self.device),
                                   create_graph=True,
                                   retain_graph=True)[0]
        grad_pen = torch.square(torch.linalg.norm(grad))

        return grad_pen

    def get_disc_reward(self, prediction):
        # style_rewards = -torch.log(1 - 1 / (1 + torch.exp(-prediction)))
        style_rewards = torch.max(torch.zeros(1).to(self.device), 1 - 0.25 * torch.square(prediction - 1))
        return style_rewards.cpu().detach().numpy().reshape((-1,))

    @staticmethod
    def unnormalise(obs, mean, var):
        if type(obs) is np.ndarray:
            return obs * np.sqrt(var) + mean
        else:
            return obs * torch.sqrt(var) + mean

    @staticmethod
    def normalise(obs, mean, var):
        if type(obs) is np.ndarray:
            return (obs - mean) / np.sqrt(var)
        else:
            return (obs - mean) / torch.sqrt(var)

