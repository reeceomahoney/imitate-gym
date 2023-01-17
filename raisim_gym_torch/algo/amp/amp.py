import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import deque


class AMP:
    def __init__(self,
                 storage,
                 disc_ob_dim,
                 discriminator,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 mini_batch_size,
                 expert_dataset,
                 buffer_size,
                 writer,
                 learning_rate=1e-5,
                 device='cpu'):

        # core
        self.storage = storage
        self.ob_dim = disc_ob_dim // 2
        self.discriminator = discriminator
        self.obs = np.zeros([num_transitions_per_env, num_envs, self.ob_dim], dtype=np.float32)
        self.transitions = np.zeros([num_transitions_per_env - 1, num_envs, *discriminator.obs_shape], dtype=np.float32)
        self.buffer = deque([], buffer_size)

        # normalising
        self.all_obs_mean = torch.zeros(self.ob_dim*2)
        self.all_obs_var = torch.ones(self.ob_dim*2)

        # torch
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam([*self.discriminator.parameters()], lr=learning_rate, weight_decay=0.0005)
        self.device = device
        self.obs_tc = torch.from_numpy(self.obs).to(self.device).type(torch.float32)
        self.transitions_tc = torch.from_numpy(self.transitions).to(self.device).type(torch.float32)

        # Log
        self.writer = writer
        self.tot_timesteps = 0
        self.tot_time = 0

        # expert dataset
        self.expert_tc = self.process_expert_dataset(expert_dataset[:, :self.ob_dim])
        self.expert_obs_norm = torch.zeros_like(self.expert_tc)
        self.expert_size = self.expert_tc.size()[0]

        # env parameters
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env

        # learning parameters
        self.num_learning_epochs = num_learning_epochs
        self.mini_batch_size = mini_batch_size
        self.grad_pen_weight = 10

        self.update_stats = True

    # ------------------------------
    # core methods

    def step(self, obs, obs_2, obs_mean, obs_var):
        # re-normalise
        transition = self.renormalise(obs, obs_2, obs_mean, obs_var)

        with torch.no_grad():
            prediction = self.discriminator.predict(torch.from_numpy(transition).to(self.device).type(torch.float32))
        return self.get_disc_reward(prediction)

    def update(self, log_this_iteration, update):
        # remove commands from observations
        self.obs = self.storage.critic_obs[:, :, :self.ob_dim]

        # un-normalise
        obs_mean = self.storage.obs_mean[:, :, :self.ob_dim]
        obs_var = self.storage.obs_var[:, :, :self.ob_dim]
        self.obs = self.obs * np.sqrt(obs_var) + obs_mean
        self.obs_tc = torch.from_numpy(self.obs).to(self.device)

        # store the transitions in self.transitions_tc
        self.generate_transitions()

        # unroll transitions and append to buffer
        transitions_flat = self.transitions_tc.view(-1, self.transitions_tc.size()[2])
        for ts in transitions_flat:
            self.buffer.append(ts)

        # train network using supervised learning
        disc_info = self.train_step(log_this_iteration)

        if log_this_iteration:
            self.log({**disc_info, 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        self.writer.add_scalar('AMP/pred_loss', variables['mean_pred_loss'], variables['it'])
        self.writer.add_scalar('AMP/discriminator_accuracy', variables['accuracy_tot'], variables['it'])
        # self.writer.add_scalar('AMP/learning_rate', variables['learning_rate'], variables['it'])

    def train_step(self, log_this_iteration):
        mean_pred_loss = 0
        accuracy_tot = 0

        # normalise expert and agent observations
        agent_obs_norm = self.normalise_all_observations()

        # convert agent data to a pytorch dataloader
        buffer_dl = DataLoader(self.agent_to_dataset(agent_obs_norm), batch_size=self.mini_batch_size, shuffle=True)

        for epoch in range(self.num_learning_epochs):
            for buffer_sample in buffer_dl:
                # get samples and process
                expert_sample, expert_targets = self.sample_expert_dataset()

                transitions = torch.concatenate((expert_sample, buffer_sample[0]))
                targets = torch.concatenate((expert_targets, buffer_sample[1]))

                predictions = self.discriminator.evaluate(transitions)

                # Calculate loss
                pred_loss = self.calculate_prediction_loss(predictions, targets)
                grad_pen_loss = self.grad_pen_weight / 2 * self.calculate_gradient_penalty(expert_sample)
                output = pred_loss + grad_pen_loss

                self.optimizer.zero_grad()
                output.backward()
                self.optimizer.step()

                if log_this_iteration:
                    mean_pred_loss += pred_loss.item()

                    # calculate classification error
                    classifications = np.sign(predictions.cpu().detach().numpy()).reshape(-1,)
                    accuracy = [bool(x == y) for x, y in zip(classifications, targets.cpu().detach().numpy())]
                    accuracy_tot += sum(accuracy) / (self.mini_batch_size * 2)

        if log_this_iteration:
            num_mini_batches = len(buffer_dl)
            num_updates = self.num_learning_epochs * num_mini_batches
            mean_pred_loss /= num_updates
            accuracy_tot /= num_updates

        # for logging
        disc_info = {
            'mean_pred_loss': mean_pred_loss,
            'accuracy_tot': accuracy_tot,
            'learning_rate': self.learning_rate
        }

        return disc_info

    # ------------------------------
    # data processing

    def generate_transitions(self):
        self.transitions_tc[:, :, :self.ob_dim] = self.obs_tc[:-1, :, :]
        self.transitions_tc[:, :, self.ob_dim:] = self.obs_tc[1:, :, :]

    def process_expert_dataset(self, dataset):
        expert = np.concatenate((dataset[:-1, :], dataset[1:, :]), axis=1)
        expert_tc = torch.from_numpy(expert).to(self.device).type(torch.float32)
        return expert_tc

    def sample_expert_dataset(self):
        idx = np.random.randint(0, self.expert_size, self.mini_batch_size)
        expert_sample = self.expert_obs_norm[idx]
        expert_targets = torch.ones((self.mini_batch_size, 1)).to(self.device)

        return expert_sample, expert_targets

    def agent_to_dataset(self, agent_obs):
        # convert the agent obs tensor into a pytorch dataset
        targets = -torch.ones((agent_obs.size()[0], 1)).to(self.device)
        ds = torch.utils.data.TensorDataset(agent_obs, targets)

        return ds

    def normalise_all_observations(self):
        expert_obs = self.expert_tc
        agent_obs = torch.stack(list(self.buffer))

        with torch.no_grad():
            # just update stats for the first iteration so scaling is constant
            if self.update_stats is True:
                all_obs = torch.cat([expert_obs, agent_obs])
                self.all_obs_mean = torch.mean(all_obs, dim=0)
                self.all_obs_var = torch.var(all_obs, dim=0)

                self.expert_obs_norm = (expert_obs - self.all_obs_mean) / torch.sqrt(self.all_obs_var)
                self.update_stats = False

            agent_obs_norm = (agent_obs - self.all_obs_mean) / torch.sqrt(self.all_obs_var)

        return agent_obs_norm

    def renormalise(self, obs, obs_2, mean, var):
        obs_mean = mean[:self.ob_dim]
        obs_var = var[:self.ob_dim]

        obs_un_norm = obs[:, :self.ob_dim] * np.sqrt(obs_var) + obs_mean
        obs_2_un_norm = obs_2[:, :self.ob_dim] * np.sqrt(obs_var) + obs_mean

        all_obs_mean = self.all_obs_mean.cpu().detach().numpy()
        all_obs_var = self.all_obs_var.cpu().detach().numpy()

        transition = np.concatenate((obs_un_norm, obs_2_un_norm), axis=1)
        transition = (transition - all_obs_mean) / np.sqrt(all_obs_var)

        return transition

    # ------------------------------
    # training

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

    @staticmethod
    def get_disc_reward(prediction):
        style_rewards = -torch.log(1 - 1 / (1 + torch.exp(-prediction)))
        return style_rewards.cpu().detach().numpy().reshape((-1,))
