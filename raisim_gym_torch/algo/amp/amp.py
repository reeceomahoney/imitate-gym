import numpy as np
import torch
import torch.optim as optim
from .amp_replay_buffer import AMPReplayBuffer


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
                 buffer_size=1000000,
                 device='cpu'):

        # discriminator
        self.ob_dim = discriminator.obs_shape[0] // 2
        self.discriminator = discriminator

        self.replay_buffer = AMPReplayBuffer(buffer_size)

        # torch
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam([*self.discriminator.parameters()], lr=learning_rate, weight_decay=weight_decay)
        self.device = device

        # log
        self.writer = writer

        # expert dataset
        self.expert_tc = self.process_expert_dataset(expert_dataset)
        self.expert_size = self.expert_tc.size()[0]

        # normalisation
        self.env_mean = np.zeros(self.ob_dim)
        self.env_var = np.ones(self.ob_dim)
        self.trans_mean = torch.zeros(self.ob_dim * 2).to(self.device)
        self.trans_var = torch.ones(self.ob_dim * 2).to(self.device)

        # learning parameters
        self.num_learning_epochs = num_learning_epochs
        self.mini_batch_size = mini_batch_size
        self.grad_pen_weight = 10
        self.logit_reg_weight = logit_reg_weight

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

        self.replay_buffer.store(transitions)

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
        self.trans_mean = torch.from_numpy(np.concatenate([self.env_mean, self.env_mean])).to(self.device)
        self.env_var = env_var[:self.ob_dim]
        self.trans_var = torch.from_numpy(np.concatenate([self.env_var, self.env_var])).to(self.device)

        # train network using supervised learning
        disc_info = self.train_step(log_this_iteration)

        if log_this_iteration:
            self.log({**disc_info, 'it': update})

    def log(self, variables):
        self.writer.add_scalars('AMP/loss', {
            'pred_loss': variables['mean_pred_loss'],
            'grad_pen_loss': variables['mean_grad_pen_loss'],
            'logit_loss': variables['mean_logit_loss']
        }, variables['it'])
        self.writer.add_scalars('AMP/accuracy', {
            'expert_acc': variables['expert_acc'],
            'agent_acc': variables['agent_acc']
        }, variables['it'])
        self.writer.add_scalars('AMP/logits', {
            'expert_logits': variables['expert_logits'],
            'agent_logits': variables['agent_logits']
        }, variables['it'])

    def train_step(self, log_this_iteration):
        mean_pred_loss, mean_grad_pen_loss, mean_logit_loss = 0, 0, 0
        expert_acc_tot, agent_acc_tot = 0, 0
        mean_expert_logits, mean_agent_logits = 0, 0

        num_mini_batches = len(self.replay_buffer) // self.mini_batch_size

        for epoch in range(self.num_learning_epochs):
            for batch in range(num_mini_batches):
                # get samples
                expert_sample = self.sample_expert_dataset()
                expert_sample = self.normalise(expert_sample, self.trans_mean, self.trans_var)

                agent_sample = self.replay_buffer.sample(self.mini_batch_size)
                agent_sample = torch.squeeze(torch.stack(agent_sample))
                agent_sample = self.normalise(agent_sample, self.trans_mean, self.trans_var)

                # calculate logits
                expert_logits = self.discriminator.evaluate(expert_sample)
                agent_logits = self.discriminator.evaluate(agent_sample)

                # calculate loss
                pred_loss = 0.5 * (self.prediction_loss_pos(expert_logits) + self.prediction_loss_neg(agent_logits))
                grad_pen_loss = self.grad_pen_weight / 2 * self.calculate_gradient_penalty(expert_sample)
                logit_loss = self.logit_reg_weight * torch.norm(torch.cat([expert_logits, agent_logits], dim=1))
                output = pred_loss + grad_pen_loss + logit_loss

                self.optimizer.zero_grad()
                output.backward()
                self.optimizer.step()

                if log_this_iteration:
                    mean_pred_loss += pred_loss.item()
                    mean_grad_pen_loss += grad_pen_loss.item()
                    mean_logit_loss += logit_loss.item()

                    mean_expert_logits += torch.mean(expert_logits).item()
                    mean_agent_logits += torch.mean(agent_logits).item()

                    # calculate classification error
                    expert_acc = expert_logits > 0
                    expert_acc_tot += torch.mean(expert_acc.float()).item()
                    agent_acc = agent_logits < 0
                    agent_acc_tot += torch.mean(agent_acc.float()).item()

        if log_this_iteration:
            num_updates = self.num_learning_epochs * num_mini_batches
            mean_pred_loss /= num_updates
            mean_grad_pen_loss /= num_updates
            mean_logit_loss /= num_updates
            expert_acc_tot /= num_updates
            agent_acc_tot /= num_updates
            mean_expert_logits /= num_updates
            mean_agent_logits /= num_updates

        # for logging
        disc_info = {
            'mean_pred_loss': mean_pred_loss,
            'mean_grad_pen_loss': mean_grad_pen_loss,
            'mean_logit_loss': mean_logit_loss,
            'expert_acc': expert_acc_tot,
            'agent_acc': agent_acc_tot,
            'expert_logits': mean_expert_logits,
            'agent_logits': mean_agent_logits,
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
        expert_sample = self.expert_tc[idx]

        return expert_sample

    # -----------------------------------------------------------------------------------------------------------------
    # training
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def prediction_loss_pos(disc_logits):
        loss = torch.nn.MSELoss()
        return loss(disc_logits, torch.ones_like(disc_logits))

    @staticmethod
    def prediction_loss_neg(disc_logits):
        loss = torch.nn.MSELoss()
        return loss(disc_logits, -torch.ones_like(disc_logits))

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
        style_rewards = torch.max(torch.zeros(1).to(self.device), 1 - 0.75 * torch.square(prediction - 1))
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

