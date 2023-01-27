import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class AMP:
    def __init__(self,
                 discriminator,
                 ppo,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 mini_batch_size,
                 expert_dataset,
                 writer,
                 learning_rate=1e-5,
                 device='cpu'):

        # core
        self.ppo = ppo
        self.ob_dim = discriminator.obs_shape[0] // 2
        self.discriminator = discriminator
        self.obs = np.zeros([num_transitions_per_env, num_envs, self.ob_dim], dtype=np.float32)

        # torch
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam([*self.discriminator.parameters()], lr=learning_rate, weight_decay=0.0005)
        self.device = device
        self.obs_tc = torch.from_numpy(self.obs).to(self.device).type(torch.float32)

        # transitions
        transitions = np.zeros([(num_transitions_per_env - 1) * num_envs, *discriminator.obs_shape], dtype=np.float32)
        self.transitions_tc = torch.from_numpy(transitions).to(self.device).type(torch.float32)
        transitions_stacked = np.zeros([num_transitions_per_env - 1, num_envs, *discriminator.obs_shape],
                                       dtype=np.float32)
        self.transitions_tc_stacked = torch.from_numpy(transitions_stacked).to(self.device).type(torch.float32)

        # Log
        self.writer = writer
        self.tot_timesteps = 0
        self.tot_time = 0

        # expert dataset
        self.expert_tc = self.process_expert_dataset(expert_dataset)
        self.expert_tc_norm = torch.zeros_like(self.expert_tc)
        self.expert_size = self.expert_tc.size()[0]
        self.obs_mean = np.zeros(self.ob_dim)
        self.obs_var = np.zeros(self.ob_dim)

        # env parameters
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env

        # learning parameters
        self.num_learning_epochs = num_learning_epochs
        self.mini_batch_size = mini_batch_size
        self.grad_pen_weight = 10

    # -----------------------------------------------------------------------------------------------------------------
    # core methods
    # -----------------------------------------------------------------------------------------------------------------

    def step(self, obs, obs_2):
        transition = np.concatenate((obs, obs_2), axis=1)

        with torch.no_grad():
            prediction = self.discriminator.predict(torch.from_numpy(transition).to(self.device).type(torch.float32))
        return self.get_disc_reward(prediction)

    def update(self, log_this_iteration, update, obs_mean, obs_var):
        # store normalising stats for expert data
        self.obs_mean = obs_mean
        self.obs_var = obs_var

        # remove commands from observations
        self.obs = self.ppo.storage.obs[..., :self.ob_dim]
        self.obs_tc = torch.from_numpy(self.obs).to(self.device)

        # store the transitions in self.transitions_tc
        self.generate_transitions()

        # train network using supervised learning
        disc_info = self.train_step(log_this_iteration)

        if log_this_iteration:
            self.log({**disc_info, 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        # self.writer.add_scalar('AMP/pred_loss', variables['mean_pred_loss'], variables['it'])
        self.writer.add_scalar('AMP/discriminator_accuracy', variables['accuracy_tot'], variables['it'])

    def train_step(self, log_this_iteration):
        mean_pred_loss = 0
        accuracy_tot = 0

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
                logit_loss = 0.05 * torch.norm(predictions)
                output = pred_loss + grad_pen_loss + logit_loss

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
            num_mini_batches = len(agent_dl)
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

    # -----------------------------------------------------------------------------------------------------------------
    # data processing
    # -----------------------------------------------------------------------------------------------------------------

    def generate_transitions(self):
        # reshape observation into transitions
        self.transitions_tc_stacked[..., :self.ob_dim] = self.obs_tc[:-1, ...]
        self.transitions_tc_stacked[..., self.ob_dim:] = self.obs_tc[1:, ...]

        # unroll
        self.transitions_tc = self.transitions_tc_stacked.view(-1, self.transitions_tc_stacked.size()[2])

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
        mean = np.concatenate([self.obs_mean, self.obs_mean])
        mean = torch.from_numpy(mean).to(self.device).type(torch.float32)

        var = np.concatenate([self.obs_var, self.obs_var])
        var = torch.from_numpy(var).to(self.device).type(torch.float32)

        self.expert_tc_norm = (self.expert_tc - mean) / torch.sqrt(var)

    def agent_to_dataloader(self):
        # convert the agent obs into a pytorch dataloader
        targets = -torch.ones((self.transitions_tc.size()[0], 1)).to(self.device)
        dataset = torch.utils.data.TensorDataset(self.transitions_tc, targets)
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

    @staticmethod
    def get_disc_reward(prediction):
        style_rewards = -torch.log(1 - 1 / (1 + torch.exp(-prediction)))
        return style_rewards.cpu().detach().numpy().reshape((-1,))
