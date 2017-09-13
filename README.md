import os

import gym
from gym import wrappers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CartPoleQLearningAgent:
    def __init__(self,
                 learning_rate=1.0,
                 discount_factor=0.0,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.state = None
        self.action = None
        self._num_actions = 2

        # =============== TODO: Your code here ===============
        #   We'll use tabular Q-Learning in our agent, which means
        #   we need to allocate a Q-Table. To do that, we need to know
        #   the exact number of possible states in our MDP. But, unfortunately,
        #   our feature space is continuous, so the number of states is infinite.
        #
        #   So, we have to discretize the space of each of the 4 features and
        #   combine these bins into a single integer value that describes the state.
        #
        #   Observe the range of each of the 4 features returned by the environment
        #   and decide about the range of the feature bins.
        #   Try a small number of bins (5-25) and see which one works best later.
        #   After that, create a Q-Table that contains all possible states and
        #   actions, and fill it with zeros.
        #
        #   Hint: use np.linspace, np.digitize, pd.cut or similar functions.

        self.q = np.zeros((1, 1))
        # ====================================================

    def _build_state(self, observation):
        
        # =============== TODO: Your code here ===============
        #   Our observations consist of 4 features, but we need to represent
        #   the state as a number to find the corresponding row in the Q-Table.
        #   Discretize the observation features and reduce them to a single integer.
        self.observation = np.array([x, x_dot, theta, theta_dot])
        self.state = np.digitize(self.observation, bins = [0.0, 0.5, 1.0, 2.5, 5, 10])
        return self.state

        #return 0
        # ====================================================

    def begin_episode(self, observation):
        self.state = self._build_state(observation)

        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate

        # =============== TODO: Your code here ===============
        #   Based on the Q-Table, get the best action for our current state.
        self.action=np.argmax(self.q[state])
        return self.action

        #return np.random.randint(self._num_actions)
        # ====================================================

    def act(self, observation, reward):
        next_state = self._build_state(observation)

        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)

        # =============== TODO: Your code here ===============
        #   If we choose exploration (enable_exploration == True), we perform a random action.
        #   If we choose exploitation, we perform the best possible action for this state.
        if enable_exploration == True:
            next_action = np.random.randint(0, self._num_actions)
        elif 
            

        next_action = np.random.randint(0, self._num_actions)
        # ====================================================

        # =============== TODO: Your code here ===============
        #   We have received a reward from our previous step, and we know our future
        #   state and what action to perform next.
        #   Now, recalculate Q[state, action] in the Q-Table using the update formula.

        self.q[0, 0] = 0
        # ====================================================

        self.state = next_state
        self.action = next_action
        return next_action


class EpisodeHistory:
    def __init__(self,
                 capacity,
                 plot_episode_count=200,
                 max_timesteps_per_episode=200,
                 goal_avg_episode_length=195,
                 goal_consecutive_episodes=100):

        self.lengths = np.zeros(capacity, dtype=int)
        self.plot_episode_count = plot_episode_count
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

        self.point_plot = None
        self.mean_plot = None
        self.fig = None
        self.ax = None

    def __getitem__(self, episode_index):
        return self.lengths[episode_index]

    def __setitem__(self, episode_index, episode_length):
        self.lengths[episode_index] = episode_length

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 7), facecolor='w', edgecolor='k')
        self.fig.canvas.set_window_title("Episode Length History")

        self.ax.set_xlim(0, self.plot_episode_count + 5)
        self.ax.set_ylim(0, self.max_timesteps_per_episode + 5)
        self.ax.yaxis.grid(True)

        self.ax.set_title("Episode Length History")
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Length, timesteps")

        self.point_plot, = plt.plot([], [], linewidth=2.0, c="#1d619b")
        self.mean_plot, = plt.plot([], [], linewidth=3.0, c="#df3930")

    def update_plot(self, episode_index):
        plot_right_edge = episode_index
        plot_left_edge = max(0, plot_right_edge - self.plot_episode_count)

        # Update point plot.
        x = range(plot_left_edge, plot_right_edge)
        y = self.lengths[plot_left_edge:plot_right_edge]
        self.point_plot.set_xdata(x)
        self.point_plot.set_ydata(y)
        self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update rolling mean plot.
        mean_kernel_size = 101
        rolling_mean_data = np.concatenate((np.zeros(mean_kernel_size), self.lengths[plot_left_edge:episode_index]))
        rolling_means = pd.rolling_mean(
            rolling_mean_data,
            window=mean_kernel_size,
            min_periods=0
        )[mean_kernel_size:]
        self.mean_plot.set_xdata(range(plot_left_edge, plot_left_edge + len(rolling_means)))
        self.mean_plot.set_ydata(rolling_means)

        # Repaint the surface.
        plt.draw()
        plt.pause(0.0001)

    def is_goal_reached(self, episode_index):
        avg = np.average(self.lengths[episode_index - self.goal_consecutive_episodes + 1:episode_index + 1])
        return avg >= self.goal_avg_episode_length


def log_timestep(index, action, reward, observation):
    format_string = "   ".join([
        "Timestep: {0:3d}",
        "Action: {1:2d}",
        "Reward: {2:5.1f}",
        "Cart Position: {3:6.3f}",
        "Cart Velocity: {4:6.3f}",
        "Angle: {5:6.3f}",
        "Tip Velocity: {6:6.3f}"
    ])
    print(format_string.format(index, action, reward, *observation))


def run_agent(env, verbose=False):
    max_episodes_to_run = 5000
    max_timesteps_per_episode = 200

    goal_avg_episode_length = 195
    goal_consecutive_episodes = 100

    plot_episode_count = 200
    plot_redraw_frequency = 10

    # =============== TODO: Your code here ===============
    #   Create a Q-Learning agent with proper parameters.
    #   Think about what learning rate and discount factor
    #   would be reasonable in this environment.

    agent = CartPoleQLearningAgent(

    )
    # ====================================================

    episode_history = EpisodeHistory(
        capacity=max_episodes_to_run,
        plot_episode_count=plot_episode_count,
        max_timesteps_per_episode=max_timesteps_per_episode,
        goal_avg_episode_length=goal_avg_episode_length,
        goal_consecutive_episodes=goal_consecutive_episodes
    )
    episode_history.create_plot()

    for episode_index in range(max_episodes_to_run):
        observation = env.reset()
        action = agent.begin_episode(observation)

        for timestep_index in range(max_timesteps_per_episode):
            # Perform the action and observe the new state.
            observation, reward, done, info = env.step(action)

            # Update the display and log the current state.
            if verbose:
                env.render()
                log_timestep(timestep_index, action, reward, observation)

            # If the episode has ended prematurely, penalize the agent.
            if done and timestep_index < max_timesteps_per_episode - 1:
                reward = -max_episodes_to_run

            # Get the next action from the agent, given our new state.
            action = agent.act(observation, reward)

            # Record this episode to the history and check if the goal has been reached.
            if done or timestep_index == max_timesteps_per_episode - 1:
                print("Episode {} finished after {} timesteps.".format(episode_index + 1, timestep_index + 1))

                episode_history[episode_index] = timestep_index + 1
                if verbose or episode_index % plot_redraw_frequency == 0:
                    episode_history.update_plot(episode_index)

                if episode_history.is_goal_reached(episode_index):
                    print()
                    print("Goal reached after {} episodes!".format(episode_index + 1))
                    return episode_history

                break

    print("Goal not reached after {} episodes.".format(max_episodes_to_run))
    return episode_history


def save_history(history, experiment_dir):
    # Save the episode lengths to CSV.
    filename = os.path.join(experiment_dir, "episode_history.csv")
    dataframe = pd.DataFrame(history.lengths, columns=["length"])
    dataframe.to_csv(filename, header=True, index_label="episode")


def main():
    random_state = 0
    experiment_dir = "cartpole-qlearning-1"

    env = gym.make("CartPole-v0")
    env.seed(random_state)
    np.random.seed(random_state)

    env = gym.wrappers.Monitor(env, experiment_dir, force=True)
    #env.monitor.start(experiment_dir, force=True, resume=False, seed=random_state)
    episode_history = run_agent(env, verbose=True)   # Set verbose=False to greatly speed up the process.
    save_history(episode_history, experiment_dir)
    #env.monitor.close()


if __name__ == "__main__":
    main()
