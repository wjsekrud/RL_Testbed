import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from trainer import train_agent_gui
from tester import test_agent_gui
from visualizer import record_agent
discrete_envs = [
    'CartPole-v1',
    'MountainCar-v0',
    'Acrobot-v1',
    'LunarLander-v2',
    # Add more discrete environments as needed
]

continuous_envs = [
    'Pendulum-v1',
    'MountainCarContinuous-v0',
    'LunarLanderContinuous-v2',
    'BipedalWalker-v3',
    # Add more continuous environments as needed
]

env_info = {
    'CartPole-v1': {'description': 'Balance a pole on a moving cart.', 'goal_reward': 500},
    'MountainCar-v0': {'description': 'Drive a car up a steep mountain.', 'goal_reward': -110},
    'Acrobot-v1': {'description': 'Swing an acrobot to reach the top.', 'goal_reward': -100},
    'LunarLander-v2': {'description': 'Safely land the lunar lander.', 'goal_reward': 200},

    'Pendulum-v1': {'description': 'Keep a pendulum upright.', 'goal_reward': 0},  # Continuous envs start here
    'MountainCarContinuous-v0': {'description': 'Drive up the mountain with continuous control.', 'goal_reward': 90},
    'LunarLanderContinuous-v2': {'description': 'Safely land the lunar lander with continuous control.', 'goal_reward': 200},
    'BipedalWalker-v3': {'description': 'Control a bipedal walker to finish the course.', 'goal_reward': 300},
}

class RLApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Reinforcement Learning GUI")
        self.geometry("800x800")

        # Default configuration
        self.config = {
            "algorithm": "A3C",
            "env_name": "CartPole-v1",
            "hyperparameters": {
                "a3c": {
                    "total_steps": 100000,
                    "learning_rate": 1e-4
                },
                "sac": {
                    "total_steps": 100000,
                    "learning_rate": 3e-4
                },
                "dpg": {
                    "total_steps": 100000,
                    "actor_lr": 1e-4,
                    "critic_lr": 1e-3
                }
            }
        }

        self.discrete_envs = discrete_envs
        self.continuous_envs = continuous_envs

        self.create_widgets()
        self.create_plot_space()

        self.iterations = []
        self.rewards = []

        self.PATH = os.getcwd()


    def create_widgets(self):
        # Algorithm Selection
        self.firstpad = ctk.CTkLabel(self, text="")
        self.firstpad.pack(anchor='w', padx=50, pady=5)

        self.algorithm_label = ctk.CTkLabel(self, text="Select Algorithm:")
        self.algorithm_label.pack(anchor='w', padx=50, pady=5)

        self.algorithm_var = tk.StringVar(value=self.config["algorithm"])
        self.algorithm_optionmenu = ctk.CTkOptionMenu(self, variable=self.algorithm_var,
                                                      values=["A3C", "SAC", "DPG"],
                                                      command=self.update_algorithm)
        self.algorithm_optionmenu.pack(anchor='w', padx=50, pady=5)

        # Environment Selection
        self.env_label = ctk.CTkLabel(self, text="Select Environment:")
        self.env_label.pack(anchor='w', padx=50, pady=5)

        self.env_var = tk.StringVar(value=self.config["env_name"])

        self.env_optionmenu_discrete = ctk.CTkOptionMenu(self, variable=self.env_var,
                                                         values=self.discrete_envs,
                                                         command=self.update_env_info)
        self.env_optionmenu_continuous = ctk.CTkOptionMenu(self, variable=self.env_var,
                                                           values=self.continuous_envs,
                                                           command=self.update_env_info)

        # Place both environment option menus but only one will be visible at a time
        self.env_optionmenu_discrete.pack(anchor='w', padx=50)
        self.env_optionmenu_continuous.pack(anchor='w', padx=100)

        # Environment Description and Goal Reward (right top)
        self.env_info_frame = ctk.CTkFrame(self)
        self.env_info_frame.place(relx=0.55, rely=0.05, anchor='n')  # Place it at the top right corner

        self.env_description_label = ctk.CTkLabel(self.env_info_frame, text="Description: ", wraplength=500)
        self.env_description_label.pack(ipadx= 50, pady=7)

        self.env_goal_label = ctk.CTkLabel(self.env_info_frame, text="Goal Reward: ")
        self.env_goal_label.pack(pady=7)

        # Hyperparameters Frame
        self.hyperparams_frame = ctk.CTkFrame(self)
        self.hyperparams_frame.pack(anchor='w', ipadx=20, padx=50, pady=50, fill="y", expand=True)

        self.hyperparams_widgets = {}
        self.create_hyperparameter_widgets()

        # Buttons
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=40)

        self.train_button = ctk.CTkButton(self.button_frame, text="Train", command=self.start_training)
        self.train_button.pack(side="left", padx=10)

        self.test_button = ctk.CTkButton(self.button_frame, text="Test", command=self.start_testing)
        self.test_button.pack(side="left", padx=10)

        self.save_config_button = ctk.CTkButton(self.button_frame, text="Save Config", command=self.save_config)
        self.save_config_button.pack(side="left", padx=10)

        self.load_config_button = ctk.CTkButton(self.button_frame, text="Load Config", command=self.load_config)
        self.load_config_button.pack(side="left", padx=10)

        # Initialize visibility of environment option menus
        self.update_algorithm()

    def create_plot_space(self):
        # Create a frame to hold the plot
        self.plot_frame = ctk.CTkFrame(self, width=600, height=400)
        self.plot_frame.place(relx = 0.65, rely=0.4, relwidth=0.6, relheight=0.4, anchor='center', bordermode="inside")

        # Initialize the figure and canvas for the matplotlib plot
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)  # A tk.DrawingArea.
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_plot(self, iteration, reward, reset=False):
        # Clear the previous plot
        self.ax.clear()
        if reset:
            self.iterations.clear()
            self.rewards.clear()
        else:
            self.iterations.append(iteration)
            self.rewards.append(reward)
        #print(f"updating plot. iterations={self.iterations}, reward={self.rewards}")
        # Plot the rewards
        self.ax.plot(self.iterations, self.rewards, label="Evaluation Rewards")
        self.ax.set_title("Evaluation Reward over Time")
        self.ax.set_xlabel("Evaluation Step")
        self.ax.set_ylabel("Reward")
        self.ax.legend()

        # Redraw the canvas with the new plot
        self.canvas.draw()

    def create_hyperparameter_widgets(self):
        # Clear previous widgets
        for widget in self.hyperparams_frame.winfo_children():
            widget.destroy()

        algorithm = self.algorithm_var.get()
        hyperparams = self.config["hyperparameters"][algorithm.lower()]

        self.hyperparams_widgets = {}
        for param, value in hyperparams.items():
            label = ctk.CTkLabel(self.hyperparams_frame, text=param)
            label.pack(pady=5)
            entry = ctk.CTkEntry(self.hyperparams_frame)
            entry.insert(0, str(value))
            entry.pack(pady=5)
            self.hyperparams_widgets[param] = entry

    def update_algorithm(self, *args):
        self.create_hyperparameter_widgets()
        algorithm = self.algorithm_var.get()
        if algorithm == 'a3c':
            # Enable discrete environments, disable continuous environments
            self.env_optionmenu_discrete.pack(after=self.env_label, anchor='w', padx=50)
            self.env_optionmenu_continuous.pack_forget()
            # Set default environment
            if self.env_var.get() not in self.discrete_envs:
                self.env_var.set(self.discrete_envs[0])
        else:
            # Enable continuous environments, disable discrete environments
            self.env_optionmenu_continuous.pack(after=self.env_label, anchor='w', padx=50)
            self.env_optionmenu_discrete.pack_forget()
            # Set default environment
            if self.env_var.get() not in self.continuous_envs:
                self.env_var.set(self.continuous_envs[0])
        self.update_env_info()
        try:
            self.update_plot(0,0,reset=True)
        except:
            pass

    def update_env_info(self, *args):
        env_name = self.env_var.get()
        description = env_info.get(env_name, {}).get('description', 'No description available.')
        goal_reward = env_info.get(env_name, {}).get('goal_reward', 'N/A')

        self.env_description_label.configure(text=f"Description: {description}")
        self.env_goal_label.configure(text=f"Goal Reward: {goal_reward}")


    def start_training(self):
        self.update_plot(0,0,True)
        # Update configuration from GUI inputs
        self.update_config_from_gui()

        # Disable buttons during training
        self.train_button.configure(state="disabled")
        self.test_button.configure(state="disabled")

        # Start training in a separate thread to keep GUI responsive
        training_thread = threading.Thread(target=self.train_agent)
        training_thread.start()

    def start_testing(self):
        # Update configuration from GUI inputs
        self.update_config_from_gui()

        # Disable buttons during testing
        self.train_button.configure(state="disabled")
        self.test_button.configure(state="disabled")

        # Start testing in a separate thread
        testing_thread = threading.Thread(target=self.test_agent(False))
        testing_thread.start()

    def train_agent(self):
        try:
            train_agent_gui(self.config, app)
            messagebox.showinfo("Training Completed", "Training has been completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            # Re-enable buttons after training
            self.train_button.configure(state="normal")
            self.test_button.configure(state="normal")

    def test_agent(self, FromAgent):
        if FromAgent == False:
            self.record_agent_gui()
        else:
            try:
                reward = test_agent_gui(self.config)
            except Exception as e:
                messagebox.showerror("Error", str(e))
            return reward

    def record_agent_gui(self):
        try:
            algorithm = self.config["algorithm"]
            env_name = self.config["env_name"]
            video_dir = f'videos\{env_name}'
            video_length = 1500  # Optionally, get this value from user input

            os.makedirs(video_dir, exist_ok=True)
            record_agent(algorithm, env_name, video_dir, video_length)
            messagebox.showinfo("Recording Completed", f"Video saved to {video_dir}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            # Re-enable buttons after recording
            self.train_button.configure(state="normal")
            self.test_button.configure(state="normal")

    def save_config(self):
        self.update_config_from_gui()
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            messagebox.showinfo("Config Saved", f"Configuration saved to {save_path}")

    def load_config(self):
        load_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if load_path:
            with open(load_path, 'r') as f:
                self.config = json.load(f)
            self.algorithm_var.set(self.config["algorithm"])
            self.env_var.set(self.config["env_name"])
            self.update_algorithm()
            messagebox.showinfo("Config Loaded", f"Configuration loaded from {load_path}")

    def update_config_from_gui(self):
        self.config["algorithm"] = self.algorithm_var.get().lower()
        self.config["env_name"] = self.env_var.get()
        algorithm = self.config["algorithm"]
        hyperparams = {}
        for param, entry in self.hyperparams_widgets.items():
            value = entry.get()
            try:
                # Convert to appropriate type
                if '.' in value:
                    hyperparams[param] = float(value)
                else:
                    hyperparams[param] = int(value)
            except ValueError:
                hyperparams[param] = value  # Keep as string if conversion fails
        self.config["hyperparameters"][algorithm] = hyperparams

if __name__ == "__main__":
    app = RLApp()
    app.mainloop()