import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import json
import os

from trainer import train_agent_gui
from tester import test_agent_gui

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

class RLApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Reinforcement Learning GUI")
        self.geometry("800x600")

        # Default configuration
        self.config = {
            "algorithm": "a3c",
            "env_name": "CartPole-v1",
            "hyperparameters": {
                "a3c": {
                    "max_steps": 1000,
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

    def create_widgets(self):
        # Algorithm Selection
        self.firstpad = ctk.CTkLabel(self, text="")
        self.firstpad.pack(anchor='w', padx=50, pady=5)

        self.algorithm_label = ctk.CTkLabel(self, text="Select Algorithm:")
        self.algorithm_label.pack(anchor='w', padx=50, pady=5)

        self.algorithm_var = tk.StringVar(value=self.config["algorithm"])
        self.algorithm_optionmenu = ctk.CTkOptionMenu(self, variable=self.algorithm_var,
                                                      values=["a3c", "sac", "dpg"],
                                                      command=self.update_algorithm)
        self.algorithm_optionmenu.pack(anchor='w', padx=50, pady=5)

        # Environment Selection
        self.env_label = ctk.CTkLabel(self, text="Select Environment:")
        self.env_label.pack(anchor='w', padx=50, pady=5)

        self.env_var = tk.StringVar(value=self.config["env_name"])

        self.env_optionmenu_discrete = ctk.CTkOptionMenu(self, variable=self.env_var,
                                                         values=self.discrete_envs)
        self.env_optionmenu_continuous = ctk.CTkOptionMenu(self, variable=self.env_var,
                                                           values=self.continuous_envs)

        # Place both environment option menus but only one will be visible at a time
        self.env_optionmenu_discrete.pack(anchor='w', padx=50)
        self.env_optionmenu_continuous.pack(anchor='w', padx=100)

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

    def create_hyperparameter_widgets(self):
        # Clear previous widgets
        for widget in self.hyperparams_frame.winfo_children():
            widget.destroy()

        algorithm = self.algorithm_var.get()
        hyperparams = self.config["hyperparameters"][algorithm]

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

    def start_training(self):
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
        testing_thread = threading.Thread(target=self.test_agent)
        testing_thread.start()

    def train_agent(self):
        try:
            train_agent_gui(self.config)
            messagebox.showinfo("Training Completed", "Training has been completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            # Re-enable buttons after training
            self.train_button.configure(state="normal")
            self.test_button.configure(state="normal")

    def test_agent(self):
        try:
            test_agent_gui(self.config)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            # Re-enable buttons after testing
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
        self.config["algorithm"] = self.algorithm_var.get()
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