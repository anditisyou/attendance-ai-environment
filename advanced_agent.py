"""
Advanced RL Agent for Attendance Validation
Demonstrates learning and improvement over baseline
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from attendance_env import AttendanceEnv, Action
import json


class QLearningAgent:
    """Q-Learning agent that improves over time"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        self.q_table = defaultdict(lambda: np.zeros(3))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.episode_rewards = []
        
    def discretize_state(self, observation: Dict) -> str:
        """Convert continuous observation to discrete state"""
        ambiguity = int(observation['ambiguity'] * 10) / 10
        difficulty = observation['difficulty']
        location_type = "classroom" if "classroom" in observation['location'] else "other"
        known = "known" if "UNKNOWN" not in observation['student_id'] else "unknown"
        
        return f"{difficulty}_{location_type}_{known}_{ambiguity}"
    
    def get_action(self, observation: Dict, explore=True) -> int:
        """Epsilon-greedy action selection"""
        state = self.discretize_state(observation)
        
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, 3)
        
        return np.argmax(self.q_table[state])
    
    def update(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-values"""
        best_next_action = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (
            reward + self.gamma * best_next_action - self.q_table[state][action]
        )
    
    def train(self, env: AttendanceEnv, episodes: int = 100):
        """Train agent"""
        for episode in range(episodes):
            difficulty = np.random.choice(['easy', 'medium', 'hard'], p=[0.2, 0.3, 0.5])
            obs = env.reset(difficulty=difficulty)
            state = self.discretize_state(obs)
            
            action = self.get_action(obs, explore=True)
            _, reward, done, info = env.step(action)
            
            next_obs = env.get_observation()
            next_state = self.discretize_state(next_obs)
            
            self.update(state, action, reward, next_state)
            self.episode_rewards.append(reward)
            
            # Decay epsilon
            self.epsilon = max(0.05, self.epsilon * 0.995)
        
        return self.episode_rewards
    
    def evaluate(self, env: AttendanceEnv, difficulty: str, num_episodes: int = 50) -> Dict:
        """Evaluate agent performance"""
        rewards = []
        correct = 0
        
        for _ in range(num_episodes):
            obs = env.reset(difficulty=difficulty)
            action = self.get_action(obs, explore=False)
            _, reward, done, info = env.step(action)
            
            rewards.append(reward)
            if info['ground_truth'] == Action(action).name:
                correct += 1
        
        return {
            "avg_reward": np.mean(rewards),
            "accuracy": correct / num_episodes,
            "std_reward": np.std(rewards)
        }


class EnsembleAgent:
    """Ensemble of multiple strategies for robust performance"""
    
    def __init__(self):
        self.q_agent = QLearningAgent()
        self.rule_agent = None  # Will use rule-based from inference
        
    def get_action(self, observation: Dict, method='ensemble') -> int:
        """Vote or average across agents"""
        if method == 'q_learning':
            return self.q_agent.get_action(observation, explore=False)
        elif method == 'rule_based':
            # Simple rule-based logic
            if observation['ambiguity'] > 0.6:
                return Action.FLAG_SUSPICIOUS.value
            elif "corridor" in observation['location'] or "UNKNOWN" in observation['student_id']:
                return Action.FLAG_SUSPICIOUS.value
            elif "classroom" not in observation['location']:
                return Action.MARK_ABSENT.value
            else:
                return Action.MARK_PRESENT.value
        else:  # ensemble - vote
            q_action = self.q_agent.get_action(observation, explore=False)
            rule_action = Action.MARK_PRESENT.value  # Default
            
            if observation['ambiguity'] > 0.6:
                rule_action = Action.FLAG_SUSPICIOUS.value
            elif "corridor" in observation['location'] or "UNKNOWN" in observation['student_id']:
                rule_action = Action.FLAG_SUSPICIOUS.value
            elif "classroom" not in observation['location']:
                rule_action = Action.MARK_ABSENT.value
            
            # If both agree, take that action
            if q_action == rule_action:
                return q_action
            # If disagree, take safer action (flag suspicious)
            return Action.FLAG_SUSPICIOUS.value