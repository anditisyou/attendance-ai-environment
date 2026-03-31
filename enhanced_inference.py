"""
Enhanced Evaluation with Multiple Agents and Visualizations
"""

import numpy as np
import random
from typing import Dict, List
from attendance_env import AttendanceEnv, Action
from advanced_agent import QLearningAgent, EnsembleAgent
from visualization import AttendanceDashboard
import json


class ImprovedStochasticAgent:
    """Improved baseline with better decision-making"""
    
    def __init__(self, uncertainty_threshold: float = 0.25, seed: int = 42):
        self.uncertainty_threshold = uncertainty_threshold
        np.random.seed(seed)
        random.seed(seed)
        
    def get_action(self, observation: Dict) -> int:
        # Reduced randomness for better performance
        if random.random() < self.uncertainty_threshold:
            return random.randint(0, 2)
        
        return self._improved_rule_based(observation)
    
    def _improved_rule_based(self, observation: Dict) -> int:
        location = observation.get("location", "").lower()
        student_id = observation.get("student_id", "")
        ambiguity = observation.get("ambiguity", 0)
        fraud_signal = observation.get("fraud_signal", False)
    
        # Fraud detection
        if fraud_signal or ambiguity > 0.7:
            return Action.FLAG_SUSPICIOUS.value
    
        # Unknown or corridor
        if "corridor" in location or "unknown" in student_id.lower():
            return Action.FLAG_SUSPICIOUS.value
    
        # ❗ NEW: Handle medium cases properly
        if ambiguity > 0.2 and ambiguity < 0.6:
            return Action.MARK_ABSENT.value
    
        # Non-classroom
        if "classroom" not in location and "lab" not in location:
            return Action.MARK_ABSENT.value
    
        # Clear valid case
        if ambiguity < 0.2:
            return Action.MARK_PRESENT.value
    
        return Action.FLAG_SUSPICIOUS.value


def comprehensive_evaluation(num_episodes_per_difficulty: int = 30):
    """Run comprehensive evaluation across multiple agents"""
    
    env = AttendanceEnv(seed=42, enable_adversarial=True)
    
    agents = {
        "Basic Stochastic": ImprovedStochasticAgent(),
        "Q-Learning (100 episodes)": QLearningAgent(),
        "Ensemble Agent": EnsembleAgent()
    }
    
    # Train Q-Learning agent
    print("Training Q-Learning agent...")
    q_agent = agents["Q-Learning (100 episodes)"]
    q_agent.train(env, episodes=100)
    
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"\nEvaluating {agent_name}...")
        agent_results = {}
        
        for difficulty in ["easy", "medium", "hard"]:
            rewards = []
            correct = 0
            fraud_detected = 0
            safe_fallback_used = 0
            total_fraud_attempts = 0
            
            for episode in range(num_episodes_per_difficulty):
                obs = env.reset(difficulty=difficulty)
                
                # Get action based on agent type
                if agent_name == "Ensemble Agent":
                    action = agent.get_action(obs, method='ensemble')
                elif agent_name == "Q-Learning (100 episodes)":
                    action = agent.get_action(obs, explore=False)
                else:
                    action = agent.get_action(obs)
                
                _, reward, done, info = env.step(action)
                
                rewards.append(reward)
                if info['ground_truth'] == Action(action).name:
                    correct += 1
                
                # Track fraud detection
                if info.get('fraud_attempt') != 'NONE':
                    total_fraud_attempts += 1
                    if action == Action.FLAG_SUSPICIOUS.value:
                        fraud_detected += 1
                
                # Track safe fallback
                if action == Action.FLAG_SUSPICIOUS.value:
                    safe_fallback_used += 1
            
            agent_results[difficulty] = {
                "avg_reward": np.mean(rewards),
                "accuracy": correct / num_episodes_per_difficulty,
                "std_reward": np.std(rewards),
                "rewards": rewards,
                "fraud_detection_rate": fraud_detected / total_fraud_attempts if total_fraud_attempts > 0 else 0,
                "safe_fallback_rate": safe_fallback_used / num_episodes_per_difficulty
            }
        
        results[agent_name] = agent_results
    
    return results, env


def generate_winning_report():
    """Generate comprehensive report for hackathon judges"""
    
    print("\n" + "="*70)
    print("🏆 HACKATHON SUBMISSION: ATTENDANCE VALIDATION ENVIRONMENT")
    print("="*70)
    print("\n🎯 KEY INNOVATIONS:")
    print("  ✓ Multi-agent adversarial testing")
    print("  ✓ Explainable AI decisions")
    print("  ✓ Advanced reward shaping with fraud bonuses")
    print("  ✓ Real-time visualization dashboard")
    print("  ✓ Production-ready deployment")
    print("  ✓ Comprehensive evaluation metrics")
    
    # Run evaluation
    results, env = comprehensive_evaluation(num_episodes_per_difficulty=30)
    
    # Display results
    print("\n📊 EVALUATION RESULTS:")
    print("-"*70)
    
    for agent_name, agent_results in results.items():
        print(f"\n🤖 {agent_name}:")
        for difficulty, metrics in agent_results.items():
            print(f"  {difficulty.upper()}:")
            print(f"    Accuracy: {metrics['accuracy']*100:.1f}%")
            print(f"    Avg Reward: {metrics['avg_reward']:.3f}")
            print(f"    Fraud Detection: {metrics['fraud_detection_rate']*100:.1f}%")
            print(f"    Safe Fallback: {metrics['safe_fallback_rate']*100:.1f}%")
    
    # Generate HTML dashboard
    dashboard = AttendanceDashboard()
    
    # Format results for dashboard
    dashboard_results = {
        "easy": results["Basic Stochastic"]["easy"],
        "medium": results["Basic Stochastic"]["medium"],
        "hard": results["Basic Stochastic"]["hard"]
    }
    
    html_report = dashboard.generate_html_report(dashboard_results, "Improved Stochastic Agent")
    
    # Save report
    with open("hackathon_report.html", "w") as f:
        f.write(html_report)
    
    print("\n✅ HTML report saved to 'hackathon_report.html'")
    
    # Calculate overall scores for leaderboard
    print("\n🏆 FINAL SCORES (Out of 100):")
    print("-"*70)
    
    for agent_name, agent_results in results.items():
        overall_score = (
            agent_results["easy"]["accuracy"] * 0.2 +
            agent_results["medium"]["accuracy"] * 0.3 +
            agent_results["hard"]["accuracy"] * 0.3 +
            agent_results["hard"]["fraud_detection_rate"] * 0.2
        ) * 100
        
        print(f"  {agent_name}: {overall_score:.1f}/100")
    
    return results


if __name__ == "__main__":
    results = generate_winning_report()