# 🏆 Student Attendance Validation Environment

🚀 Built for OpenEnv Hackathon | Scaler x Meta

---

## 🎯 Overview

This project implements a production-grade AI training environment for student attendance validation. The system simulates real-world scenarios where an agent must decide whether a student should be marked present, absent, or suspicious based on contextual signals such as location, time, identity, and behavioral patterns.

Unlike traditional rule-based systems, this environment incorporates adversarial fraud scenarios, uncertainty modeling, and reward shaping to evaluate safe and intelligent decision-making.

---

## 🧠 Key Features

- 🔍 **Fraud Detection System**  
  Simulates real-world attacks such as ID spoofing, location spoofing, replay attacks, and Sybil attacks.

- 🎯 **Reward Shaping**  
  Encourages cautious AI behavior with partial rewards and penalties for overconfidence.

- 🤖 **Multi-Agent Evaluation**  
  Includes:
  - Stochastic Agent  
  - Q-Learning Agent  
  - Ensemble Agent  

- 📊 **Interactive Dashboard**  
  Visualizes performance using charts, confusion matrix, and metrics.

- ⚡ **Realistic Environment Design**  
  Handles ambiguity, noise injection, and edge cases.

---

## 🧪 Task Design

| Difficulty | Scenario |
|------------|--------|
| Easy       | Valid student in classroom |
| Medium     | Minor inconsistencies (late / location mismatch) |
| Hard       | Fraud attempts, ambiguity, adversarial cases |

---

## 🏆 Reward System

| Outcome            | Reward |
|--------------------|--------|
| Correct decision   | +1.0   |
| Safe fallback      | +0.3   |
| Incorrect decision | -1.0   |
| Fraud detected     | +1.5   |

---

## 📊 Example Results

- Easy Accuracy: 100%  
- Medium Accuracy: varies (uncertainty-driven decisions)  
- Hard Accuracy: ~60–80%  
- Fraud Detection: ~50%+  

---

## 🚀 Deployment

Deployed on HuggingFace Spaces using Docker + FastAPI.

🔗 Live Demo:  
https://huggingface.co/spaces/VaishnaviKhan/Docker

---

## 💡 Why This Project Stands Out

- Not a toy problem — real-world fraud detection scenario  
- Handles uncertainty instead of overfitting rules  
- Demonstrates safe AI decision-making  
- Includes learning agents and evaluation pipeline  
- Fully deployable and interactive  

---

## 👩‍💻 Author

Vaishnavi Khandelwal
