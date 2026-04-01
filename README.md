---
title: Attendance AI Environment
emoji: 🎓
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---
# 🏆 Student Attendance Validation Environment

🚀 Built for OpenEnv Hackathon | Scaler x Meta

---

## 🎯 Overview

This project implements a **real-world AI training environment** for student attendance validation.  
The system simulates scenarios where an agent must decide whether a student should be:

- ✅ mark_present  
- ❌ mark_absent  
- ⚠️ flag_suspicious  
based on signals like **location, time, identity, and behavioral patterns**.
Unlike simple rule-based systems, this environment introduces:
- uncertainty
- adversarial fraud scenarios
- reward shaping
to evaluate **safe and intelligent AI decision-making**.
---
## 🧠 Key Features
### 🔍 Fraud Detection System
Simulates real-world fraud scenarios:
- ID spoofing  
- Location spoofing  
- Replay attacks  
- Time manipulation  
- Sybil attacks  
---
### 🎯 Reward Shaping (Core Innovation)
| Outcome            | Reward |
|--------------------|--------|
| Correct decision   | +1.0   |
| Safe fallback      | +0.3   |
| Incorrect decision | -1.0   |
| Fraud detected     | +1.5   |
✔ Encourages cautious AI behavior  
✔ Penalizes overconfidence  
✔ Supports partial correctness  
---
### 🤖 Multi-Agent Evaluation
Includes multiple agents:
- Stochastic Agent  
- Q-Learning Agent  
- Ensemble Agent  
✔ Enables comparative evaluation  
✔ Demonstrates learning vs rule-based behavior  
---
### 📊 Interactive Dashboard
- Performance visualization (Plotly)  
- Accuracy & fraud detection metrics  
- Leaderboard comparison  
- Confusion matrix  
---
### ⚡ Realistic Environment Design
- Noise injection  
- Ambiguity handling  
- Edge-case simulation  
- Stochastic scenario generation  
---
## 🧪 Task Design
| Difficulty | Scenario |
|------------|--------|
| Easy       | Valid student in the classroom |
| Medium     | Minor inconsistencies (late / location mismatch) |
| Hard       | Fraud attempts, ambiguous cases |
---
## 🧠 Example Decision Output
```json
{
  "decision": "FLAG_SUSPICIOUS",
  "confidence": 0.87,
  "reasoning": "Multiple inconsistencies detected",
  "risk_factors": ["identity_unknown", "location_anomaly"]
}
```
}
## 🏗️ System Architecture
**User → FastAPI → Environment → Agent → Reward System → Dashboard**
## 📊 Example Results
Easy Accuracy: ~100%
Medium Accuracy: varies (uncertainty-driven decisions)
Hard Accuracy: ~60–80%
Fraud Detection Rate: ~70%+
## 🚀 Deployment
Deployed using:
FastAPI
Docker
HuggingFace Spaces
## 🔗 **Live Demo:**
https://huggingface.co/spaces/VaishnaviKhan/Docker
## 🔗 GitHub:
https://github.com/anditisyou/attendance-ai-environment
## 💡 Why This Project Stands Out
- **Real-world problem (attendance fraud detection)**
- **Models uncertainty instead of overfitting rules**
- **Reward shaping encourages safe AI decisions**
- **Multi-agent evaluation framework**
- **Fully deployable and interactive**
## 🔮 Future Improvements
- **Biometric verification simulation**
- **Multi-agent interaction**
- **Real-time anomaly detection**
- **Adaptive policy learning**
## 👩‍💻 Author
***Vaishnavi Khandelwal***
