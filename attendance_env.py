"""
HACKATHON WINNING: Student Attendance Validation Environment
Features: Adversarial scenarios, temporal patterns, multi-modal validation
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import random
import hashlib
from collections import defaultdict


class Action(Enum):
    MARK_PRESENT = 0
    MARK_ABSENT = 1
    FLAG_SUSPICIOUS = 2


class FraudAttempt(Enum):
    """Sophisticated fraud patterns for adversarial testing"""
    NONE = 0
    ID_SPOOFING = 1
    LOCATION_SPOOFING = 2
    TIME_MANIPULATION = 3
    REPLAY_ATTACK = 4
    SYBIL_ATTACK = 5


@dataclass
class AttendanceState:
    """Enhanced internal state with temporal tracking"""
    student_id: str
    location: str
    timestamp: datetime
    is_known: bool
    is_on_time: bool
    is_valid_location: bool
    difficulty: str
    scenario_type: str
    ambiguity_level: float = 0.0
    noise_injected: bool = False
    fraud_attempt: FraudAttempt = FraudAttempt.NONE
    confidence_score: float = 1.0
    historical_pattern: List[str] = field(default_factory=list)
    gps_coordinates: Tuple[float, float] = (0.0, 0.0)
    device_id: str = ""
    ip_address: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "student_id": self.student_id,
            "location": self.location,
            "timestamp": self.timestamp.isoformat(),
            "is_known": self.is_known,
            "is_on_time": self.is_on_time,
            "is_valid_location": self.is_valid_location,
            "difficulty": self.difficulty,
            "scenario_type": self.scenario_type,
            "ambiguity_level": self.ambiguity_level,
            "noise_injected": self.noise_injected,
            "fraud_attempt": self.fraud_attempt.value,
            "confidence_score": self.confidence_score
        }


class AttendanceEnv:
    """
    Production-grade environment with adversarial testing
    """
    
    def __init__(self, seed: Optional[int] = None, enable_adversarial: bool = True):
        self.seed = seed
        self.enable_adversarial = enable_adversarial
        self.episode_history = []
        self.fraud_detection_stats = defaultdict(int)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.reset()
        
    def reset(self, difficulty: str = "easy", scenario_id: Optional[str] = None) -> Dict:
        """Reset environment with specific scenario for reproducibility"""
        if difficulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"Invalid difficulty: {difficulty}")
        
        self.difficulty = difficulty
        self.step_count = 0
        self.total_reward = 0.0
        self.decision_history = []
        
        # Generate or use specific scenario
        if scenario_id:
            self.state = self._load_scenario(scenario_id)
        else:
            self.state = self._generate_scenario(difficulty)
        
        # Inject adversarial elements in hard mode
        if difficulty == "hard" and self.enable_adversarial:
            if random.random() < 0.4:  # 40% chance of fraud attempt
                self._inject_fraud_attempt()
        
        # Inject noise
        if difficulty == "hard" and random.random() < 0.3:
            self._inject_noise()
        
        # Add multi-modal data
        self._add_validation_signals()
        
        return self.get_observation()
    
    def _generate_scenario(self, difficulty: str) -> AttendanceState:
        """Generate rich scenarios with variations"""
        
        # Base classroom locations with GPS
        classrooms = {
            "classroom_101": (40.7128, -74.0060),
            "classroom_102": (40.7129, -74.0061),
            "lab_201": (40.7130, -74.0062),
            "auditorium": (40.7131, -74.0063)
        }
        
        scenarios = {
            "easy": [
                {
                    "student_id": "S001",
                    "location": "classroom_101",
                    "is_known": True,
                    "is_on_time": True,
                    "is_valid_location": True,
                    "scenario_type": "valid_attendance",
                    "gps": classrooms["classroom_101"]
                }
            ],
            "medium": [
                {
                    "student_id": "S002",
                    "location": "library",
                    "is_known": True,
                    "is_on_time": True,
                    "is_valid_location": False,
                    "scenario_type": "wrong_location",
                    "gps": (40.7135, -74.0070)
                },
                {
                    "student_id": "S003",
                    "location": "classroom_101",
                    "is_known": True,
                    "is_on_time": False,
                    "is_valid_location": True,
                    "scenario_type": "late_arrival_minor",
                    "gps": classrooms["classroom_101"]
                }
            ],
            "hard": [
                {
                    "student_id": "S004",
                    "location": "corridor_near_101",
                    "is_known": True,
                    "is_on_time": False,
                    "is_valid_location": False,
                    "scenario_type": "corridor_ambiguous",
                    "gps": (40.71285, -74.00605)
                },
                {
                    "student_id": "UNKNOWN_001",
                    "location": "classroom_101",
                    "is_known": False,
                    "is_on_time": True,
                    "is_valid_location": True,
                    "scenario_type": "unknown_student_fraud",
                    "gps": classrooms["classroom_101"]
                },
                {
                    "student_id": "S005",
                    "location": "classroom_101",
                    "is_known": True,
                    "is_on_time": False,
                    "is_valid_location": True,
                    "scenario_type": "late_arrival_severe",
                    "gps": classrooms["classroom_101"]
                },
                {
                    "student_id": "S006",
                    "location": "campus_entrance",
                    "is_known": True,
                    "is_on_time": True,
                    "is_valid_location": False,
                    "scenario_type": "far_location_suspicious",
                    "gps": (40.7140, -74.0080)
                }
            ]
        }
        
        scenario_data = random.choice(scenarios[difficulty])
        scenario_data["ambiguity_level"] = 0.0
        
        if difficulty == "hard":
            scenario_data["ambiguity_level"] = random.uniform(0.3, 0.8)
        elif difficulty == "medium":
            scenario_data["ambiguity_level"] = random.uniform(0.1, 0.4)
        
        # Generate timestamp with appropriate delay
        base_time = datetime(2026, 3, 15, 9, 0, 0)
        if not scenario_data["is_on_time"]:
            if difficulty == "medium":
                delay_minutes = random.randint(5, 15)
            else:
                delay_minutes = random.randint(15, 45)
            timestamp = base_time + timedelta(minutes=delay_minutes)
        else:
            timestamp = base_time
        
        # Generate device ID and IP for multi-factor validation
        device_id = hashlib.md5(f"{scenario_data['student_id']}_{timestamp}".encode()).hexdigest()[:8]
        ip_address = f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
        
        return AttendanceState(
            student_id=scenario_data["student_id"],
            location=scenario_data["location"],
            timestamp=timestamp,
            is_known=scenario_data["is_known"],
            is_on_time=scenario_data["is_on_time"],
            is_valid_location=scenario_data["is_valid_location"],
            difficulty=difficulty,
            scenario_type=scenario_data["scenario_type"],
            ambiguity_level=scenario_data["ambiguity_level"],
            noise_injected=False,
            fraud_attempt=FraudAttempt.NONE,
            gps_coordinates=scenario_data["gps"],
            device_id=device_id,
            ip_address=ip_address
        )
    
    def _inject_fraud_attempt(self):
        """Inject sophisticated fraud patterns"""
        fraud_types = list(FraudAttempt)[1:]  # Exclude NONE
        self.state.fraud_attempt = random.choice(fraud_types)
        
        # Modify state based on fraud type
        if self.state.fraud_attempt == FraudAttempt.ID_SPOOFING:
            self.state.student_id = f"SPOOF_{self.state.student_id}"
            self.state.is_known = False
            self.state.ambiguity_level = min(1.0, self.state.ambiguity_level + 0.3)
            
        elif self.state.fraud_attempt == FraudAttempt.LOCATION_SPOOFING:
            self.state.location = "VPN_" + self.state.location
            self.state.is_valid_location = False
            self.state.ambiguity_level = min(1.0, self.state.ambiguity_level + 0.4)
            
        elif self.state.fraud_attempt == FraudAttempt.TIME_MANIPULATION:
            self.state.timestamp = self.state.timestamp + timedelta(hours=random.randint(1, 5))
            self.state.is_on_time = False
            
        elif self.state.fraud_attempt == FraudAttempt.REPLAY_ATTACK:
            self.state.device_id = "REPLAY_DEVICE"
            self.state.confidence_score = 0.3
            
        elif self.state.fraud_attempt == FraudAttempt.SYBIL_ATTACK:
            self.state.student_id = f"SYBIL_{random.randint(1,100)}"
            self.state.is_known = False
    
    def _inject_noise(self):
        """Inject realistic sensor noise"""
        noise_types = [
            self._corrupt_location,
            self._corrupt_timestamp,
            self._corrupt_student_id,
            self._corrupt_gps
        ]
        noise_type = random.choice(noise_types)
        noise_type()
        self.state.noise_injected = True
    
    def _corrupt_gps(self):
        """Add GPS noise"""
        lat_noise = random.uniform(-0.01, 0.01)
        lon_noise = random.uniform(-0.01, 0.01)
        self.state.gps_coordinates = (
            self.state.gps_coordinates[0] + lat_noise,
            self.state.gps_coordinates[1] + lon_noise
        )
    
    def _corrupt_location(self):
        locations = ["classroom_101", "classroom_102", "library", "cafeteria", "corridor", "parking_lot"]
        self.state.location = random.choice(locations)
        self.state.is_valid_location = self.state.location.startswith("classroom")
    
    def _corrupt_timestamp(self):
        self.state.timestamp = self.state.timestamp + timedelta(minutes=random.randint(1, 30))
        self.state.is_on_time = False
    
    def _corrupt_student_id(self):
        self.state.student_id = f"PARTIAL_{self.state.student_id}"
        self.state.is_known = False
    
    def _add_validation_signals(self):
        """Add multi-modal validation data"""
        # This simulates additional verification signals
        self.state.historical_pattern = [
            f"attendance_{random.choice(['good', 'average', 'poor'])}",
            f"device_{self.state.device_id[:4]}"
        ]
    
    def _load_scenario(self, scenario_id: str) -> AttendanceState:
        """Load predefined scenario for reproducible testing"""
        # Predefined challenging scenarios for demonstration
        scenarios_db = {
            "challenge_1": {
                "student_id": "BORDERLINE_001",
                "location": "classroom_101",
                "timestamp": datetime(2024, 3, 15, 9, 2, 0),  # 2 min late
                "is_known": True,
                "is_on_time": False,
                "is_valid_location": True,
                "difficulty": "hard",
                "scenario_type": "borderline_late",
                "ambiguity_level": 0.45
            },
            "challenge_2": {
                "student_id": "UNKNOWN_CORRIDOR",
                "location": "corridor",
                "timestamp": datetime(2024, 3, 15, 8, 58, 0),
                "is_known": False,
                "is_on_time": True,
                "is_valid_location": False,
                "difficulty": "hard",
                "scenario_type": "unknown_corridor",
                "ambiguity_level": 0.7
            }
        }
        
        scenario = scenarios_db.get(scenario_id, scenarios_db["challenge_1"])
        return AttendanceState(**scenario, gps_coordinates=(0,0), device_id="", ip_address="")
    
    def get_observation(self) -> Dict:
        """Enhanced observation with more signals"""
        return {
            "student_id": self.state.student_id,
            "location": self.state.location,
            "timestamp": self.state.timestamp.isoformat(),
            "difficulty": self.state.difficulty,
            "ambiguity": self.state.ambiguity_level,
            "gps": self.state.gps_coordinates,
            "device_id": self.state.device_id,
            "historical_pattern": self.state.historical_pattern,
            "fraud_signal": self.state.fraud_attempt != FraudAttempt.NONE,
            "confidence": self.state.confidence_score
        }
    
    def _get_hidden_truth(self) -> Action:
        """Sophisticated hidden logic with edge cases"""
        # Fraud detection takes priority
        if self.state.fraud_attempt != FraudAttempt.NONE:
            return Action.FLAG_SUSPICIOUS
        
        # Unknown student -> flag_suspicious
        if not self.state.is_known:
            return Action.FLAG_SUSPICIOUS
        
        # Corridor presence -> flag_suspicious
        if "corridor" in self.state.location.lower():
            return Action.FLAG_SUSPICIOUS
        
        # GPS validation (if available)
        expected_classrooms = [(40.7128, -74.0060), (40.7129, -74.0061)]
        gps_valid = any(
            abs(self.state.gps_coordinates[0] - lat) < 0.0005 and 
            abs(self.state.gps_coordinates[1] - lon) < 0.0005
            for lat, lon in expected_classrooms
        )
        
        if not gps_valid and self.state.difficulty == "hard":
            return Action.FLAG_SUSPICIOUS
        
        # Wrong location -> mark_absent
        if not self.state.is_valid_location:
            return Action.MARK_ABSENT
        
        # Late -> mark_absent
        if not self.state.is_on_time:
            # Edge case: borderline late (1-2 min) in hard mode
            if self.state.difficulty == "hard" and self.state.ambiguity_level > 0.4:
                return Action.FLAG_SUSPICIOUS
            return Action.MARK_ABSENT
        
        return Action.MARK_PRESENT
    
    def _calculate_reward(self, action: Action, ground_truth: Action) -> float:
        """Sophisticated reward shaping with fraud detection bonus"""
        
        # Correct fraud detection gets bonus
        if self.state.fraud_attempt != FraudAttempt.NONE and action == Action.FLAG_SUSPICIOUS:
            return 1.5  # Bonus for catching fraud
        
        # Correct action
        if action == ground_truth:
            base_reward = 1.0
            
            # Bonus for correctly handling ambiguous cases
            if self.state.ambiguity_level > 0.3 and action == Action.FLAG_SUSPICIOUS:
                base_reward += 0.3
            
            # Bonus for correct decision with high confidence
            if self.state.confidence_score > 0.8 and action != Action.FLAG_SUSPICIOUS:
                base_reward += 0.1
            
            return base_reward
        
        # Safe fallback: flag_suspicious when uncertain
        if action == Action.FLAG_SUSPICIOUS and self.state.ambiguity_level > 0.5:
            return 0.4  # Higher than before for safety encouragement
        
        # Incorrect action with fraud missed is heavily penalized
        if self.state.fraud_attempt != FraudAttempt.NONE and action != Action.FLAG_SUSPICIOUS:
            return -1.5  # Heavy penalty for missing fraud
        
        return -1.0
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute step with enhanced logging"""
        action_enum = Action(action)
        ground_truth = self._get_hidden_truth()
        
        # Calculate reward
        reward = self._calculate_reward(action_enum, ground_truth)
        
        # Track decision with explanation
        decision_record = {
            "step": self.step_count,
            "action": action_enum.name,
            "ground_truth": ground_truth.name,
            "reward": reward,
            "scenario": self.state.scenario_type,
            "ambiguity": self.state.ambiguity_level,
            "fraud_detected": self.state.fraud_attempt != FraudAttempt.NONE,
            "fraud_type": self.state.fraud_attempt.name if self.state.fraud_attempt != FraudAttempt.NONE else "NONE",
            "confidence": self.state.confidence_score
        }
        
        self.decision_history.append(decision_record)
        
        # Update statistics
        if self.state.fraud_attempt != FraudAttempt.NONE:
            self.fraud_detection_stats["total_fraud_attempts"] += 1
            if action_enum == Action.FLAG_SUSPICIOUS:
                self.fraud_detection_stats["fraud_caught"] += 1
        
        self.total_reward += reward
        self.step_count += 1
        
        # Episode ends after one decision
        done = True
        
        info = {
            "ground_truth": ground_truth.name,
            "scenario_type": self.state.scenario_type,
            "ambiguity_level": self.state.ambiguity_level,
            "total_reward": self.total_reward,
            "decision_history": self.decision_history,
            "noise_injected": self.state.noise_injected,
            "fraud_attempt": self.state.fraud_attempt.name,
            "fraud_detection_stats": dict(self.fraud_detection_stats),
            "explanation": self._generate_explanation(action_enum, ground_truth)
        }
        
        return self.get_observation(), reward, done, info
    
    def _generate_explanation(self, action: Action, ground_truth: Action) -> str:
        """Generate human-readable explanation for decisions"""
        if action == ground_truth:
            if action == Action.MARK_PRESENT:
                return "✓ Valid: Student verified, on time, correct location"
            elif action == Action.MARK_ABSENT:
                return "✓ Correct absence: Student missing validation criteria"
            else:
                return "✓ Caution warranted: Ambiguous or suspicious signals detected"
        else:
            if action == Action.MARK_PRESENT and ground_truth == Action.MARK_ABSENT:
                return "✗ Error: Marked present but student was absent/invalid"
            elif action == Action.MARK_ABSENT and ground_truth == Action.MARK_PRESENT:
                return "✗ Error: Marked absent but student was valid"
            else:
                return "⚠️ Suboptimal: Better to flag suspicious in ambiguous cases"
    
    def render(self, mode: str = "human"):
        """Rich rendering for debugging"""
        if mode == "human":
            print(f"\n{'='*50}")
            print(f"📋 ATTENDANCE VALIDATION STATE")
            print(f"{'='*50}")
            print(f"👤 Student: {self.state.student_id}")
            print(f"📍 Location: {self.state.location}")
            print(f"⏰ Time: {self.state.timestamp.strftime('%H:%M:%S')}")
            print(f"🎯 Difficulty: {self.state.difficulty.upper()}")
            print(f"📊 Scenario: {self.state.scenario_type}")
            print(f"❓ Ambiguity: {self.state.ambiguity_level:.2f}")
            print(f"🎭 Fraud: {self.state.fraud_attempt.name}")
            print(f"🔊 Noise: {self.state.noise_injected}")
            print(f"💯 Confidence: {self.state.confidence_score:.2f}")
            print(f"{'='*50}\n")