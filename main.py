# MEC (Consciousness Emergence Metric) - Basic Implementation
# Repository: consciousness-metric-mec
# Author: Daniel Alejandro Gascón Castaño

import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

class ConsciousAgent:
    """Basic agent with simulated bodily needs"""
    
    def __init__(self):
        # Basic needs: Energy, Homeostasis, Protection
        self.needs = {
            'energy': 0.7,
            'homeostasis': 0.8,
            'protection': 0.6
        }
        
        # Need decay rates
        self.decay_rates = {
            'energy': 0.05,
            'homeostasis': 0.03,
            'protection': 0.02
        }
        
        # Evolutionary weights
        self.weights = {
            'energy': 0.4,
            'homeostasis': 0.3,
            'protection': 0.3
        }
        
        # History tracking
        self.need_history = []
        self.conflict_history = []
        self.survival_history = []
        self.mec_history = []
        
    def calculate_needs_conflict(self) -> float:
        """Calculate conflict between needs (CIN)"""
        needs_values = list(self.needs.values())
        n = len(needs_values)
        total_conflict = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                conflict = abs(needs_values[i] - needs_values[j])
                total_conflict += conflict * self.weights[list(self.needs.keys())[i]]
        
        return total_conflict
    
    def calculate_survival_efficacy(self, threats: List[float]) -> float:
        """Calculate survival efficacy (EAP)"""
        if not threats:
            return 0.0
            
        # Simulate survival decisions
        survival_score = 0
        for threat in threats:
            # Agent tries to protect most threatened need
            most_threatened = min(self.needs, key=self.needs.get)
            
            # Survival decision based on need importance
            if self.needs[most_threatened] > 0.3:  # Has resources to respond
                survival_score += 0.8
            else:
                survival_score += 0.2
                
        return survival_score / len(threats)
    
    def calculate_anticipation(self, predictions: List[Dict]) -> float:
        """Calculate anticipation capacity (CA)"""
        if not predictions:
            return 0.0
            
        total_accuracy = 0
        total_horizon = 0
        
        for pred in predictions:
            accuracy = pred.get('accuracy', 0)
            horizon = pred.get('horizon', 1)
            total_accuracy += accuracy * horizon
            total_horizon += horizon
            
        return total_accuracy / total_horizon if total_horizon > 0 else 0.0
    
    def calculate_mec(self, 
                     cin_weight: float = 0.4,
                     eap_weight: float = 0.4,
                     ca_weight: float = 0.2) -> float:
        """Calculate complete MEC score"""
        
        # Get individual components
        cin = self.calculate_needs_conflict()
        
        # Simulate some threats
        threats = [0.3, 0.5, 0.2]  # Example threat magnitudes
        eap = self.calculate_survival_efficacy(threats)
        
        # Simulate some predictions
        predictions = [
            {'accuracy': 0.7, 'horizon': 2},
            {'accuracy': 0.5, 'horizon': 3},
            {'accuracy': 0.8, 'horizon': 1}
        ]
        ca = self.calculate_anticipation(predictions)
        
        # Calculate MEC
        mecmec_score = (
            cin_weight * np.log(cin + 1) +
            eap_weight * eap +
            ca_weight * np.tanh(ca)
        )
        
        # Store history
        self.need_history.append(self.needs.copy())
        self.conflict_history.append(cin)
        self.mec_history.append(mecmec_score)
        
        return mecmec_score
    
    def simulate_time_step(self, external_threat: float = 0.0):
        """Simulate one time step of agent existence"""
        
        # Needs naturally decay
        for need in self.needs:
            self.needs[need] -= self.decay_rates[need]
            
            # Add some randomness
            self.needs[need] += np.random.uniform(-0.05, 0.05)
            
            # Keep within bounds
            self.needs[need] = max(0.1, min(1.0, self.needs[need]))
        
        # Apply external threat if any
        if external_threat > 0:
            most_vulnerable = max(self.needs, key=self.needs.get)
            self.needs[most_vulnerable] -= external_threat
        
        # Calculate current MEC
        current_mec = self.calculate_mec()
        return current_mec

class MECEvaluator:
    """Evaluator for measuring consciousness emergence"""
    
    def __init__(self):
        self.agents = []
        self.results = []
        
    def run_simulation(self, steps: int = 100, n_agents: int = 3):
        """Run consciousness emergence simulation"""
        
        print("🚀 Starting MEC Simulation...")
        print(f"Steps: {steps} | Agents: {n_agents}")
        print("-" * 40)
        
        for agent_id in range(n_agents):
            agent = ConsciousAgent()
            mec_trajectory = []
            
            for step in range(steps):
                # Varying external threats
                threat = 0.1 if step % 20 < 10 else 0.3
                
                # Simulate step
                mec = agent.simulate_time_step(threat)
                mec_trajectory.append(mec)
                
                # Progress indicator
                if step % 20 == 0:
                    level = self._get_consciousness_level(mec)
                    print(f"Agent {agent_id} | Step {step:3d} | MEC: {mec:.3f} | Level: {level}")
            
            # Store results
            final_mec = mec_trajectory[-1]
            final_level = self._get_consciousness_level(final_mec)
            
            self.results.append({
                'agent_id': agent_id,
                'final_mec': final_mec,
                'final_level': final_level,
                'trajectory': mec_trajectory
            })
            
            print(f"Agent {agent_id} complete. Final MEC: {final_mec:.3f} ({final_level})")
            print("-" * 40)
        
        return self.results
    
    def _get_consciousness_level(self, mec: float) -> str:
        """Convert MEC score to consciousness level"""
        if mec < 0.2:
            return "Level 0: Reactive"
        elif mec < 0.5:
            return "Level 1: Incipient"
        elif mec < 0.8:
            return "Level 2: Functional"
        else:
            return "Level 3: Full"
    
    def plot_results(self):
        """Plot MEC trajectories"""
        plt.figure(figsize=(10, 6))
        
        for result in self.results:
            trajectory = result['trajectory']
            plt.plot(trajectory, label=f"Agent {result['agent_id']}")
        
        # Add consciousness level thresholds
        plt.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Level 0→1')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Level 1→2')
        plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Level 2→3')
        
        plt.title('Consciousness Emergence (MEC) Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('MEC Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('mec_simulation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Plot saved as 'mec_simulation.png'")

def main():
    """Main demonstration of MEC framework"""
    
    print("=" * 50)
    print("🧠 MEC FRAMEWORK - Consciousness Emergence Metric")
    print("Implementation based on: DOI 10.5281/zenodo.17781647")
    print("=" * 50)
    
    # Create evaluator and run simulation
    evaluator = MECEvaluator()
    results = evaluator.run_simulation(steps=100, n_agents=3)
    
    # Display summary
    print("\n📈 SIMULATION SUMMARY:")
    print("-" * 30)
    
    for result in results:
        print(f"Agent {result['agent_id']}:")
        print(f"  Final MEC: {result['final_mec']:.3f}")
        print(f"  Consciousness: {result['final_level']}")
        print()
    
    # Generate plot
    evaluator.plot_results()
    
    # Example single agent analysis
    print("\n🔍 SINGLE AGENT ANALYSIS:")
    agent = ConsciousAgent()
    
    print("Initial needs:", agent.needs)
    print("Initial conflict (CIN):", agent.calculate_needs_conflict())
    
    # Calculate initial MEC
    initial_mec = agent.calculate_mec()
    print(f"Initial MEC: {initial_mec:.3f}")
    print(f"Initial Level: {evaluator._get_consciousness_level(initial_mec)}")
    
    # Simulate 10 steps
    print("\n📊 Simulating 10 time steps...")
    for i in range(10):
        mec = agent.simulate_time_step()
        if i % 2 == 0:
            level = evaluator._get_consciousness_level(mec)
            print(f"Step {i}: MEC = {mec:.3f} | {level}")
    
    final_mec = agent.calculate_mec()
    print(f"\n🎯 Final MEC after 10 steps: {final_mec:.3f}")
    print(f"Final Level: {evaluator._get_consciousness_level(final_mec)}")

if __name__ == "__main__":
    main()