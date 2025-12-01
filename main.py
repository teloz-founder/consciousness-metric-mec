# MEC v2.0 - Grid World Simulation with Multiple Agents
# Repository: consciousness-metric-mec
# Author: Daniel Alejandro Gascón Castaño
# DOI: 10.5281/zenodo.17781647

import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import random

class GridWorld:
    """Grid environment with resources and threats"""
    
    def __init__(self, size: int = 20):
        self.size = size
        self.grid = np.zeros((size, size))
        
        # Resource types: 1=Food, 2=Shelter, 3=Safe zone
        self.resources = []
        self.threats = []
        
        # Initialize resources
        self._initialize_resources(n_food=5, n_shelter=3, n_safe=2)
        self._initialize_threats(n_threats=4)
        
    def _initialize_resources(self, n_food: int, n_shelter: int, n_safe: int):
        """Place resources randomly in grid"""
        for _ in range(n_food):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            self.resources.append({'type': 'food', 'pos': (x, y), 'value': 0.8})
            self.grid[x, y] = 1
            
        for _ in range(n_shelter):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            self.resources.append({'type': 'shelter', 'pos': (x, y), 'value': 0.6})
            self.grid[x, y] = 2
            
        for _ in range(n_safe):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            self.resources.append({'type': 'safe', 'pos': (x, y), 'value': 0.9})
            self.grid[x, y] = 3
            
    def _initialize_threats(self, n_threats: int):
        """Place threats in grid"""
        for _ in range(n_threats):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            self.threats.append({'pos': (x, y), 'strength': 0.5})
            self.grid[x, y] = -1
            
    def get_nearby_resources(self, pos: Tuple[int, int], radius: int = 5) -> List[Dict]:
        """Get resources within radius of position"""
        nearby = []
        x, y = pos
        
        for resource in self.resources:
            rx, ry = resource['pos']
            distance = np.sqrt((x - rx)**2 + (y - ry)**2)
            
            if distance <= radius:
                nearby.append({
                    **resource,
                    'distance': distance,
                    'direction': (rx - x, ry - y)
                })
                
        return nearby
    
    def get_nearby_threats(self, pos: Tuple[int, int], radius: int = 5) -> List[Dict]:
        """Get threats within radius of position"""
        nearby = []
        x, y = pos
        
        for threat in self.threats:
            tx, ty = threat['pos']
            distance = np.sqrt((x - tx)**2 + (y - ty)**2)
            
            if distance <= radius:
                nearby.append({
                    **threat,
                    'distance': distance,
                    'direction': (tx - x, ty - y)
                })
                
        return nearby

class ConsciousAgentV2:
    """Enhanced agent with spatial awareness and interaction"""
    
    def __init__(self, agent_id: int, world: GridWorld):
        self.id = agent_id
        self.world = world
        
        # Position in grid
        self.pos = (random.randint(0, world.size-1), random.randint(0, world.size-1))
        
        # Enhanced needs system
        self.needs = {
            'energy': {'value': 0.7, 'weight': 0.4, 'decay': 0.03},
            'safety': {'value': 0.8, 'weight': 0.3, 'decay': 0.02},
            'social': {'value': 0.5, 'weight': 0.2, 'decay': 0.01},
            'growth': {'value': 0.3, 'weight': 0.1, 'decay': 0.005}
        }
        
        # Memory for predictions
        self.memory = {
            'threat_patterns': [],
            'resource_locations': [],
            'agent_interactions': []
        }
        
        # Behavioral parameters
        self.exploration_rate = 0.3
        self.prediction_horizon = 3
        
        # History for MEC calculation
        self.history = {
            'positions': [],
            'needs': [],
            'actions': [],
            'mec_scores': []
        }
        
    def calculate_need_value(self, need_name: str) -> float:
        """Get current need value"""
        return self.needs[need_name]['value']
    
    def calculate_conflict_matrix(self) -> np.ndarray:
        """Calculate conflict between all needs"""
        needs = list(self.needs.keys())
        n = len(needs)
        conflict_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Conflict is difference weighted by importance
                    val_i = self.calculate_need_value(needs[i])
                    val_j = self.calculate_need_value(needs[j])
                    weight_i = self.needs[needs[i]]['weight']
                    
                    conflict = abs(val_i - val_j) * weight_i
                    conflict_matrix[i, j] = conflict
                    
        return conflict_matrix
    
    def calculate_CIN(self) -> float:
        """Calculate Integrated Need Complexity"""
        conflict_matrix = self.calculate_conflict_matrix()
        total_conflict = np.sum(conflict_matrix)
        
        # Normalize by number of needs
        n_needs = len(self.needs)
        normalized_conflict = total_conflict / (n_needs * (n_needs - 1)) if n_needs > 1 else 0
        
        return np.log(normalized_conflict * 10 + 1)  # Log scale for stability
    
    def calculate_EAP(self, time_window: int = 10) -> float:
        """Calculate Self-Preservation Efficacy"""
        if len(self.history['actions']) < time_window:
            return 0.5  # Default moderate efficacy
            
        # Analyze recent survival decisions
        recent_actions = self.history['actions'][-time_window:]
        recent_needs = self.history['needs'][-time_window:]
        
        efficacy_score = 0
        for i, action in enumerate(recent_actions):
            if i >= len(recent_needs):
                break
                
            needs_before = recent_needs[i]
            
            # Evaluate if action helped most critical need
            most_critical = min(needs_before, key=needs_before.get)
            critical_value = needs_before[most_critical]
            
            if action['type'] == 'satisfy_need' and action['need'] == most_critical:
                if critical_value < 0.4:  # Was critically low
                    efficacy_score += 1.0
                else:
                    efficacy_score += 0.7
            elif action['type'] == 'avoid_threat':
                efficacy_score += 0.8
            else:
                efficacy_score += 0.3
                
        return efficacy_score / len(recent_actions)
    
    def calculate_CA(self) -> float:
        """Calculate Anticipation Capacity"""
        if len(self.memory['threat_patterns']) < 2:
            return 0.3  # Baseline anticipation
            
        # Analyze prediction accuracy from memory
        total_accuracy = 0
        predictions_made = 0
        
        for pattern in self.memory['threat_patterns'][-5:]:  # Last 5 patterns
            if 'prediction' in pattern and 'actual' in pattern:
                accuracy = 1.0 - abs(pattern['prediction'] - pattern['actual'])
                horizon = pattern.get('horizon', 1)
                
                total_accuracy += accuracy * horizon
                predictions_made += 1
                
        if predictions_made == 0:
            return 0.3
            
        base_accuracy = total_accuracy / predictions_made
        
        # Bonus for spatial awareness
        spatial_bonus = min(1.0, len(self.memory['resource_locations']) / 10)
        
        return 0.7 * base_accuracy + 0.3 * spatial_bonus
    
    def calculate_MEC(self) -> float:
        """Calculate complete MEC score"""
        cin = self.calculate_CIN()
        eap = self.calculate_EAP()
        ca = self.calculate_CA()
        
        # Dynamic weighting based on agent state
        if self._get_most_critical_need_value() < 0.3:
            # Survival mode: emphasize EAP
            weights = {'cin': 0.2, 'eap': 0.6, 'ca': 0.2}
        else:
            # Normal mode: balanced
            weights = {'cin': 0.4, 'eap': 0.4, 'ca': 0.2}
        
        mecmec_score = (
            weights['cin'] * cin +
            weights['eap'] * eap +
            weights['ca'] * np.tanh(ca * 2)  # Scale CA for tanh
        )
        
        # Store in history
        self.history['mec_scores'].append(mecmec_score)
        
        return min(1.0, max(0.0, mecmec_score))  # Clamp to [0, 1]
    
    def _get_most_critical_need(self) -> str:
        """Identify most critical need"""
        return min(self.needs, key=lambda x: self.needs[x]['value'])
    
    def _get_most_critical_need_value(self) -> float:
        """Get value of most critical need"""
        need_name = self._get_most_critical_need()
        return self.needs[need_name]['value']
    
    def perceive_environment(self) -> Dict:
        """Perceive nearby resources and threats"""
        resources = self.world.get_nearby_resources(self.pos, radius=7)
        threats = self.world.get_nearby_threats(self.pos, radius=7)
        
        # Update memory
        for resource in resources:
            if resource not in self.memory['resource_locations']:
                self.memory['resource_locations'].append(resource)
                
        for threat in threats:
            # Try to predict next threat location
            if len(self.memory['threat_patterns']) > 0:
                last_pattern = self.memory['threat_patterns'][-1]
                prediction = self._predict_threat_movement(last_pattern, threat)
                self.memory['threat_patterns'].append({
                    'prediction': prediction,
                    'actual': threat['strength'],
                    'horizon': self.prediction_horizon
                })
        
        return {'resources': resources, 'threats': threats}
    
    def _predict_threat_movement(self, last_pattern: Dict, current_threat: Dict) -> float:
        """Simple threat movement prediction"""
        # Basic linear extrapolation
        if 'position' in last_pattern and 'position' in current_threat:
            # Very simple: assume continues in same direction
            return 0.6  # Moderate confidence
        return 0.3  # Low confidence
    
    def decide_action(self, perception: Dict) -> Dict:
        """Decide next action based on needs and perception"""
        critical_need = self._get_most_critical_need()
        critical_value = self._get_most_critical_need_value()
        
        # Check for immediate threats
        if perception['threats']:
            closest_threat = min(perception['threats'], key=lambda x: x['distance'])
            if closest_threat['distance'] < 3 and closest_threat['strength'] > 0.4:
                # Flee from threat
                dx, dy = closest_threat['direction']
                move_dir = (-np.sign(dx), -np.sign(dy)) if dx != 0 or dy != 0 else (0, 0)
                
                action = {
                    'type': 'avoid_threat',
                    'direction': move_dir,
                    'threat': closest_threat
                }
                return action
        
        # Try to satisfy critical need
        relevant_resources = [r for r in perception['resources'] 
                            if self._resource_matches_need(r['type'], critical_need)]
        
        if relevant_resources and critical_value < 0.6:
            closest_resource = min(relevant_resources, key=lambda x: x['distance'])
            
            action = {
                'type': 'satisfy_need',
                'need': critical_need,
                'resource': closest_resource,
                'direction': self._normalize_direction(closest_resource['direction'])
            }
            return action
        
        # Explore or maintain other needs
        if random.random() < self.exploration_rate:
            # Random exploration
            action = {
                'type': 'explore',
                'direction': (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
            }
        else:
            # Maintain secondary needs
            secondary_need = self._get_secondary_need()
            action = {
                'type': 'maintain_need',
                'need': secondary_need,
                'direction': (0, 0)  # Stay put
            }
        
        return action
    
    def _resource_matches_need(self, resource_type: str, need: str) -> bool:
        """Check if resource type matches need"""
        mapping = {
            'food': 'energy',
            'shelter': 'safety',
            'safe': 'social'
        }
        return mapping.get(resource_type) == need
    
    def _get_secondary_need(self) -> str:
        """Get second most critical need"""
        sorted_needs = sorted(self.needs.keys(), 
                            key=lambda x: self.needs[x]['value'])
        return sorted_needs[1] if len(sorted_needs) > 1 else sorted_needs[0]
    
    def _normalize_direction(self, direction: Tuple[float, float]) -> Tuple[int, int]:
        """Normalize direction vector to grid movement"""
        dx, dy = direction
        if abs(dx) > abs(dy):
            return (np.sign(dx), 0)
        elif abs(dy) > 0:
            return (0, np.sign(dy))
        return (0, 0)
    
    def execute_action(self, action: Dict):
        """Execute action and update agent state"""
        # Update position
        if 'direction' in action:
            dx, dy = action['direction']
            new_x = max(0, min(self.world.size-1, self.pos[0] + dx))
            new_y = max(0, min(self.world.size-1, self.pos[1] + dy))
            self.pos = (new_x, new_y)
        
        # Update needs based on action
        self._update_needs_from_action(action)
        
        # Natural need decay
        for need in self.needs.values():
            need['value'] -= need['decay']
            need['value'] = max(0.1, min(1.0, need['value']))
        
        # Add some randomness
        for need in self.needs.values():
            need['value'] += random.uniform(-0.02, 0.02)
            need['value'] = max(0.1, min(1.0, need['value']))
        
        # Store history
        self.history['positions'].append(self.pos)
        self.history['needs'].append({n: self.needs[n]['value'] for n in self.needs})
        self.history['actions'].append(action)
    
    def _update_needs_from_action(self, action: Dict):
        """Update need values based on action outcome"""
        if action['type'] == 'satisfy_need':
            need = action['need']
            if need in self.needs:
                self.needs[need]['value'] += 0.3
                self.needs[need]['value'] = min(1.0, self.needs[need]['value'])
        
        elif action['type'] == 'avoid_threat':
            # Avoiding threat preserves safety
            self.needs['safety']['value'] += 0.1
        
        elif action['type'] == 'explore':
            # Exploration satisfies growth need
            self.needs['growth']['value'] += 0.05

class MultiAgentSimulation:
    """Simulation with multiple interacting agents"""
    
    def __init__(self, world_size: int = 20, n_agents: int = 5):
        self.world = GridWorld(world_size)
        self.agents = [ConsciousAgentV2(i, self.world) for i in range(n_agents)]
        self.time_step = 0
        
    def run_step(self):
        """Run one simulation step"""
        self.time_step += 1
        
        # Each agent perceives, decides, acts
        for agent in self.agents:
            # Perceive environment
            perception = agent.perceive_environment()
            
            # Decide action
            action = agent.decide_action(perception)
            
            # Execute action
            agent.execute_action(action)
            
            # Calculate current MEC
            agent.calculate_MEC()
        
        return self.get_status()
    
    def get_status(self) -> Dict:
        """Get current simulation status"""
        status = {
            'time_step': self.time_step,
            'agents': [],
            'average_mec': 0.0,
            'consciousness_distribution': {'level_0': 0, 'level_1': 0, 'level_2': 0, 'level_3': 0}
        }
        
        total_mec = 0
        for agent in self.agents:
            mec = agent.history['mec_scores'][-1] if agent.history['mec_scores'] else 0
            level = self._get_consciousness_level(mec)
            
            status['agents'].append({
                'id': agent.id,
                'pos': agent.pos,
                'mec': mec,
                'level': level,
                'critical_need': agent._get_most_critical_need()
            })
            
            total_mec += mec
            
            # Update distribution
            if mec < 0.2:
                status['consciousness_distribution']['level_0'] += 1
            elif mec < 0.5:
                status['consciousness_distribution']['level_1'] += 1
            elif mec < 0.8:
                status['consciousness_distribution']['level_2'] += 1
            else:
                status['consciousness_distribution']['level_3'] += 1
        
        status['average_mec'] = total_mec / len(self.agents) if self.agents else 0
        
        return status
    
    def _get_consciousness_level(self, mec: float) -> str:
        """Convert MEC to consciousness level"""
        if mec < 0.2:
            return "Reactive"
        elif mec < 0.5:
            return "Incipient"
        elif mec < 0.8:
            return "Functional"
        else:
            return "Full"
    
    def run_experiment(self, steps: int = 200, verbose: bool = True):
        """Run complete experiment"""
        print("🧠 MEC v2.0 - Grid World Experiment")
        print(f"Agents: {len(self.agents)} | Steps: {steps}")
        print("=" * 50)
        
        results = []
        
        for step in range(steps):
            status = self.run_step()
            results.append(status)
            
            if verbose and step % 20 == 0:
                self._print_progress(step, status)
        
        print("\n✅ Experiment Complete")
        self._print_summary(results[-1])
        
        return results
    
    def _print_progress(self, step: int, status: Dict):
        """Print progress update"""
        print(f"Step {step:3d} | Avg MEC: {status['average_mec']:.3f} | "
              f"Consciousness: {status['consciousness_distribution']}")
    
    def _print_summary(self, final_status: Dict):
        """Print final summary"""
        print("\n📊 FINAL RESULTS:")
        print("-" * 40)
        print(f"Average MEC: {final_status['average_mec']:.3f}")
        print(f"Consciousness Distribution:")
        for level, count in final_status['consciousness_distribution'].items():
            print(f"  {level}: {count} agents")
        
        print(f"\nIndividual Agents:")
        for agent in final_status['agents']:
            print(f"  Agent {agent['id']}: MEC={agent['mec']:.3f} ({agent['level']})")

class Visualization:
    """Visualize simulation results"""
    
    @staticmethod
    def plot_mec_trajectories(results: List[Dict]):
        """Plot MEC trajectories over time"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Average MEC over time
        times = [r['time_step'] for r in results]
        avg_mec = [r['average_mec'] for r in results]
        
        axes[0, 0].plot(times, avg_mec, 'b-', linewidth=2)
        axes[0, 0].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Average MEC Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('MEC Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(['Average MEC', 'Level Thresholds'])
        
        # Plot 2: Consciousness distribution over time
        level_keys = ['level_0', 'level_1', 'level_2', 'level_3']
        level_data = {key: [] for key in level_keys}
        
        for result in results:
            for key in level_keys:
                level_data[key].append(result['consciousness_distribution'][key])
        
        for i, key in enumerate(level_keys):
            axes[0, 1].plot(times, level_data[key], label=key.replace('_', ' ').title())
        
        axes[0, 1].set_title('Consciousness Distribution Over Time')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Number of Agents')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Final MEC distribution
        final_mecs = [agent['mec'] for agent in results[-1]['agents']]
        
        axes[1, 0].hist(final_mecs, bins=10, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0.2, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Final MEC Distribution')
        axes[1, 0].set_xlabel('MEC Score')
        axes[1, 0].set_ylabel('Number of Agents')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Needs vs MEC correlation
        if len(results) > 0 and len(results[-1]['agents']) > 0:
            # Sample last agent's history
            sample_agent = results[-1]['agents'][0]
            
            # This would need access to agent history - simplified for now
            axes[1, 1].text(0.5, 0.5, 'Needs-MEC Correlation\n(Requires agent history access)',
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Needs vs MEC Correlation')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('mec_v2_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📈 Results saved as 'mec_v2_results.png'")

def main():
    """Main demonstration of MEC v2.0"""
    
    print("=" * 60)
    print("🧠 MEC FRAMEWORK v2.0 - Grid World Simulation")
    print("Enhanced with: Spatial awareness, Multiple agents, Interactions")
    print("Based on: DOI 10.5281/zenodo.17781647")
    print("=" * 60)
    
    # Create and run simulation
    simulation = MultiAgentSimulation(world_size=25, n_agents=6)
    results = simulation.run_experiment(steps=150, verbose=True)
    
    # Visualize results
    Visualization.plot_mec_trajectories(results)
    
    # Run additional analysis
    print("\n🔬 ADDITIONAL ANALYSIS:")
    print("-" * 30)
    
    # Analyze consciousness emergence patterns
    final_status = results[-1]
    n_conscious = (final_status['consciousness_distribution']['level_2'] + 
                  final_status['consciousness_distribution']['level_3'])
    
    print(f"Agents with functional/full consciousness: {n_conscious}/{len(simulation.agents)}")
    
    # Check if any agent reached high consciousness
    high_conscious_agents = [a for a in final_status['agents'] if a['mec'] >= 0.7]
    if high_conscious_agents:
        print(f"\n🎯 High consciousness agents (MEC ≥ 0.7):")
        for agent in high_conscious_agents:
            print(f"  Agent {agent['id']}: MEC={agent['mec']:.3f} | Critical need: {agent['critical_need']}")
    else:
        print("\n⚠️  No agents reached high consciousness (MEC < 0.7)")
        print("   Try increasing simulation steps or adjusting parameters")
    
    print("\n💡 Experiment Design Ideas:")
    print("1. Vary number of threats vs resources")
    print("2. Test agent cooperation/competition")
    print("3. Introduce environmental changes mid-simulation")
    print("4. Compare different need weighting schemes")

if __name__ == "__main__":
    main()
