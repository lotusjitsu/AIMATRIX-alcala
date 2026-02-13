"""
AIMATRIX Agent Loader
Loads and manages agent configurations from YAML files
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Agent configuration data class"""
    name: str
    description: str
    version: str
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    triggers: List[Dict[str, str]] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    integrations: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)


class AgentLoader:
    """Loads and manages AIMATRIX agents"""

    def __init__(self, agents_dir: Optional[str] = None):
        if agents_dir is None:
            # Default to .claude/agents in AIMATRIX folder
            base_path = Path(__file__).parent.parent.parent
            self.agents_dir = base_path / ".claude" / "agents"
        else:
            self.agents_dir = Path(agents_dir)

        self.agents: Dict[str, AgentConfig] = {}
        self.loaded = False

    def load_all_agents(self) -> Dict[str, AgentConfig]:
        """Load all agent configurations from YAML files"""
        if not self.agents_dir.exists():
            print(f"[AgentLoader] Agents directory not found: {self.agents_dir}")
            return {}

        for yaml_file in self.agents_dir.glob("*.yaml"):
            try:
                agent = self._load_agent_file(yaml_file)
                if agent:
                    self.agents[agent.name] = agent
                    print(f"[AgentLoader] Loaded agent: {agent.name}")
            except Exception as e:
                print(f"[AgentLoader] Error loading {yaml_file.name}: {e}")

        self.loaded = True
        return self.agents

    def _load_agent_file(self, filepath: Path) -> Optional[AgentConfig]:
        """Load a single agent configuration file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        return AgentConfig(
            name=data.get('name', filepath.stem),
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            capabilities=data.get('capabilities', []),
            tools=data.get('tools', []),
            triggers=data.get('triggers', []),
            settings=data.get('settings', {}),
            integrations=data.get('integrations', {}),
            monitoring=data.get('monitoring', {})
        )

    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name"""
        if not self.loaded:
            self.load_all_agents()
        return self.agents.get(name)

    def get_agent_for_keyword(self, keyword: str) -> Optional[AgentConfig]:
        """Find the most appropriate agent for a given keyword"""
        if not self.loaded:
            self.load_all_agents()

        keyword_lower = keyword.lower()
        for agent in self.agents.values():
            for trigger in agent.triggers:
                if 'keyword' in trigger:
                    if trigger['keyword'].lower() in keyword_lower or keyword_lower in trigger['keyword'].lower():
                        return agent
        return None

    def list_agents(self) -> List[str]:
        """List all loaded agent names"""
        if not self.loaded:
            self.load_all_agents()
        return list(self.agents.keys())

    def get_agent_capabilities(self, name: str) -> List[str]:
        """Get capabilities for a specific agent"""
        agent = self.get_agent(name)
        return agent.capabilities if agent else []

    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities for all agents"""
        if not self.loaded:
            self.load_all_agents()
        return {name: agent.capabilities for name, agent in self.agents.items()}


# Singleton instance
_agent_loader: Optional[AgentLoader] = None


def get_agent_loader() -> AgentLoader:
    """Get or create the agent loader singleton"""
    global _agent_loader
    if _agent_loader is None:
        _agent_loader = AgentLoader()
    return _agent_loader


def load_agents() -> Dict[str, AgentConfig]:
    """Convenience function to load all agents"""
    return get_agent_loader().load_all_agents()


def get_agent(name: str) -> Optional[AgentConfig]:
    """Convenience function to get an agent by name"""
    return get_agent_loader().get_agent(name)


def find_agent_for_task(task_description: str) -> Optional[AgentConfig]:
    """Find the best agent for a given task description"""
    loader = get_agent_loader()

    # Check for keyword matches
    keywords = ['trade', 'mt5', 'crypto', 'helium', 'network', 'security',
                'optimize', 'system', 'monitor', 'bot']

    for keyword in keywords:
        if keyword in task_description.lower():
            agent = loader.get_agent_for_keyword(keyword)
            if agent:
                return agent

    # Default to system agent
    return loader.get_agent('system_agent')


if __name__ == "__main__":
    # Test the agent loader
    print("Loading AIMATRIX Agents...")
    agents = load_agents()

    print(f"\nLoaded {len(agents)} agents:")
    for name, agent in agents.items():
        print(f"\n  {name}:")
        print(f"    Description: {agent.description}")
        print(f"    Version: {agent.version}")
        print(f"    Capabilities: {len(agent.capabilities)}")
        print(f"    Tools: {len(agent.tools)}")
