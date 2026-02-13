"""
AIMATRIX Agent System
Complete list and management of all AI agents for the AIMATRIX ecosystem

Agents handle:
- Mining operations
- Web scraping for Mysterium revenue
- Network monitoring
- Database synchronization
- Security operations
- Trading automation
- Infrastructure management
"""

import threading
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class AgentCategory(Enum):
    """Agent categories"""
    MINING = "mining"
    NETWORK = "network"
    TRADING = "trading"
    SECURITY = "security"
    DATABASE = "database"
    AI = "ai"
    SCRAPING = "scraping"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class AIAgent:
    """Individual AI Agent definition"""
    agent_id: str
    name: str
    category: AgentCategory
    description: str
    status: AgentStatus = AgentStatus.IDLE
    auto_start: bool = False
    interval_seconds: int = 60
    last_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    config: Dict = field(default_factory=dict)
    thread: Optional[threading.Thread] = None


# ============================================================
# AIMATRIX AGENT REGISTRY - ALL REQUIRED AGENTS
# ============================================================

AIMATRIX_AGENTS = {
    # MINING AGENTS
    "mining_ravencoin": AIAgent(
        agent_id="mining_ravencoin",
        name="Ravencoin Mining Agent",
        category=AgentCategory.MINING,
        description="Manages Ravencoin (RVN) mining with KawPow algorithm",
        auto_start=True,
        interval_seconds=30,
        config={
            "algorithm": "kawpow",
            "wallet": "RUQBaH6yfNseYcgJQqVMCyTs9Ssxne1RZi",
            "pools": [
                "stratum+tcp://pool.us.woolypooly.com:55555",
                "stratum+tcp://us-rvn.2miners.com:6060"
            ],
            "miner": "bzminer"
        }
    ),
    "mining_necv1": AIAgent(
        agent_id="mining_necv1",
        name="NecV1 Mining Agent",
        category=AgentCategory.MINING,
        description="Manages NecV1 algorithm mining for compatible coins",
        auto_start=False,
        interval_seconds=30,
        config={
            "algorithm": "necv1",
            "pools": [],
            "miner": "custom"
        }
    ),
    "mining_kaspa": AIAgent(
        agent_id="mining_kaspa",
        name="Kaspa Mining Agent",
        category=AgentCategory.MINING,
        description="Manages Kaspa (KAS) mining with kHeavyHash",
        auto_start=False,
        config={
            "algorithm": "kaspa",
            "pools": ["stratum+tcp://pool.woolypooly.com:3112"]
        }
    ),
    "mining_monitor": AIAgent(
        agent_id="mining_monitor",
        name="Mining Monitor Agent",
        category=AgentCategory.MINING,
        description="Monitors all mining operations, hashrates, and profitability",
        auto_start=True,
        interval_seconds=60
    ),
    "mining_optimizer": AIAgent(
        agent_id="mining_optimizer",
        name="Mining Optimizer Agent",
        category=AgentCategory.MINING,
        description="AI-powered mining optimization - selects best coins and pools",
        auto_start=True,
        interval_seconds=300
    ),

    # NETWORK/MYSTERIUM AGENTS
    "mysterium_scraper": AIAgent(
        agent_id="mysterium_scraper",
        name="Mysterium Web Scraper Agent",
        category=AgentCategory.SCRAPING,
        description="Web scraping through Mysterium nodes to increase traffic and revenue",
        auto_start=True,
        interval_seconds=10,
        config={
            "targets": [
                "news_sites",
                "data_aggregators",
                "social_media_apis",
                "market_data",
                "weather_services"
            ],
            "max_concurrent": 5,
            "rotate_nodes": True
        }
    ),
    "mysterium_node_manager": AIAgent(
        agent_id="mysterium_node_manager",
        name="Mysterium Node Manager Agent",
        category=AgentCategory.NETWORK,
        description="Manages Mysterium VPN nodes across all servers",
        auto_start=True,
        interval_seconds=120
    ),
    "mysterium_earnings_tracker": AIAgent(
        agent_id="mysterium_earnings_tracker",
        name="Mysterium Earnings Tracker Agent",
        category=AgentCategory.NETWORK,
        description="Tracks and optimizes Mysterium MYST earnings",
        auto_start=True,
        interval_seconds=300
    ),

    # HELIUM AGENTS
    "helium_packet_router": AIAgent(
        agent_id="helium_packet_router",
        name="Helium Packet Router Agent",
        category=AgentCategory.NETWORK,
        description="Routes LoRaWAN packets through Helium network",
        auto_start=True,
        interval_seconds=5
    ),
    "helium_hotspot_monitor": AIAgent(
        agent_id="helium_hotspot_monitor",
        name="Helium Hotspot Monitor Agent",
        category=AgentCategory.NETWORK,
        description="Monitors Helium hotspot status and rewards",
        auto_start=True,
        interval_seconds=60
    ),
    "helium_iot_manager": AIAgent(
        agent_id="helium_iot_manager",
        name="Helium IoT Manager Agent",
        category=AgentCategory.NETWORK,
        description="Manages IoT sensors connected to Helium network",
        auto_start=True,
        interval_seconds=30
    ),

    # TRADING AGENTS
    "trading_mt5_bot": AIAgent(
        agent_id="trading_mt5_bot",
        name="MT5 Trading Bot Agent",
        category=AgentCategory.TRADING,
        description="Automated forex/CFD trading on MetaTrader 5",
        auto_start=False,
        interval_seconds=1,
        config={
            "risk_percent": 1.0,
            "max_positions": 5
        }
    ),
    "trading_crypto_bot": AIAgent(
        agent_id="trading_crypto_bot",
        name="Crypto Trading Bot Agent",
        category=AgentCategory.TRADING,
        description="Automated cryptocurrency trading on Jupiter/Solana",
        auto_start=False,
        config={
            "dex": "jupiter",
            "chain": "solana"
        }
    ),
    "trading_signal_analyzer": AIAgent(
        agent_id="trading_signal_analyzer",
        name="Trading Signal Analyzer Agent",
        category=AgentCategory.TRADING,
        description="AI analysis of trading signals and market conditions",
        auto_start=True,
        interval_seconds=60
    ),

    # SECURITY AGENTS
    "security_ids": AIAgent(
        agent_id="security_ids",
        name="Intrusion Detection Agent",
        category=AgentCategory.SECURITY,
        description="Monitors for security threats and intrusions",
        auto_start=True,
        interval_seconds=10
    ),
    "security_scanner": AIAgent(
        agent_id="security_scanner",
        name="Vulnerability Scanner Agent",
        category=AgentCategory.SECURITY,
        description="Scans systems for vulnerabilities",
        auto_start=True,
        interval_seconds=3600
    ),
    "security_firewall": AIAgent(
        agent_id="security_firewall",
        name="Firewall Manager Agent",
        category=AgentCategory.SECURITY,
        description="Manages firewall rules and blocked IPs",
        auto_start=True,
        interval_seconds=30
    ),

    # DATABASE AGENTS
    "database_sync": AIAgent(
        agent_id="database_sync",
        name="Database Sync Agent",
        category=AgentCategory.DATABASE,
        description="Synchronizes data across SQL Server and SQLite databases",
        auto_start=True,
        interval_seconds=300
    ),
    "database_backup": AIAgent(
        agent_id="database_backup",
        name="Database Backup Agent",
        category=AgentCategory.DATABASE,
        description="Automated database backups",
        auto_start=True,
        interval_seconds=3600
    ),
    "database_cleanup": AIAgent(
        agent_id="database_cleanup",
        name="Database Cleanup Agent",
        category=AgentCategory.DATABASE,
        description="Cleans old data and optimizes database performance",
        auto_start=True,
        interval_seconds=86400
    ),

    # AI AGENTS
    "ai_alcala": AIAgent(
        agent_id="ai_alcala",
        name="ALCALA AI Core Agent",
        category=AgentCategory.AI,
        description="Main ALCALA AI assistant agent",
        auto_start=True,
        interval_seconds=1
    ),
    "ai_llm_orchestrator": AIAgent(
        agent_id="ai_llm_orchestrator",
        name="LLM Orchestrator Agent",
        category=AgentCategory.AI,
        description="Manages and orchestrates all LLM models",
        auto_start=True,
        interval_seconds=5
    ),
    "ai_learning": AIAgent(
        agent_id="ai_learning",
        name="Self-Learning Agent",
        category=AgentCategory.AI,
        description="Continuous learning and model improvement",
        auto_start=True,
        interval_seconds=60
    ),

    # INFRASTRUCTURE AGENTS
    "infra_ssh_manager": AIAgent(
        agent_id="infra_ssh_manager",
        name="SSH Server Manager Agent",
        category=AgentCategory.INFRASTRUCTURE,
        description="Manages SSH connections to all remote servers",
        auto_start=True,
        interval_seconds=60
    ),
    "infra_gpu_monitor": AIAgent(
        agent_id="infra_gpu_monitor",
        name="GPU Monitor Agent",
        category=AgentCategory.INFRASTRUCTURE,
        description="Monitors GPU health, temperature, and utilization",
        auto_start=True,
        interval_seconds=30
    ),
    "infra_network_monitor": AIAgent(
        agent_id="infra_network_monitor",
        name="Network Monitor Agent",
        category=AgentCategory.INFRASTRUCTURE,
        description="Monitors network connectivity and bandwidth",
        auto_start=True,
        interval_seconds=30
    ),
    "infra_topology_updater": AIAgent(
        agent_id="infra_topology_updater",
        name="Topology Updater Agent",
        category=AgentCategory.INFRASTRUCTURE,
        description="Updates network topology map",
        auto_start=True,
        interval_seconds=300
    ),
}


class AIMATRIXAgentManager:
    """
    Central manager for all AIMATRIX agents
    """

    def __init__(self):
        self.agents: Dict[str, AIAgent] = AIMATRIX_AGENTS.copy()
        self.running = False
        self.manager_thread: Optional[threading.Thread] = None

        print(f"[Agent Manager] Initialized with {len(self.agents)} agents")

    def start_agent(self, agent_id: str) -> bool:
        """Start a specific agent"""
        if agent_id not in self.agents:
            print(f"[Agent Manager] Unknown agent: {agent_id}")
            return False

        agent = self.agents[agent_id]
        if agent.status == AgentStatus.RUNNING:
            print(f"[Agent Manager] Agent {agent_id} already running")
            return True

        agent.status = AgentStatus.RUNNING
        print(f"[Agent Manager] Started agent: {agent.name}")
        return True

    def stop_agent(self, agent_id: str) -> bool:
        """Stop a specific agent"""
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]
        agent.status = AgentStatus.STOPPED
        print(f"[Agent Manager] Stopped agent: {agent.name}")
        return True

    def start_all_auto_agents(self):
        """Start all agents marked for auto-start"""
        started = 0
        for agent_id, agent in self.agents.items():
            if agent.auto_start:
                self.start_agent(agent_id)
                started += 1
        print(f"[Agent Manager] Auto-started {started} agents")

    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """Get status of a specific agent"""
        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]
        return {
            "id": agent.agent_id,
            "name": agent.name,
            "category": agent.category.value,
            "status": agent.status.value,
            "run_count": agent.run_count,
            "error_count": agent.error_count,
            "last_run": agent.last_run.isoformat() if agent.last_run else None
        }

    def get_all_agents_status(self) -> List[Dict]:
        """Get status of all agents"""
        return [self.get_agent_status(aid) for aid in self.agents.keys()]

    def get_agents_by_category(self, category: AgentCategory) -> List[AIAgent]:
        """Get all agents in a category"""
        return [a for a in self.agents.values() if a.category == category]

    def get_running_agents(self) -> List[AIAgent]:
        """Get all running agents"""
        return [a for a in self.agents.values() if a.status == AgentStatus.RUNNING]

    def get_summary(self) -> Dict:
        """Get agent manager summary"""
        by_category = {}
        for cat in AgentCategory:
            agents = self.get_agents_by_category(cat)
            by_category[cat.value] = {
                "total": len(agents),
                "running": len([a for a in agents if a.status == AgentStatus.RUNNING])
            }

        return {
            "total_agents": len(self.agents),
            "running_agents": len(self.get_running_agents()),
            "by_category": by_category
        }


# Singleton instance
_agent_manager: Optional[AIMATRIXAgentManager] = None


def get_agent_manager() -> AIMATRIXAgentManager:
    """Get or create singleton agent manager"""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AIMATRIXAgentManager()
    return _agent_manager


def list_all_agents() -> None:
    """Print all agents in a formatted list"""
    print("\n" + "="*70)
    print("AIMATRIX AGENT REGISTRY")
    print("="*70)

    for category in AgentCategory:
        agents = [a for a in AIMATRIX_AGENTS.values() if a.category == category]
        if agents:
            print(f"\n[{category.value.upper()}]")
            print("-"*50)
            for agent in agents:
                auto = "[AUTO]" if agent.auto_start else ""
                print(f"  {agent.agent_id:30} {auto}")
                print(f"    {agent.description}")

    print("\n" + "="*70)
    print(f"Total Agents: {len(AIMATRIX_AGENTS)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    list_all_agents()

    manager = get_agent_manager()
    print("\nAgent Manager Summary:")
    print(json.dumps(manager.get_summary(), indent=2))
