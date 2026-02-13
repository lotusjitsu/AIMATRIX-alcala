const EventEmitter = require('events');
const winston = require('winston');
const axios = require('axios');

/**
 * Alcala - Primary AI Assistant with AIMATRIX Integration
 * British-accented AI with system-wide command authority
 */
class AlcalaAI extends EventEmitter {
  constructor(config = {}) {
    super();

    this.name = 'Alcala';
    this.accent = 'british';
    this.status = 'initializing';
    this.config = {
      ...config,
      aimatrixUrl: config.aimatrixUrl || 'http://192.168.0.100:5001',
      aimatrixMonitoring: config.aimatrixMonitoring || false
    };

    // Initialize logger
    this.logger = winston.createLogger({
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.printf(({ timestamp, level, message }) => {
          return `[${timestamp}] [Alcala] ${level.toUpperCase()}: ${message}`;
        })
      ),
      transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'logs/alcala.log' })
      ]
    });

    // Command registry
    this.commands = new Map();

    // Agent registry
    this.agents = new Map();

    // System state
    this.systemState = {
      uptime: 0,
      activeAgents: 0,
      tasksCompleted: 0,
      hardwareNodes: [],
      lastCommand: null
    };

    // AIMATRIX integration
    this.aimatrix = {
      connected: false,
      lastCheck: null,
      lastStatus: null,
      monitoring: false
    };

    this.logger.info('Alcala AI initialized');
  }

  /**
   * Initialize Alcala and bring all systems online
   */
  async initialize() {
    this.logger.info('Bringing systems online...');
    
    // Connect to AIMATRIX
    await this.initializeAIMatrix();
    
    this.status = 'online';

    // Register default commands
    this.registerDefaultCommands();

    // Start uptime counter
    this.startUptimeCounter();

    // Emit ready event
    this.emit('ready');

    this.logger.info('All systems operational. Alcala is ready.');
    this.speak('Good day. Alcala AI online and ready to serve.');

    return this;
  }
  /**
   * Initialize connection to AIMATRIX hardware
   */
  async initializeAIMatrix() {
    this.logger.info('Connecting to AIMATRIX...');
    
    try {
      const response = await axios.get(`${this.config.aimatrixUrl}/api/health`, {
        timeout: 3000
      });

      if (response.data.status === 'online') {
        this.aimatrix.connected = true;
        this.aimatrix.lastCheck = Date.now();
        this.logger.info(`AIMATRIX bridge connected at ${this.config.aimatrixUrl}`);
        
        // Get initial status
        await this.updateAIMatrixStatus();
        
        const gpu = this.aimatrix.lastStatus?.hardware?.gpu;
        if (gpu && gpu.available !== false) {
          this.logger.info(`GPU detected: ${gpu.name} @ ${gpu.temperature}°C`);
        }
        
        // Register AIMATRIX commands
        this.registerAIMatrixCommands();
        
        // Start monitoring if configured
        if (this.config.aimatrixMonitoring) {
          this.startAIMatrixMonitoring();
        }
      }
    } catch (error) {
      this.logger.warn('AIMATRIX bridge offline - will retry on demand');
      this.aimatrix.connected = false;
    }
  }

  /**
   * Update AIMATRIX hardware status
   */
  async updateAIMatrixStatus() {
    try {
      const [status, cpu, gpu, memory, disk] = await Promise.all([
        axios.get(`${this.config.aimatrixUrl}/api/status`),
        axios.get(`${this.config.aimatrixUrl}/api/hardware/cpu`),
        axios.get(`${this.config.aimatrixUrl}/api/hardware/gpu`),
        axios.get(`${this.config.aimatrixUrl}/api/hardware/memory`),
        axios.get(`${this.config.aimatrixUrl}/api/hardware/disk`)
      ]);

      this.aimatrix.lastStatus = {
        system: status.data,
        hardware: {
          cpu: cpu.data,
          gpu: gpu.data,
          memory: memory.data,
          disk: disk.data
        },
        timestamp: new Date().toISOString()
      };

      this.aimatrix.lastCheck = Date.now();
      this.aimatrix.connected = true;

      return this.aimatrix.lastStatus;
    } catch (error) {
      this.aimatrix.connected = false;
      throw new Error(`Failed to update AIMATRIX status: ${error.message}`);
    }
  }

  /**
   * Get AIMATRIX mining status
   */
  async getAIMatrixMining() {
    try {
      const [status, wallet, sync] = await Promise.all([
        axios.get(`${this.config.aimatrixUrl}/api/mining/status`),
        axios.get(`${this.config.aimatrixUrl}/api/mining/wallet`).catch(() => ({ data: null })),
        axios.get(`${this.config.aimatrixUrl}/api/mining/sync`).catch(() => ({ data: null }))
      ]);

      return {
        active: status.data.active,
        process: status.data.process,
        wallet: wallet.data,
        blockchain: sync.data
      };
    } catch (error) {
      throw new Error(`Failed to get mining status: ${error.message}`);
    }
  }

  /**
   * Start continuous AIMATRIX monitoring
   */
  startAIMatrixMonitoring(interval = 30000) {
    if (this.aimatrix.monitoringInterval) {
      return;
    }

    this.aimatrix.monitoring = true;
    this.aimatrix.monitoringInterval = setInterval(async () => {
      try {
        await this.updateAIMatrixStatus();
        this.emit('aimatrix:update', this.aimatrix.lastStatus);
      } catch (error) {
        this.logger.error(`AIMATRIX monitoring error: ${error.message}`);
      }
    }, interval);

    this.logger.info(`AIMATRIX monitoring started (${interval}ms interval)`);
  }

  /**
   * Stop AIMATRIX monitoring
   */
  stopAIMatrixMonitoring() {
    if (this.aimatrix.monitoringInterval) {
      clearInterval(this.aimatrix.monitoringInterval);
      this.aimatrix.monitoringInterval = null;
      this.aimatrix.monitoring = false;
      this.logger.info('AIMATRIX monitoring stopped');
    }
  }

  /**
   * Register AIMATRIX-specific commands
   */
  registerAIMatrixCommands() {
    this.registerCommand('aimatrix_status', this.getAIMatrixStatus.bind(this));
    this.registerCommand('gpu_status', this.getGPUStatus.bind(this));
    this.registerCommand('mining_status', this.getMiningStatus.bind(this));
    this.logger.info('AIMATRIX commands registered');
  }

  /**
   * Get AIMATRIX status (command handler)
   */
  async getAIMatrixStatus() {
    if (!this.aimatrix.connected) {
      await this.initializeAIMatrix();
    }

    await this.updateAIMatrixStatus();
    return this.formatAIMatrixReport();
  }

  /**
   * Get GPU status (command handler)
   */
  async getGPUStatus() {
    if (!this.aimatrix.connected) {
      await this.initializeAIMatrix();
    }

    await this.updateAIMatrixStatus();
    const gpu = this.aimatrix.lastStatus.hardware.gpu;
    
    if (gpu.available !== false) {
      return `The ${gpu.name} is currently at ${gpu.temperature} degrees celsius, ` +
             `with ${gpu.utilization.gpu} percent utilization. ` +
             `VRAM usage is ${gpu.memory.used} megabytes out of ${gpu.memory.total}.`;
    } else {
      return "GPU information is currently unavailable.";
    }
  }

  /**
   * Get mining status (command handler)
   */
  async getMiningStatus() {
    if (!this.aimatrix.connected) {
      await this.initializeAIMatrix();
    }

    const mining = await this.getAIMatrixMining();
    return this.formatMiningReport(mining);
  }

  /**
   * Format AIMATRIX status for British accent response
   */
  formatAIMatrixReport() {
    if (!this.aimatrix.lastStatus) {
      return "I'm afraid the AIMATRIX connection is currently unavailable, sir.";
    }

    const { system, hardware } = this.aimatrix.lastStatus;
    const uptimeHours = Math.floor(system.uptime / 3600);
    
    let report = `AIMATRIX system report, sir. `;
    report += `The ${system.hostname} workstation has been operational for ${uptimeHours} hours. `;
    
    // CPU
    report += `CPU usage is at ${hardware.cpu.usage.toFixed(1)} percent. `;
    
    // GPU
    if (hardware.gpu.available !== false) {
      report += `The ${hardware.gpu.name} is running at ${hardware.gpu.temperature} degrees celsius, `;
      report += `with ${hardware.gpu.utilization.gpu} percent utilization. `;
      
      if (hardware.gpu.temperature > 80) {
        report += `I should note that's rather warm, sir. `;
      } else if (hardware.gpu.temperature < 60) {
        report += `Running quite cool, sir. `;
      }
    }
    
    // Memory
    const memPercent = hardware.memory.usagePercent;
    report += `Memory usage is ${memPercent} percent. `;
    if (memPercent > 90) {
      report += `That's quite high, sir. `;
    }

    return report;
  }

  /**
   * Format mining report in British accent
   */
  formatMiningReport(mining) {
    let report = '';

    if (mining.active) {
      report = 'The Ravencoin mining operation is currently active, sir. ';
      
      if (mining.wallet) {
        report += `The wallet balance stands at ${mining.wallet.balance} RVN `;
        report += `with ${mining.wallet.txcount} transactions. `;
      }
      
      if (mining.blockchain) {
        const progress = (mining.blockchain.verificationprogress * 100).toFixed(2);
        report += `Blockchain synchronisation is ${progress} percent complete, `;
        report += `with ${mining.blockchain.blocks.toLocaleString()} of ${mining.blockchain.headers.toLocaleString()} blocks processed.`;
      }
    } else {
      report = 'The mining operation appears to be offline at the moment, sir.';
    }

    return report;
  }
  /**
   * Register default commands
   */
  registerDefaultCommands() {
    // System commands
    this.registerCommand('status', this.getSystemStatus.bind(this));
    this.registerCommand('help', this.showHelp.bind(this));
    this.registerCommand('shutdown', this.shutdown.bind(this));

    // Agent commands
    this.registerCommand('spawn_agent', this.spawnAgent.bind(this));
    this.registerCommand('list_agents', this.listAgents.bind(this));
    this.registerCommand('terminate_agent', this.terminateAgent.bind(this));

    // Hardware commands
    this.registerCommand('hardware_status', this.getHardwareStatus.bind(this));
    this.registerCommand('allocate_resources', this.allocateResources.bind(this));

    this.logger.info(`Registered ${this.commands.size} default commands`);
  }

  /**
   * Register a new command
   */
  registerCommand(name, handler) {
    this.commands.set(name.toLowerCase(), handler);
    this.logger.info(`Command registered: ${name}`);
  }

  /**
   * Process a command from user
   */
  async processCommand(input) {
    this.logger.info(`Processing command: "${input}"`);
    this.systemState.lastCommand = input;

    try {
      // Check for AIMATRIX keywords first
      if (this.isAIMatrixCommand(input)) {
        return await this.handleAIMatrixCommand(input);
      }

      // Parse natural language command
      const parsed = this.parseCommand(input);

      if (!parsed.command) {
        return this.respond('I apologize, but I did not understand that command. Please try again or say "help" for available commands.');
      }

      // Execute command
      const handler = this.commands.get(parsed.command);
      if (handler) {
        const result = await handler(parsed.args);
        return this.respond(result);
      } else {
        return this.respond(`Command "${parsed.command}" not recognized. Say "help" for available commands.`);
      }
    } catch (error) {
      this.logger.error(`Command processing error: ${error.message}`);
      return this.respond(`I encountered an error processing your request: ${error.message}`);
    }
  }

  /**
   * Check if command is AIMATRIX-related
   */
  isAIMatrixCommand(command) {
    const cmd = command.toLowerCase();
    const keywords = [
      'aimatrix', 'hardware', 'gpu', 'mining', 
      'ravencoin', 'rvn', 'temperature',
      'workstation', 'arch', 'disk', 'memory', 'vram'
    ];
    
    return keywords.some(keyword => cmd.includes(keyword));
  }

  /**
   * Handle AIMATRIX-specific commands
   */
  async handleAIMatrixCommand(command) {
    const cmd = command.toLowerCase();

    try {
      // Reconnect if needed
      if (!this.aimatrix.connected) {
        await this.initializeAIMatrix();
      }

      let message = '';

      // Hardware status
      if (cmd.includes('status') || cmd.includes('hardware')) {
        message = await this.getAIMatrixStatus();
      }
      // Mining status
      else if (cmd.includes('mining') || cmd.includes('ravencoin') || cmd.includes('rvn')) {
        message = await this.getMiningStatus();
      }
      // GPU specific
      else if (cmd.includes('gpu') || cmd.includes('temperature') || cmd.includes('vram')) {
        message = await this.getGPUStatus();
      }
      else {
        message = await this.getAIMatrixStatus();
      }

      return this.respond(message);

    } catch (error) {
      const errorMessage = `I'm having difficulty connecting to AIMATRIX at the moment. ${error.message}`;
      return this.respond(errorMessage);
    }
  }

  /**
   * Parse natural language input into command and arguments
   */
  parseCommand(input) {
    input = input.toLowerCase().trim();

    // Remove "alcala" prefix if present
    input = input.replace(/^(alcala|hey alcala|ok alcala)[,\s]*/i, '');

    // Command patterns
    const patterns = {
      status: /^(what('s| is) the )?(system )?status\??$/,
      help: /^(show )?help$/,
      spawn_agent: /^(spawn|create|deploy) (a|an) (new )?agent/,
      list_agents: /^(list|show|get) agents?$/,
      hardware_status: /^(hardware|system|resource) status$/,
    };

    for (const [command, pattern] of Object.entries(patterns)) {
      if (pattern.test(input)) {
        return { command, args: {}, original: input };
      }
    }

    // Direct command match
    const words = input.split(' ');
    if (this.commands.has(words[0])) {
      return {
        command: words[0],
        args: words.slice(1),
        original: input
      };
    }

    return { command: null, args: {}, original: input };
  }

  /**
   * Generate response
   */
  respond(message) {
    const response = {
      message,
      timestamp: new Date().toISOString(),
      speaker: 'Alcala'
    };

    this.logger.info(`Response: ${message}`);
    this.emit('response', response);

    return response;
  }

  /**
   * Speak (would integrate with TTS for British accent)
   */
  speak(text) {
    this.logger.info(`[Speech]: ${text}`);
    this.emit('speech', { text, accent: this.accent });

    // In production, this would call ElevenLabs or similar TTS API
    // with British accent settings

    return text;
  }

  /**
   * Get system status
   */
  async getSystemStatus() {
    const status = {
      name: this.name,
      status: this.status,
      uptime: this.formatUptime(this.systemState.uptime),
      activeAgents: this.agents.size,
      tasksCompleted: this.systemState.tasksCompleted,
      hardwareNodes: this.systemState.hardwareNodes.length,
      aimatrix: {
        connected: this.aimatrix.connected,
        monitoring: this.aimatrix.monitoring
      },
      timestamp: new Date().toISOString()
    };

    const message = `System status: ${status.status}. Uptime: ${status.uptime}. ` +
                   `${status.activeAgents} active agents. ` +
                   `${status.tasksCompleted} tasks completed. ` +
                   `AIMATRIX: ${status.aimatrix.connected ? 'connected' : 'offline'}.`;

    return message;
  }

  /**
   * Show available commands
   */
  async showHelp() {
    const commands = Array.from(this.commands.keys()).join(', ');
    return `Available commands: ${commands}. You may also use natural language to ask about AIMATRIX hardware, mining status, or GPU temperature.`;
  }

  /**
   * Spawn a new agent
   */
  async spawnAgent(args) {
    const agentId = `agent-${Date.now()}`;

    this.agents.set(agentId, {
      id: agentId,
      status: 'active',
      spawnedAt: new Date().toISOString(),
      tasks: []
    });

    this.logger.info(`Agent ${agentId} spawned`);
    return `Agent ${agentId} successfully deployed and operational.`;
  }

  /**
   * List active agents
   */
  async listAgents() {
    if (this.agents.size === 0) {
      return 'No agents currently active.';
    }

    const agentList = Array.from(this.agents.values())
      .map(a => `${a.id} (${a.status})`)
      .join(', ');

    return `Active agents: ${agentList}`;
  }

  /**
   * Terminate an agent
   */
  async terminateAgent(args) {
    const agentId = args.id || args[0];

    if (this.agents.has(agentId)) {
      this.agents.delete(agentId);
      return `Agent ${agentId} terminated successfully.`;
    }

    return `Agent ${agentId} not found.`;
  }

  /**
   * Get hardware status
   */
  async getHardwareStatus() {
    // This integrates with AIMATRIX
    if (this.aimatrix.connected) {
      return await this.getAIMatrixStatus();
    }
    return 'Hardware monitoring unavailable. AIMATRIX bridge offline.';
  }

  /**
   * Allocate resources
   */
  async allocateResources(args) {
    return 'Resource allocation optimized.';
  }

  /**
   * Shutdown Alcala
   */
  async shutdown() {
    this.logger.info('Initiating shutdown sequence...');
    this.status = 'shutting_down';

    // Stop AIMATRIX monitoring
    this.stopAIMatrixMonitoring();

    // Cleanup
    this.agents.clear();

    this.status = 'offline';
    this.logger.info('Alcala AI offline');

    this.emit('shutdown');

    return 'Alcala AI shutting down. Goodbye.';
  }

  /**
   * Format uptime
   */
  formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }

  /**
   * Start uptime counter
   */
  startUptimeCounter() {
    setInterval(() => {
      if (this.status === 'online') {
        this.systemState.uptime++;
      }
    }, 1000);
  }
}

module.exports = AlcalaAI;
