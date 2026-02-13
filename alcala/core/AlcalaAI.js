/**
 * Enhanced AlcalaAI with AIMATRIX Hardware Integration
 * Complete Alcala AI System with British Personality
 */

const EventEmitter = require('events');
const axios = require('axios');

class AlcalaAI extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      llmProvider: config.llmProvider || 'local',
      model: config.model || 'default',
      voice: config.voice !== false,
      aimatrixUrl: config.aimatrixUrl || 'http://192.168.0.100:5001',
      aimatrixMonitoring: config.aimatrixMonitoring || false,
      ...config
    };

    this.status = 'offline';
    this.agents = new Map();
    this.systemState = {
      uptime: 0,
      startTime: Date.now()
    };

    // AIMATRIX integration
    this.aimatrix = {
      connected: false,
      lastCheck: null,
      lastStatus: null,
      monitoring: false
    };
  }

  /**
   * Initialize Alcala and all subsystems
   */
  async initialize() {
    console.log('\n╔═══════════════════════════════════════════════╗');
    console.log('║          ALCALA AI ASSISTANT v2.0           ║');
    console.log('║    With AIMATRIX Hardware Integration      ║');
    console.log('╚═══════════════════════════════════════════════╝\n');

    console.log('🚀 Initializing Alcala AI...');

    // Initialize AIMATRIX connection
    await this.initializeAIMatrix();

    // Register built-in agents
    this.registerBuiltInAgents();

    // Start uptime counter
    this.startUptimeCounter();

    this.status = 'online';
    this.emit('ready');
    
    console.log('✓ Alcala AI is now online');
    console.log('\n💬 Ready for your commands, sir!\n');
    
    return true;
  }
  /**
   * Initialize connection to AIMATRIX hardware
   */
  async initializeAIMatrix() {
    console.log('🔗 Connecting to AIMATRIX...');
    
    try {
      const response = await axios.get(`${this.config.aimatrixUrl}/api/health`, {
        timeout: 3000
      });

      if (response.data.status === 'online') {
        this.aimatrix.connected = true;
        this.aimatrix.lastCheck = Date.now();
        console.log('✓ AIMATRIX bridge connected');
        console.log(`  URL: ${this.config.aimatrixUrl}`);
        
        // Get initial status
        await this.updateAIMatrixStatus();
        
        const gpu = this.aimatrix.lastStatus?.hardware?.gpu;
        if (gpu && gpu.available !== false) {
          console.log(`  GPU: ${gpu.name} @ ${gpu.temperature}°C`);
        }
        
        // Start monitoring if configured
        if (this.config.aimatrixMonitoring) {
          this.startAIMatrixMonitoring();
        }
      }
    } catch (error) {
      console.warn('⚠ AIMATRIX bridge offline - will retry on demand');
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
        console.error('AIMATRIX monitoring error:', error.message);
      }
    }, interval);

    console.log(`✓ AIMATRIX monitoring started (${interval}ms interval)`);
  }

  /**
   * Stop AIMATRIX monitoring
   */
  stopAIMatrixMonitoring() {
    if (this.aimatrix.monitoringInterval) {
      clearInterval(this.aimatrix.monitoringInterval);
      this.aimatrix.monitoringInterval = null;
      this.aimatrix.monitoring = false;
      console.log('✓ AIMATRIX monitoring stopped');
    }
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
   * Register built-in agents
   */
  registerBuiltInAgents() {
    console.log('🤖 Registering agents...');
    // Built-in agents would be registered here
  }

  /**
   * Process user command
   */
  async processCommand(command) {
    const timestamp = new Date().toISOString();
    
    // Check for AIMATRIX-related commands
    if (this.isAIMatrixCommand(command)) {
      return await this.handleAIMatrixCommand(command, timestamp);
    }

    // Regular command processing
    const response = {
      message: this.generateResponse(command),
      timestamp
    };

    this.emit('response', response);

    if (this.config.voice) {
      this.emit('speech', { text: response.message });
    }

    return response;
  }

  /**
   * Check if command is AIMATRIX-related
   */
  isAIMatrixCommand(command) {
    const cmd = command.toLowerCase();
    const keywords = [
      'aimatrix', 'hardware', 'gpu', 'mining', 
      'ravencoin', 'rvn', 'temperature', 'status',
      'system', 'workstation', 'arch', 'disk', 'memory'
    ];
    
    return keywords.some(keyword => cmd.includes(keyword));
  }
  /**
   * Handle AIMATRIX-specific commands
   */
  async handleAIMatrixCommand(command, timestamp) {
    const cmd = command.toLowerCase();

    try {
      // Reconnect if needed
      if (!this.aimatrix.connected) {
        await this.initializeAIMatrix();
      }

      let message = '';

      // Hardware status
      if (cmd.includes('status') || cmd.includes('hardware') || cmd.includes('system')) {
        await this.updateAIMatrixStatus();
        message = this.formatAIMatrixReport();
      }
      // Mining status
      else if (cmd.includes('mining') || cmd.includes('ravencoin') || cmd.includes('rvn')) {
        const mining = await this.getAIMatrixMining();
        message = this.formatMiningReport(mining);
      }
      // GPU specific
      else if (cmd.includes('gpu') || cmd.includes('temperature')) {
        await this.updateAIMatrixStatus();
        const gpu = this.aimatrix.lastStatus.hardware.gpu;
        
        if (gpu.available !== false) {
          message = `The ${gpu.name} is currently at ${gpu.temperature} degrees celsius, `;
          message += `with ${gpu.utilization.gpu} percent utilization. `;
          message += `VRAM usage is ${gpu.memory.used} megabytes out of ${gpu.memory.total}, sir.`;
        } else {
          message = "I'm unable to detect the GPU at the moment, sir.";
        }
      }
      else {
        message = "I'm not quite sure what you'd like to know about AIMATRIX, sir. Perhaps ask about the hardware status or mining operations?";
      }

      const response = { message, timestamp };
      this.emit('response', response);

      if (this.config.voice) {
        this.emit('speech', { text: message });
      }

      return response;

    } catch (error) {
      const errorMessage = `I'm having difficulty connecting to AIMATRIX at the moment, sir. ${error.message}`;
      const response = { message: errorMessage, timestamp };
      
      this.emit('response', response);
      return response;
    }
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
   * Generate response (basic implementation)
   */
  generateResponse(command) {
    // British personality responses
    const responses = [
      `Certainly, sir. I've noted your request: "${command}"`,
      `Very good, sir. Regarding "${command}", I shall attend to that directly.`,
      `Right away, sir. Your command has been received.`,
      `Understood, sir. Processing your request.`
    ];

    return responses[Math.floor(Math.random() * responses.length)];
  }

  /**
   * Start uptime counter
   */
  startUptimeCounter() {
    setInterval(() => {
      this.systemState.uptime = Math.floor((Date.now() - this.systemState.startTime) / 1000);
    }, 1000);
  }

  /**
   * Shutdown Alcala
   */
  async shutdown() {
    console.log('\nShutting down Alcala AI...');
    
    this.stopAIMatrixMonitoring();
    this.status = 'offline';
    this.emit('shutdown');
    
    console.log('✓ Alcala AI shut down complete');
  }
}

module.exports = AlcalaAI;
