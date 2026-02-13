const express = require('express');
const cors = require('cors');
const path = require('path');
const AlcalaAI = require('./core/AlcalaAI');

const app = express();
const PORT = 4001;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

let alcala = null;
let conversationHistory = [];

/**
 * Alcala API Server with AIMATRIX Integration
 */

// Initialize Alcala
async function initializeAlcala() {
  console.log('Initializing Alcala AI...');
  alcala = new AlcalaAI({
    aimatrixUrl: 'http://192.168.0.100:5001',
    aimatrixMonitoring: true,
    voice: true
  });

  // Listen to Alcala's responses
  alcala.on('response', (response) => {
    conversationHistory.push({
      type: 'alcala',
      ...response
    });
  });

  alcala.on('speech', (speech) => {
    console.log(`[Speech]: "${speech.text}"`);
  });

  await alcala.initialize();
  console.log('Alcala is ready!\n');
}

// API Endpoints

app.get('/status', (req, res) => {
  res.json({
    status: alcala ? alcala.status : 'offline',
    uptime: alcala ? alcala.systemState.uptime : 0,
    activeAgents: alcala ? alcala.agents.size : 0,
    available: alcala !== null,
    aimatrix: alcala ? {
      connected: alcala.aimatrix.connected,
      monitoring: alcala.aimatrix.monitoring,
      lastCheck: alcala.aimatrix.lastCheck
    } : null
  });
});

app.post('/command', async (req, res) => {
  try {
    const { command } = req.body;

    if (!command) {
      return res.status(400).json({ error: 'Command required' });
    }

    if (!alcala || alcala.status !== 'online') {
      return res.status(503).json({ error: 'Alcala is not online' });
    }

    console.log(`\n[User]: ${command}`);

    // Add to history
    conversationHistory.push({
      type: 'user',
      message: command,
      timestamp: new Date().toISOString()
    });

    // Process command
    const response = await alcala.processCommand(command);

    res.json({
      success: true,
      response: response.message,
      timestamp: response.timestamp
    });

  } catch (error) {
    console.error('Error processing command:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/history', (req, res) => {
  res.json({
    history: conversationHistory,
    count: conversationHistory.length
  });
});

app.get('/aimatrix', async (req, res) => {
  if (!alcala) {
    return res.status(503).json({ error: 'Alcala not initialized' });
  }

  try {
    if (!alcala.aimatrix.connected) {
      await alcala.initializeAIMatrix();
    }

    res.json({
      connected: alcala.aimatrix.connected,
      lastStatus: alcala.aimatrix.lastStatus,
      lastCheck: alcala.aimatrix.lastCheck,
      monitoring: alcala.aimatrix.monitoring
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/shutdown', async (req, res) => {
  if (alcala) {
    await alcala.shutdown();
    res.json({ message: 'Alcala shut down successfully' });
    process.exit(0);
  } else {
    res.status(503).json({ error: 'Alcala not running' });
  }
});

// Start server
app.listen(PORT, async () => {
  console.log('\n╔═══════════════════════════════════════════════╗');
  console.log('║        ALCALA AI API SERVER v2.0            ║');
  console.log('║    With AIMATRIX Hardware Integration      ║');
  console.log('╚═══════════════════════════════════════════════╝\n');
  console.log(`API running on http://localhost:${PORT}\n`);
  console.log('Endpoints:');
  console.log(`  POST /command      - Send command to Alcala`);
  console.log(`  GET  /status       - Get Alcala status`);
  console.log(`  GET  /history      - Get conversation history`);
  console.log(`  GET  /aimatrix     - Get AIMATRIX status`);
  console.log(`  POST /shutdown     - Shutdown Alcala\n`);

  await initializeAlcala();
});

// Handle shutdown
process.on('SIGINT', async () => {
  console.log('\n\nShutting down...');
  if (alcala) {
    await alcala.shutdown();
  }
  process.exit(0);
});
