const AlcalaAI = require('./core/AlcalaAI');
const readline = require('readline');

/**
 * Alcala AI - Interactive Command Line Interface
 * With AIMATRIX Hardware Integration
 */

// Create readline interface for CLI interaction
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: 'You: '
});

async function main() {
  // Initialize Alcala with AIMATRIX integration
  const alcala = new AlcalaAI({
    llmProvider: process.env.ALCALA_LLM_PROVIDER || 'local',
    model: process.env.ALCALA_MODEL || 'default',
    voice: true,
    aimatrixUrl: 'http://192.168.0.100:5001',
    aimatrixMonitoring: false // Set to true for continuous monitoring
  });

  // Set up event listeners
  alcala.on('ready', () => {
    rl.prompt();
  });

  alcala.on('response', (response) => {
    console.log(`\nAlcala: ${response.message}\n`);
    rl.prompt();
  });

  alcala.on('speech', (speech) => {
    // In production, this would trigger TTS
    console.log(`[🔊 British accent]: "${speech.text}"`);
  });

  alcala.on('shutdown', () => {
    console.log('\nAlcala AI has been shut down.');
    rl.close();
    process.exit(0);
  });

  // Initialize
  await alcala.initialize();

  // Handle user input
  rl.on('line', async (input) => {
    input = input.trim();

    if (!input) {
      rl.prompt();
      return;
    }

    // Check for exit commands
    if (['exit', 'quit', 'bye'].includes(input.toLowerCase())) {
      await alcala.shutdown();
      return;
    }

    // Process command
    console.log(''); // Blank line before response
    await alcala.processCommand(input);
  });

  rl.on('close', () => {
    console.log('\nGoodbye, sir!');
    process.exit(0);
  });

  // Handle process termination
  process.on('SIGINT', async () => {
    console.log('\n\nReceived interrupt signal...');
    await alcala.shutdown();
  });
}

// Run
main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
