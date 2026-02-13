/**
 * AIMATRIX Bridge Test Client
 * Simple test to verify Alcala can connect to AIMATRIX
 */

const axios = require('axios');

const BRIDGE_URL = 'http://192.168.0.100:5001';

async function testConnection() {
  console.log('\n╔═══════════════════════════════════════════════╗');
  console.log('║   ALCALA ↔ AIMATRIX CONNECTION TEST         ║');
  console.log('╚═══════════════════════════════════════════════╝\n');

  try {
    // Test 1: Health Check
    console.log('📊 Test 1: Health Check');
    const health = await axios.get(`${BRIDGE_URL}/api/health`);
    console.log(`✅ Bridge Status: ${health.data.status}`);
    console.log(`   Service: ${health.data.service} v${health.data.version}\n`);

    // Test 2: System Status
    console.log('📊 Test 2: System Status');
    const status = await axios.get(`${BRIDGE_URL}/api/status`);
    console.log(`✅ Hostname: ${status.data.hostname}`);
    console.log(`   Uptime: ${Math.floor(status.data.uptime / 3600)} hours\n`);

    // Test 3: GPU Stats
    console.log('📊 Test 3: GPU Statistics');
    const gpu = await axios.get(`${BRIDGE_URL}/api/hardware/gpu`);
    if (gpu.data.available !== false) {
      console.log(`✅ GPU: ${gpu.data.name}`);
      console.log(`   Temperature: ${gpu.data.temperature}°C`);
      console.log(`   Utilization: ${gpu.data.utilization.gpu}%`);
      console.log(`   VRAM: ${gpu.data.memory.used}MB / ${gpu.data.memory.total}MB\n`);
    }

    // Test 4: Mining Status
    console.log('📊 Test 4: Mining Status');
    const mining = await axios.get(`${BRIDGE_URL}/api/mining/status`);
    console.log(`✅ Mining Active: ${mining.data.active}`);
    if (mining.data.active) {
      console.log(`   Process ID: ${mining.data.process.pid}`);
      console.log(`   Memory: ${mining.data.process.memory}MB\n`);
    }

    // Test 5: Get all hardware stats
    console.log('📊 Test 5: Complete Hardware Report');
    const [cpu, memory, disk] = await Promise.all([
      axios.get(`${BRIDGE_URL}/api/hardware/cpu`),
      axios.get(`${BRIDGE_URL}/api/hardware/memory`),
      axios.get(`${BRIDGE_URL}/api/hardware/disk`)
    ]);
    
    console.log(`✅ CPU: ${cpu.data.model}`);
    console.log(`   Cores: ${cpu.data.cores}`);
    console.log(`   Usage: ${cpu.data.usage.toFixed(1)}%`);
    console.log(`✅ Memory: ${memory.data.used.toFixed(1)}GB / ${memory.data.total.toFixed(1)}GB (${memory.data.usagePercent}%)`);
    console.log(`✅ Disk C: ${disk.data.Used}GB / ${disk.data.Total}GB used\n`);

    console.log('╔═══════════════════════════════════════════════╗');
    console.log('║        ALL TESTS PASSED! ✅                   ║');
    console.log('║  Alcala can now access AIMATRIX hardware!    ║');
    console.log('╚═══════════════════════════════════════════════╝\n');

  } catch (error) {
    console.error('\n❌ Connection failed:', error.message);
    console.error('\nTroubleshooting:');
    console.error('1. Is the AIMATRIX bridge running? (node aimatrix-bridge.js)');
    console.error('2. Is port 5001 accessible?');
    console.error('3. Check Windows Firewall settings');
    console.error('4. Verify IP address is correct (192.168.0.100)\n');
  }
}

// Run tests
testConnection();
