import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  VStack, 
  HStack, 
  Text, 
  Card, 
  Badge, 
  Progress, 
  Button, 
  Select, 
  useToast,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Grid,
  GridItem,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Spinner,
  CardHeader,
  CardBody,
  Heading
} from '@chakra-ui/react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { AlertTriangle, Thermometer, Droplets, Shield, Zap } from 'lucide-react';

const ColdChainMonitoring = () => {
  const [sensorData, setSensorData] = useState([]);
  const [riskAnalysis, setRiskAnalysis] = useState({});
  const [selectedBatch, setSelectedBatch] = useState('BATCH001');
  const [anomalyAlerts, setAnomalyAlerts] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStats, setCurrentStats] = useState({ temperature: 0, humidity: 0, riskScore: 0, status: 'LOADING' });
  const [isConnected, setIsConnected] = useState(false);
  const [alertsDisabled, setAlertsDisabled] = useState(false);
  const [activeToasts, setActiveToasts] = useState([]);
  const [lastAlertTime, setLastAlertTime] = useState({});
  const wsRef = useRef(null);
  const toast = useToast();

  useEffect(() => {
    connectWebSocket();
    fetchInitialData();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (selectedBatch) {
      fetchBatchData(selectedBatch);
      // Don't clear alerts when switching - keep all batch alerts visible
    }
  }, [selectedBatch]);

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket('ws://localhost:8000/ws');
      wsRef.current = ws;
      
      ws.onopen = () => {
        setIsConnected(true);
        console.log('WebSocket connected');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'sensor_data') {
            const newData = data.data;
            setSensorData(prev => {
              const filtered = prev.filter(item => item.batchID !== newData.batchID || 
                new Date(item.timestamp) < new Date(newData.timestamp));
              return [...filtered, newData].slice(-100); // Keep last 100 readings
            });
            
            // Always detect anomalies for all batches
            detectAnomalies(newData);
            
            // Update current stats only for selected batch
            if (newData.batchID === selectedBatch) {
              updateCurrentStats(newData);
              fetchRiskAnalysis(newData.batchID);
            }
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected');
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setTimeout(connectWebSocket, 3000);
    }
  };

  const fetchInitialData = async () => {
    try {
      const response = await fetch(`http://localhost:8000/coldchain/data/${selectedBatch}`);
      if (response.ok) {
        const data = await response.json();
        setSensorData(data.data || []);
        if (data.data && data.data.length > 0) {
          const latest = data.data[data.data.length - 1];
          updateCurrentStats(latest);
          fetchRiskAnalysis(selectedBatch);
          // Force anomaly detection for initial data
          detectAnomalies(latest);
        }
      }
    } catch (error) {
      console.error('Error fetching initial data:', error);
    }
  };

  const fetchBatchData = async (batchId) => {
    try {
      const response = await fetch(`http://localhost:8000/coldchain/data/${batchId}`);
      if (response.ok) {
        const data = await response.json();
        const batchData = data.data || [];
        setSensorData(prev => {
          const otherBatches = prev.filter(item => item.batchID !== batchId);
          return [...otherBatches, ...batchData];
        });
        
        if (batchData.length > 0) {
          const latest = batchData[batchData.length - 1];
          updateCurrentStats(latest);
          fetchRiskAnalysis(batchId);
          // Force anomaly detection for batch data
          detectAnomalies(latest);
        }
      }
    } catch (error) {
      console.error('Error fetching batch data:', error);
    }
  };

  const updateCurrentStats = (data) => {
    // Only update if it's the selected batch to prevent cross-batch contamination
    if (data.batchID === selectedBatch) {
      setCurrentStats({
        temperature: data.temperature,
        humidity: data.humidity,
        riskScore: calculateRiskScore(data.temperature, data.humidity),
        status: getStatusFromTemp(data.temperature)
      });
    }
  };

  const calculateRiskScore = (temp, humidity) => {
    let risk = 0;
    if (temp > 8 || temp < 2) risk += 50;
    if (humidity > 70 || humidity < 30) risk += 30;
    if (Math.abs(temp - 5) > 2) risk += 20;
    return Math.min(risk, 100);
  };

  const getStatusFromTemp = (temp) => {
    if (temp >= 2 && temp <= 8) return 'SAFE';
    if (temp > 8 && temp <= 10) return 'WARNING';
    if (temp > 10) return 'CRITICAL';
    if (temp < 2) return 'CRITICAL';
    return 'CRITICAL';
  };

  const fetchRiskAnalysis = async (batchId) => {
    try {
      const response = await fetch(`http://localhost:8000/coldchain/risk?batch_id=${batchId}`);
      const data = await response.json();
      setRiskAnalysis(prev => ({...prev, [batchId]: data}));
    } catch (error) {
      console.error('Error fetching risk analysis:', error);
    }
  };

  const detectAnomalies = (data) => {
    const temp = data.temperature;
    const humidity = data.humidity;
    const now = Date.now();
    
    // Always add anomalies to the list (don't filter by selected batch)
    if (temp < 2 || temp > 8 || humidity < 30 || humidity > 70) {
      let severity = temp > 10 || temp < 0 || humidity > 80 || humidity < 20 ? 'CRITICAL' : 'WARNING';
      const anomaly = {
        id: Date.now() + Math.random(), // Ensure unique IDs
        batchID: data.batchID,
        type: temp < 2 || temp > 8 ? 'Temperature' : 'Humidity',
        value: temp < 2 || temp > 8 ? `${temp}°C` : `${humidity}%`,
        severity: severity,
        timestamp: data.timestamp,
        message: `${severity}: ${data.batchID} - ${temp < 2 || temp > 8 ? `Temperature ${temp}°C` : `Humidity ${humidity}%`} out of range`
      };
      setAnomalyAlerts(prev => [anomaly, ...prev.slice(0, 19)]); // Keep more alerts
    }
    
    // Only show toasts for selected batch
    if (data.batchID !== selectedBatch) {
      return;
    }
    
    // Rate limiting: only show alerts every 10 seconds globally
    const globalKey = 'global_alert';
    if (lastAlertTime[globalKey] && (now - lastAlertTime[globalKey]) < 10000) {
      return;
    }
    
    // Skip if alerts are disabled
    if (alertsDisabled) return;
    
    // Show toast for selected batch anomalies
    if (temp < 2 || temp > 8 || humidity < 30 || humidity > 70) {
      let severity = temp > 10 || temp < 0 || humidity > 80 || humidity < 20 ? 'CRITICAL' : 'WARNING';
      
      // Update global alert time
      setLastAlertTime(prev => ({...prev, [globalKey]: now}));
      
      // Show single toast
      const toastId = toast({
        title: `${data.batchID} Alert`,
        description: `${temp}°C ${severity === 'CRITICAL' ? '🚨' : '⚠️'}`,
        status: severity === 'CRITICAL' ? "error" : "warning",
        duration: 3000,
        isClosable: true,
        position: "bottom-right",
        variant: "left-accent",
        containerStyle: {
          maxWidth: '280px',
          fontSize: '14px'
        }
      });
    }
  };

  const getBatchData = (batchId) => {
    return sensorData
      .filter(data => data.batchID === batchId)
      .slice(-50)
      .map(item => ({
        ...item,
        time: new Date(item.timestamp).toLocaleTimeString(),
        temp: item.temperature,
        hum: item.humidity
      }));
  };

  const currentBatchData = getBatchData(selectedBatch);
  const currentRisk = riskAnalysis[selectedBatch];

  const getTemperatureChartData = () => {
    return currentBatchData.map(item => ({
      time: item.time,
      temperature: item.temperature,
      humidity: item.humidity,
      optimalMin: 2,
      optimalMax: 8
    }));
  };

  const getRiskColor = (status) => {
    switch (status) {
      case 'SAFE': return 'green';
      case 'WARNING': return 'yellow';
      case 'CRITICAL': return 'red';
      default: return 'gray';
    }
  };

  const getAnomalyColor = (riskLevel) => {
    switch (riskLevel) {
      case 'CRITICAL': return 'red';
      case 'HIGH': return 'orange';
      case 'MEDIUM': return 'yellow';
      case 'LOW': return 'green';
      default: return 'gray';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-6">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          AI-Powered Cold Chain Monitoring
        </h2>
        <p className="text-lg text-gray-600 mt-2">
          Real-time temperature monitoring with intelligent anomaly detection
        </p>
      </div>

      {/* Batch Selection */}
      <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-900">Select Batch for Monitoring</h3>
          <select 
            value={selectedBatch}
            onChange={(e) => setSelectedBatch(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg bg-white"
          >
            <option value="BATCH001">BATCH001 - COVID-19 Vaccine</option>
            <option value="BATCH002">BATCH002 - Cancer Treatment</option>
            <option value="BATCH003">BATCH003 - Diabetes Medication</option>
          </select>
        </div>
      </div>

      {/* Connection Status */}
      <Box mb={4}>
        <HStack spacing={4} align="center" justify="space-between">
          <HStack spacing={4}>
            <Badge colorScheme={isConnected ? 'green' : 'red'} size="lg">
              {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
            </Badge>
            <Text fontSize="sm" color="gray.600">
              {isConnected ? 'Real-time monitoring active' : 'Attempting to reconnect...'}
            </Text>
          </HStack>
          <Button
            size="sm"
            colorScheme={alertsDisabled ? 'green' : 'red'}
            onClick={() => setAlertsDisabled(!alertsDisabled)}
            leftIcon={alertsDisabled ? '🔔' : '🔕'}
          >
            {alertsDisabled ? 'Enable Alerts' : 'Disable Alerts'}
          </Button>
        </HStack>
      </Box>

      {/* Real-time Status Cards */}
      <Grid templateColumns="repeat(auto-fit, minmax(250px, 1fr))" gap={6} mb={8}>
        <Card bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)" color="white" shadow="lg">
          <CardBody>
            <HStack justify="space-between">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" opacity={0.9}>Current Temperature</Text>
                <Text fontSize="3xl" fontWeight="bold">
                  {currentStats.temperature.toFixed(1)}°C
                </Text>
                <Text fontSize="sm" opacity={0.8}>
                  {currentStats.temperature >= 2 && currentStats.temperature <= 8 ? 'Optimal Range' : 'Out of Range'}
                </Text>
              </VStack>
              <Box fontSize="4xl">🌡️</Box>
            </HStack>
          </CardBody>
        </Card>

        <Card bg="linear-gradient(135deg, #f093fb 0%, #f5576c 100%)" color="white" shadow="lg">
          <CardBody>
            <HStack justify="space-between">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" opacity={0.9}>Current Humidity</Text>
                <Text fontSize="3xl" fontWeight="bold">
                  {currentStats.humidity.toFixed(1)}%
                </Text>
                <Text fontSize="sm" opacity={0.8}>
                  Optimal: 30-70%
                </Text>
              </VStack>
              <Box fontSize="4xl">💧</Box>
            </HStack>
          </CardBody>
        </Card>

        <Card bg={`linear-gradient(135deg, ${currentStats.status === 'SAFE' ? '#4facfe 0%, #00f2fe' : currentStats.status === 'WARNING' ? '#fa709a 0%, #fee140' : '#ff6b6b 0%, #ee5a24'} 100%)`} color="white" shadow="lg">
          <CardBody>
            <HStack justify="space-between">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" opacity={0.9}>AI Status</Text>
                <Text fontSize="3xl" fontWeight="bold">
                  {currentStats.status}
                </Text>
                <Text fontSize="sm" opacity={0.8}>
                  AI-Powered Analysis
                </Text>
              </VStack>
              <Box fontSize="4xl">🛡️</Box>
            </HStack>
          </CardBody>
        </Card>

        <Card bg="linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)" color="gray.800" shadow="lg">
          <CardBody>
            <HStack justify="space-between">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" opacity={0.8}>Risk Score</Text>
                <Text fontSize="3xl" fontWeight="bold">
                  {currentStats.riskScore}%
                </Text>
                <Text fontSize="sm" opacity={0.7}>
                  Lower is Better
                </Text>
              </VStack>
              <Box fontSize="4xl">⚡</Box>
            </HStack>
          </CardBody>
        </Card>
      </Grid>

      {/* Live Charts */}
      <Card bg="white" shadow="lg" mb={6}>
        <CardHeader>
          <HStack justify="space-between">
            <Heading size="lg" color="gray.800">Live Temperature & Humidity Monitoring</Heading>
            <Badge colorScheme={currentBatchData.length > 0 ? 'green' : 'gray'}>
              {currentBatchData.length} data points
            </Badge>
          </HStack>
        </CardHeader>
        <CardBody>
          {currentBatchData.length > 0 ? (
            <Box h="400px">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={getTemperatureChartData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis 
                    dataKey="time" 
                    stroke="#4a5568"
                    fontSize={12}
                  />
                  <YAxis 
                    stroke="#4a5568"
                    fontSize={12}
                  />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1a202c',
                      border: 'none',
                      borderRadius: '8px',
                      color: 'white'
                    }}
                  />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey="optimalMax" 
                    fill="#68d391" 
                    fillOpacity={0.1}
                    stroke="none"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="optimalMin" 
                    fill="#68d391" 
                    fillOpacity={0.1}
                    stroke="none"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="temperature" 
                    stroke="#3182ce" 
                    strokeWidth={3}
                    dot={{ fill: '#3182ce', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, stroke: '#3182ce', strokeWidth: 2 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="humidity" 
                    stroke="#e53e3e" 
                    strokeWidth={3}
                    dot={{ fill: '#e53e3e', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, stroke: '#e53e3e', strokeWidth: 2 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="optimalMin" 
                    stroke="#68d391" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="optimalMax" 
                    stroke="#68d391" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          ) : (
            <VStack spacing={4} py={20}>
              <Spinner size="xl" color="blue.500" />
              <Text fontSize="lg" color="gray.600">Loading sensor data...</Text>
              <Text fontSize="sm" color="gray.500">Waiting for real-time updates from {selectedBatch}</Text>
            </VStack>
          )}
        </CardBody>
      </Card>

      {/* AI Analysis & Recommendations */}
      {currentRisk && (
        <Card bg="white" shadow="lg" mb={6}>
          <CardHeader>
            <HStack spacing={4}>
              <Box fontSize="2xl">🤖</Box>
              <Heading size="lg" color="gray.800">AI Analysis & Recommendations</Heading>
              <Badge colorScheme={currentRisk.risk_level === 'LOW' ? 'green' : currentRisk.risk_level === 'MEDIUM' ? 'yellow' : 'red'}>
                {currentRisk.risk_level} RISK
              </Badge>
            </HStack>
          </CardHeader>
          <CardBody>
            <VStack spacing={6} align="stretch">
              {/* Analysis */}
              <Box p={4} bg="blue.50" borderRadius="lg" border="1px solid" borderColor="blue.200">
                <Text fontSize="sm" fontWeight="bold" color="blue.800" mb={2}>📊 AI Analysis:</Text>
                <Text fontSize="sm" color="blue.700">{currentRisk.analysis}</Text>
              </Box>
              
              {/* Recommendations */}
              <Box p={4} bg="green.50" borderRadius="lg" border="1px solid" borderColor="green.200">
                <Text fontSize="sm" fontWeight="bold" color="green.800" mb={2}>💡 Recommendations:</Text>
                <VStack align="start" spacing={1}>
                  {currentRisk.recommendations?.map((rec, index) => (
                    <Text key={index} fontSize="sm" color="green.700">• {rec}</Text>
                  ))}
                </VStack>
              </Box>
              
              {/* AI Insights */}
              {currentRisk.ai_insights && (
                <Box p={4} bg="purple.50" borderRadius="lg" border="1px solid" borderColor="purple.200">
                  <Text fontSize="sm" fontWeight="bold" color="purple.800" mb={2}>🧠 AI Insights:</Text>
                  <VStack align="start" spacing={1}>
                    {currentRisk.ai_insights.map((insight, index) => (
                      <Text key={index} fontSize="sm" color="purple.700">• {insight}</Text>
                    ))}
                  </VStack>
                </Box>
              )}
            </VStack>
          </CardBody>
        </Card>
      )}

      {/* AI Anomaly Detection */}
      <Card bg="white" shadow="lg">
        <CardHeader>
          <HStack spacing={4}>
            <Box fontSize="2xl">🚨</Box>
            <Heading size="lg" color="gray.800">AI Anomaly Detection</Heading>
            <Badge colorScheme={anomalyAlerts.length > 0 ? 'red' : 'green'}>
              {anomalyAlerts.length} alerts
            </Badge>
          </HStack>
        </CardHeader>
        <CardBody>
          {anomalyAlerts.length > 0 ? (
            <VStack spacing={4} align="stretch">
              {anomalyAlerts.slice(0, 3).map((alert) => (
                <Alert 
                  key={alert.id} 
                  status={alert.severity === 'CRITICAL' ? 'error' : 'warning'}
                  borderRadius="lg"
                >
                  <AlertIcon />
                  <Box flex="1">
                    <AlertTitle>
                      {alert.severity} - {alert.batchID} {alert.type} Alert
                    </AlertTitle>
                    <AlertDescription>
                      <VStack align="start" spacing={1} mt={2}>
                        <Text fontSize="sm">{alert.message}</Text>
                        <Text fontSize="sm" fontWeight="bold">Value: {alert.value}</Text>
                        <Text fontSize="xs" color="gray.500">
                          {new Date(alert.timestamp).toLocaleString()}
                        </Text>
                      </VStack>
                    </AlertDescription>
                  </Box>
                </Alert>
              ))}
            </VStack>
          ) : (
            <VStack spacing={4} py={8}>
              <Box fontSize="6xl">🛡️</Box>
              <Text fontSize="lg" color="green.600" fontWeight="semibold">
                No anomalies detected
              </Text>
              <Text color="gray.600">All systems operating normally</Text>
              <HStack spacing={2}>
                <Badge colorScheme="green">AI Monitoring Active</Badge>
                <Badge colorScheme="blue">Real-time Analysis</Badge>
              </HStack>
            </VStack>
          )}
        </CardBody>
      </Card>
    </div>
  );
};

export default ColdChainMonitoring;
