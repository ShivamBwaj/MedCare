import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart3, 
  Package, 
  Thermometer, 
  Users, 
  Settings, 
  LogOut,
  Plus,
  Search,
  Filter,
  MoreVertical,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  Database,
  Activity,
  Shield,
  Bell,
  TestTube,
  Zap,
  Camera,
  UserCheck,
  Brain
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import AIDrugVerification from './AIDrugVerification';
import PatientAdherence from './PatientAdherence';

const Dashboard = ({ onLogout }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [user] = useState({ name: 'Dr. Sarah Johnson', role: 'Administrator' });
  
  // TrialChain+ColdCare state
  const [batches, setBatches] = useState([]);
  const [sensorData, setSensorData] = useState([]);
  

  const [riskAnalysis, setRiskAnalysis] = useState({});
  const [selectedBatch, setSelectedBatch] = useState('BATCH001');
  const [formData, setFormData] = useState({
    drugName: '',
    expiry: '',
    sender: '',
    receiver: ''
  });
  const [loading, setLoading] = useState(false);
  const [graphLoaded, setGraphLoaded] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);

  const batchIds = ['BATCH001', 'BATCH002', 'BATCH003'];

  // Mock data
  const storageItems = [
    { id: 1, name: 'Amoxicillin 500mg', category: 'Antibiotics', stock: 45, maxStock: 50, price: 45.99, status: 'low' },
    { id: 2, name: 'Surgical Gloves (Box)', category: 'Consumables', stock: 156, maxStock: 100, price: 23.5, status: 'good' },
    { id: 3, name: 'Insulin Pens', category: 'Diabetes Care', stock: 12, maxStock: 25, price: 89.99, status: 'critical' },
    { id: 4, name: 'Blood Pressure Monitors', category: 'Equipment', stock: 8, maxStock: 5, price: 129.99, status: 'good' },
    { id: 5, name: 'Paracetamol 500mg', category: 'Pain Management', stock: 25, maxStock: 30, price: 12.99, status: 'low' },
  ];

  const alerts = [
    { id: 1, type: 'low_stock', message: 'Amoxicillin 500mg is running low (45 units left)', time: '5 min ago', severity: 'warning' },
    { id: 2, type: 'expiry', message: 'Insulin Pens expire in 30 days', time: '1 hour ago', severity: 'warning' },
    { id: 3, type: 'critical', message: 'Insulin Pens below critical threshold (12 units)', time: '2 hours ago', severity: 'critical' },
    { id: 4, type: 'verification', message: 'New transaction recorded: BATCH001 approved', time: '15 min ago', severity: 'success' },
    { id: 5, type: 'expiry', message: 'Insulin Pens expire in 30 days', time: '1 hour ago', severity: 'warning'},
    { id: 6, type: 'critical', message: 'Insulin Pens below critical threshold (12 units)', time: '2 hours ago', severity: 'critical' },
  ];

  const notifications = [
    { id: 1, type: 'clinical_trial', message: 'New drug batch BATCH003 awaiting approval', time: '2 min ago', severity: 'info', icon: TestTube },
    { id: 2, type: 'cold_chain', message: 'Temperature alert: BATCH003 exceeded safe range', time: '5 min ago', severity: 'warning', icon: Thermometer },
    { id: 3, type: 'storage', message: 'Amoxicillin 500mg is running low (45 units left)', time: '10 min ago', severity: 'warning', icon: Package },
    { id: 4, type: 'verification', message: 'New transaction recorded: BATCH001 approved', time: '15 min ago', severity: 'success', icon: Shield },
    { id: 5, type: 'expiry', message: 'Insulin Pens expire in 30 days', time: '1 hour ago', severity: 'warning', icon: AlertTriangle },
    { id: 6, type: 'critical', message: 'Insulin Pens below critical threshold (12 units)', time: '2 hours ago', severity: 'critical', icon: AlertTriangle },
  ];

  const verificationActivity = [
    { id: 1, type: 'stock_update', hash: '0x1a2b3c4d', item: 'Amoxicillin 500mg', action: 'Quantity updated: 50 → 45', time: '2 min ago', block: '12847592', contract: '0xE2DFC07f329041a05f5257f27CE01e4329FC64Ef' },
    { id: 2, type: 'purchase_order', hash: '0x5e6f7g8h', item: 'Surgical Gloves', action: 'Order created: 500 units', time: '15 min ago', block: '12847588', contract: '0xE2DFC07f329041a05f5257f27CE01e4329FC64Ef' },
    { id: 3, type: 'item_added', hash: '0x9i0j1k2l', item: 'Blood Pressure Monitor', action: 'New item added to storage', time: '1 hour ago', block: '12847585', contract: '0xE2DFC07f329041a05f5257f27CE01e4329FC64Ef' },
  ];

  const analyticsData = [
    { month: 'Jan', items: 1185, value: 44250 },
    { month: 'Feb', items: 1234, value: 46780 },
    { month: 'Mar', items: 1298, value: 51890 },
    { month: 'Apr', items: 1256, value: 48120 },
    { month: 'May', items: 1312, value: 52340 },
    { month: 'Jun', items: 1345, value: 53890 },
    { month: 'Jul', items: 1378, value: 55230 },
    { month: 'Aug', items: 1412, value: 56890 },
    { month: 'Sep', items: 1389, value: 54320 },
  ];

  const pieData = [
    { name: 'Antibiotics', value: 32, color: '#3B82F6' },
    { name: 'Consumables', value: 28, color: '#10B981' },
    { name: 'Equipment', value: 22, color: '#F59E0B' },
    { name: 'Pain Management', value: 12, color: '#EF4444' },
    { name: 'Diabetes Care', value: 6, color: '#8B5CF6' },
  ];

  // Clinical Trials Overview Data
  const clinicalTrialsOverview = [
    { id: 1, batchID: 'BATCH001', drugName: 'COVID-19 Vaccine (Moderna)', status: 'approved', progress: 100, approvedBy: 'Regulator_001' },
    { id: 2, batchID: 'BATCH002', drugName: 'Cancer Treatment Drug (Keytruda)', status: 'approved', progress: 100, approvedBy: 'Regulator_002' },
    { id: 3, batchID: 'BATCH003', drugName: 'Diabetes Medication (Ozempic)', status: 'pending', progress: 65, approvedBy: null },
    { id: 4, batchID: 'BATCH004', drugName: 'Antibiotic (Amoxicillin)', status: 'approved', progress: 100, approvedBy: 'Regulator_001' },
    { id: 5, batchID: 'BATCH005', drugName: 'Pain Management (Oxycodone)', status: 'pending', progress: 45, approvedBy: null },
  ];

  // Cold Chain Overview Data
  const coldChainOverview = [
    { batchID: 'BATCH001', avgTemp: 4.2, status: 'SAFE', riskScore: 0.15, lastUpdate: '2 min ago' },
    { batchID: 'BATCH002', avgTemp: 3.8, status: 'SAFE', riskScore: 0.08, lastUpdate: '5 min ago' },
    { batchID: 'BATCH003', avgTemp: 9.5, status: 'CRITICAL', riskScore: 0.95, lastUpdate: '1 min ago' },
  ];

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'storage', label: 'Inventory', icon: Package },
    { id: 'clinical-trials', label: 'Clinical Trials', icon: TestTube },
    { id: 'coldchain', label: 'Cold Chain', icon: Thermometer },
    { id: 'ai-drug-verification', label: 'AI Drug Verification', icon: Camera },
    { id: 'patient-adherence', label: 'Patient Adherence', icon: UserCheck },
    { id: 'ai-analytics', label: 'AI Analytics', icon: Brain },
    { id: 'analytics', label: 'Analytics', icon: TrendingUp },
    { id: 'alerts', label: 'Alerts', icon: Bell },
    { id: 'verification', label: 'Security', icon: Shield },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  // TrialChain+ColdCare API functions
  const fetchBatches = async () => {
    try {
      const response = await fetch('http://localhost:8000/trials');
      const data = await response.json();
      setBatches(data.batches);
    } catch (error) {
      console.error('Error fetching batches:', error);
    }
  };

  const handleSubmitBatch = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/trials', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (response.ok) {
        setFormData({
          drugName: '',
          expiry: '',
          sender: '',
          receiver: ''
        });
        fetchBatches();
      }
    } catch (error) {
      console.error('Error creating batch:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleApproveBatch = async (batchId) => {
    try {
      const response = await fetch(`http://localhost:8000/trials/${batchId}/approve`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        fetchBatches();
      }
    } catch (error) {
      console.error('Error approving batch:', error);
    }
  };

  const fetchRiskAnalysis = async (batchId) => {
    try {
      // Get current sensor data for the specific batch
      const currentData = sensorData[sensorData.length - 1];
      
      if (currentData) {
        const temp = parseFloat(currentData.temperature.toFixed(1));
        const humidity = parseFloat(currentData.humidity.toFixed(1));

        // Logical risk calculation based on temperature thresholds
        let riskScore = 0;
        let status = 'SAFE';
        let recommendations = [];

        // Temperature-based risk assessment (more logical)
        if (temp >= 8.0 || temp <= 2.0) {
          riskScore = 0.95; // High risk
          status = 'CRITICAL';
          recommendations = [
            `CRITICAL - Temperature ${temp}°C is outside safe range!`,
            'Temperature exceeds 8°C or below 2°C',
            'Immediate action required',
            'Check cooling system immediately',
            'Contact maintenance team'
          ];
        } else if (temp >= 6.0 || temp <= 3.0) {
          riskScore = 0.65; // Medium risk
          status = 'WARNING';
          recommendations = [
            `WARNING - Temperature ${temp}°C approaching limits`,
            'Temperature between 3-6°C is optimal',
            'Monitor closely',
            'Consider adjusting settings',
            'Prepare contingency plan'
          ];
        } else {
          riskScore = 0.15; // Low risk
          status = 'SAFE';
          recommendations = [
            `SAFE - Temperature ${temp}°C within optimal range`,
            'Temperature between 3-6°C is optimal',
            'Continue standard monitoring',
            'No immediate action required'
          ];
        }

        // Adjust for humidity if needed
        if (humidity < 30 || humidity > 70) {
          riskScore = Math.min(riskScore + 0.1, 1.0);
          recommendations.push('Humidity levels need attention');
        }

        setRiskAnalysis({
          status,
          risk_score: riskScore,
          recommendations
        });
        return;
      }



      // Fallback to traditional risk calculation
      const response = await fetch(`http://localhost:8000/coldchain/risk?batch_id=${batchId}`);
      if (response.ok) {
        const analysis = await response.json();
        setRiskAnalysis(analysis);
      } else {
        // Use the existing currentData variable
        if (currentData) {
          const temp = currentData.temperature;
          const humidity = currentData.humidity;

          // Calculate risk based on temperature
          if (temp < 2 || temp > 8) {
            riskScore = 0.9; // High risk
            status = 'CRITICAL';
            recommendations = [
              'Temperature outside safe range!',
              'Immediate action required',
              'Check cooling system',
              'Contact maintenance team'
            ];
          } else if (temp > 6 || temp < 3) {
            riskScore = 0.6; // Medium risk
            status = 'WARNING';
            recommendations = [
              'Temperature approaching limits',
              'Monitor closely',
              'Consider adjusting settings',
              'Prepare contingency plan'
            ];
          } else {
            riskScore = 0.1; // Low risk
            status = 'SAFE';
            recommendations = [
              'Temperature within optimal range',
              'Continue standard monitoring',
              'No immediate action required'
            ];
          }

          // Adjust for humidity if needed
          if (humidity < 30 || humidity > 70) {
            riskScore = Math.min(riskScore + 0.2, 1.0);
            recommendations.push('Humidity levels need attention');
          }
        } else {
          // Fallback to batch-specific profiles
          const riskProfiles = {
            'BATCH001': {
              status: 'SAFE',
              risk_score: 0.12,
              recommendations: [
                'Temperature stable at 4.2°C',
                'Humidity levels optimal',
                'Continue standard monitoring'
              ]
            },
            'BATCH002': {
              status: 'SAFE',
              risk_score: 0.08,
              recommendations: [
                'Excellent temperature control at 3.8°C',
                'Humidity slightly elevated but acceptable',
                'Maintain current settings'
              ]
            },
            'BATCH003': {
              status: 'WARNING',
              risk_score: 0.72,
              recommendations: [
                'Temperature approaching upper limit (5.8°C)',
                'Consider adjusting cooling system',
                'Increase monitoring frequency',
                'Prepare contingency plan if trend continues'
              ]
            }
          };
          
          const profile = riskProfiles[batchId] || riskProfiles['BATCH001'];
          setRiskAnalysis(profile);
          return;
        }

        setRiskAnalysis({
          status,
          risk_score: riskScore,
          recommendations
        });
      }
    } catch (error) {
      console.error('Error fetching risk analysis:', error);
      // Fallback to default profile
      setRiskAnalysis({
        status: 'SAFE',
        risk_score: 0.12,
        recommendations: [
          'Temperature stable at 4.2°C',
          'Humidity levels optimal',
          'Continue standard monitoring'
        ]
      });
    }
  };



  // Live temperature data generation
  useEffect(() => {
    if (activeTab === 'coldchain') {
      // Clear existing data when switching batches
      setSensorData([]);
      
      // Generate initial dataset with stable base values
      const generateInitialData = () => {
        const data = [];
        const now = Date.now();
        
        // Base values for each batch (more realistic)
        const baseValues = {
          'BATCH001': { temp: 4.2, humidity: 45 },
          'BATCH002': { temp: 3.8, humidity: 47 },
          'BATCH003': { temp: 9.5, humidity: 55 }  // Higher temperature to trigger ML model
        };
        
        const base = baseValues[selectedBatch] || { temp: 4.2, humidity: 45 };
        
        for (let i = 19; i >= 0; i--) {
          const timestamp = new Date(now - i * 3000); // 3 second intervals
          // Small variations around base values (more realistic)
          const tempVariation = (Math.random() - 0.5) * 0.4; // ±0.2°C
          const humidityVariation = (Math.random() - 0.5) * 2; // ±1%
          
          data.push({
            time: timestamp.toLocaleTimeString(),
            temperature: parseFloat((base.temp + tempVariation).toFixed(1)),
            humidity: parseFloat((base.humidity + humidityVariation).toFixed(1)),
            timestamp: timestamp.toISOString()
          });
        }
        return data;
      };

      setSensorData(generateInitialData());

      // Real-time updates with minimal variations
      const interval = setInterval(() => {
        const baseValues = {
          'BATCH001': { temp: 4.2, humidity: 45 },
          'BATCH002': { temp: 3.8, humidity: 47 },
          'BATCH003': { temp: 9.5, humidity: 55 }  // Higher temperature to trigger ML model
        };
        
        const base = baseValues[selectedBatch] || { temp: 4.2, humidity: 45 };
        
        // Very small realistic variations
        const tempVariation = (Math.random() - 0.5) * 0.3; // ±0.15°C
        const humidityVariation = (Math.random() - 0.5) * 1.5; // ±0.75%
        
        const newPoint = {
          time: new Date().toLocaleTimeString(),
          temperature: parseFloat((base.temp + tempVariation).toFixed(1)),
          humidity: parseFloat((base.humidity + humidityVariation).toFixed(1)),
          timestamp: new Date().toISOString()
        };

        setSensorData(prev => {
          const updated = [...prev, newPoint];
          return updated.slice(-25); // Keep last 25 points for smooth display
        });
      }, 3000); // Update every 3 seconds

      return () => clearInterval(interval);
    }
  }, [activeTab, selectedBatch]);

  useEffect(() => {
    if (activeTab === 'clinical-trials') {
      fetchBatches();
    }
  }, [activeTab]);

  useEffect(() => {
    if (activeTab === 'coldchain') {
      // Simulate graph loading animation
      setGraphLoaded(false);
      setTimeout(() => setGraphLoaded(true), 1000);
    }
  }, [activeTab]);

  // Separate useEffect for risk analysis to ensure it updates when batch changes
  useEffect(() => {
    if (activeTab === 'coldchain') {
      fetchRiskAnalysis(selectedBatch);
    }
  }, [selectedBatch, activeTab]);

  // Update risk analysis when sensor data changes
  useEffect(() => {
    if (activeTab === 'coldchain' && sensorData.length > 0) {
      // Update risk analysis immediately when sensor data changes
      fetchRiskAnalysis(selectedBatch);
      
      // Also update every 10 seconds for continuous monitoring
      const interval = setInterval(() => {
        fetchRiskAnalysis(selectedBatch);
      }, 10000);
      
      return () => clearInterval(interval);
    }
  }, [activeTab, sensorData, selectedBatch]);

  // Close notifications when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showNotifications && !event.target.closest('.notification-dropdown')) {
        setShowNotifications(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showNotifications]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'good': return 'text-green-500 bg-green-100';
      case 'low': return 'text-yellow-500 bg-yellow-100';
      case 'critical': return 'text-red-500 bg-red-100';
      default: return 'text-gray-500 bg-gray-100';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'good': return 'good';
      case 'low': return 'low';
      case 'critical': return 'critical';
      default: return 'unknown';
    }
  };

  const getStatusColorColdChain = (status) => {
    return status === 'SAFE' ? 'text-green-600' : 'text-red-600';
  };

  const getStatusBgColorColdChain = (status) => {
    return status === 'SAFE' ? 'bg-green-100' : 'bg-red-100';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Large blurred balls moving slowly */}
        <motion.div
          className="absolute w-96 h-96 rounded-full blur-xl bg-gradient-to-r from-blue-400/40 to-purple-400/40"
          style={{
            top: '10%',
            left: '10%',
          }}
          animate={{
            x: [0, 100, 0],
            y: [0, -50, 0],
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 30,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        
        <motion.div
          className="absolute w-80 h-80 rounded-full blur-xl bg-gradient-to-r from-purple-400/40 to-pink-400/40"
          style={{
            top: '60%',
            right: '15%',
          }}
          animate={{
            x: [0, -80, 0],
            y: [0, 40, 0],
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 35,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        
        <motion.div
          className="absolute w-72 h-72 rounded-full blur-xl bg-gradient-to-r from-green-400/35 to-blue-400/35"
          style={{
            top: '30%',
            right: '30%',
          }}
          animate={{
            x: [0, 60, 0],
            y: [0, -30, 0],
            rotate: [0, 180, 360],
          }}
          transition={{
            duration: 40,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* More moving balls for richer background */}
        <motion.div
          className="absolute w-64 h-64 rounded-full blur-lg bg-gradient-to-r from-indigo-400/35 to-cyan-400/35"
          style={{
            top: '70%',
            left: '5%',
          }}
          animate={{
            x: [0, 120, 0],
            y: [0, -40, 0],
            scale: [1, 1.15, 1],
          }}
          transition={{
            duration: 45,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        <motion.div
          className="absolute w-56 h-56 rounded-full blur-lg bg-gradient-to-r from-pink-400/35 to-orange-400/35"
          style={{
            top: '20%',
            left: '60%',
          }}
          animate={{
            x: [0, -100, 0],
            y: [0, 60, 0],
            rotate: [0, 180, 360],
          }}
          transition={{
            duration: 50,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        <motion.div
          className="absolute w-48 h-48 rounded-full blur-lg bg-gradient-to-r from-emerald-400/35 to-teal-400/35"
          style={{
            top: '80%',
            right: '5%',
          }}
          animate={{
            x: [0, 80, 0],
            y: [0, -60, 0],
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 55,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Subtle floating particles */}
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 rounded-full bg-blue-500/20"
            style={{
              left: `${5 + Math.random() * 90}%`,
              top: `${5 + Math.random() * 90}%`,
            }}
            animate={{
              y: [0, -40, 0],
              opacity: [0.3, 0.8, 0.3],
            }}
            transition={{
              duration: 25 + Math.random() * 15,
              repeat: Infinity,
              delay: Math.random() * 15,
              ease: "easeInOut",
            }}
          />
        ))}

        {/* Subtle grid pattern */}
        <div className="absolute inset-0 opacity-3 bg-[radial-gradient(circle_at_1px_1px,rgba(0,0,0,0.05)_1px,transparent_0)] bg-[length:30px_30px]" />
      </div>

      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-sm border-b border-gray-200/50 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="text-2xl font-bold text-gray-900">
                <span className="text-blue-600">Med</span>
                <span className="text-green-600">Care</span>
              </div>
              <div className="h-6 w-px bg-gray-300"></div>
              <div className="text-sm text-gray-600">
                Welcome back, <span className="font-semibold">{user.name}</span>
              </div>
            </div>
                          <div className="flex items-center space-x-4">
                <div className="relative notification-dropdown">
                  <button 
                    onClick={() => setShowNotifications(!showNotifications)}
                    className="p-2 text-gray-400 hover:text-gray-600 transition-colors relative"
                  >
                    <Bell size={20} />
                    {notifications.length > 0 && (
                      <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                        {notifications.length}
                      </span>
                    )}
                  </button>
                
                {/* Notification Dropdown */}
                {showNotifications && (
                  <motion.div
                    initial={{ opacity: 0, y: -10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-lg border border-gray-200 z-50 max-h-96 overflow-y-auto"
                  >
                    <div className="p-4 border-b border-gray-200">
                      <h3 className="text-lg font-semibold text-gray-900">Notifications</h3>
                    </div>
                    <div className="divide-y divide-gray-100">
                      {notifications.map((notification) => (
                        <div key={notification.id} className="p-4 hover:bg-gray-50 transition-colors">
                          <div className="flex items-start space-x-3">
                            <div className={`p-2 rounded-full ${
                              notification.severity === 'critical' ? 'bg-red-100' :
                              notification.severity === 'warning' ? 'bg-yellow-100' :
                              notification.severity === 'success' ? 'bg-green-100' :
                              'bg-blue-100'
                            }`}>
                              <notification.icon size={16} className={
                                notification.severity === 'critical' ? 'text-red-600' :
                                notification.severity === 'warning' ? 'text-yellow-600' :
                                notification.severity === 'success' ? 'text-green-600' :
                                'text-blue-600'
                              } />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm text-gray-900">{notification.message}</p>
                              <p className="text-xs text-gray-500 mt-1">{notification.time}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </div>
              <button 
                onClick={onLogout}
                className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors"
              >
                <LogOut size={16} />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {/* Tab Navigation */}
        <div className="flex space-x-1 bg-white/80 backdrop-blur-md rounded-lg p-1 shadow-lg mb-8 overflow-x-auto border border-white/20">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 whitespace-nowrap ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              <tab.icon size={16} />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === 'overview' && (
              <div className="space-y-8">
                {/* Stats Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Total Items</p>
                        <p className="text-2xl font-bold text-gray-900">1,389</p>
                        <p className="text-sm text-green-600 flex items-center mt-1">
                          <TrendingUp size={14} className="mr-1" />
                          +8.7% from last month
                        </p>
                      </div>
                      <div className="p-3 bg-blue-100 rounded-lg">
                        <Database className="w-6 h-6 text-blue-600" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Low Stock Alerts</p>
                        <p className="text-2xl font-bold text-gray-900">31</p>
                        <p className="text-sm text-red-600 flex items-center mt-1">
                          <AlertTriangle size={14} className="mr-1" />
                          +7 today
                        </p>
                      </div>
                      <div className="p-3 bg-red-100 rounded-lg">
                        <AlertTriangle className="w-6 h-6 text-red-600" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Monthly Usage</p>
                        <p className="text-2xl font-bold text-gray-900">$54,320</p>
                        <p className="text-sm text-green-600 flex items-center mt-1">
                          <TrendingUp size={14} className="mr-1" />
                          +13.6% from last month
                        </p>
                      </div>
                      <div className="p-3 bg-green-100 rounded-lg">
                        <DollarSign className="w-6 h-6 text-green-600" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Inventory Value</p>
                        <p className="text-2xl font-bold text-gray-900">$312,450</p>
                        <p className="text-sm text-green-600 flex items-center mt-1">
                          <TrendingUp size={14} className="mr-1" />
                          +9.9% from last month
                        </p>
                      </div>
                      <div className="p-3 bg-purple-100 rounded-lg">
                        <Activity className="w-6 h-6 text-purple-600" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Inventory Trends</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={analyticsData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="items" stroke="#3B82F6" strokeWidth={2} />
                        <Line type="monotone" dataKey="value" stroke="#10B981" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Category Distribution</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {pieData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Recent Activity and Clinical Trials - Side by Side */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Recent Activity */}
                  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span className="text-xs text-green-600 font-medium">System Active</span>
                      </div>
                    </div>
                    <div className="space-y-4">
                      {verificationActivity.slice(0, 5).map((activity) => (
                        <div key={activity.id} className="flex items-center space-x-4 p-3 bg-gray-50 rounded-lg">
                          <div className="p-2 bg-blue-100 rounded-lg">
                            <Shield className="w-4 h-4 text-blue-600" />
                          </div>
                          <div className="flex-1">
                            <p className="text-sm font-medium text-gray-900">{activity.item}</p>
                            <p className="text-xs text-gray-600">{activity.action}</p>
                            <p className="text-xs text-purple-600 font-mono">Hash: {activity.contract.slice(0, 8)}...</p>
                          </div>
                          <div className="text-right">
                            <p className="text-xs text-gray-500">{activity.time}</p>
                            <p className="text-xs text-blue-600 font-mono">{activity.hash}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Clinical Trials Overview */}
                  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">Clinical Trials Status</h3>
                      <span className="text-sm text-gray-500">{clinicalTrialsOverview.length} Active Trials</span>
                    </div>
                    <div className="space-y-4">
                      {clinicalTrialsOverview.map((trial) => (
                        <div key={trial.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                          <div className="flex items-center space-x-4">
                            <div className={`p-2 rounded-full ${
                              trial.status === 'approved' ? 'bg-green-100' : 'bg-yellow-100'
                            }`}>
                              <TestTube className={`w-4 h-4 ${
                                trial.status === 'approved' ? 'text-green-600' : 'text-yellow-600'
                              }`} />
                            </div>
                            <div>
                              <p className="text-sm font-medium text-gray-900">{trial.drugName}</p>
                              <p className="text-xs text-gray-500">{trial.batchID}</p>
                            </div>
                          </div>
                          <div className="flex items-center space-x-4">
                            <div className="text-right">
                              <div className="flex items-center space-x-2">
                                <div className="w-20 bg-gray-200 rounded-full h-2">
                                  <div 
                                    className={`h-2 rounded-full ${
                                      trial.status === 'approved' ? 'bg-green-500' : 'bg-yellow-500'
                                    }`}
                                    style={{ width: `${trial.progress}%` }}
                                  ></div>
                                </div>
                                <span className="text-xs text-gray-600">{trial.progress}%</span>
                              </div>
                              <p className="text-xs text-gray-500 mt-1">
                                {trial.status === 'approved' ? `Approved by ${trial.approvedBy}` : 'Pending approval'}
                              </p>
                            </div>
                            <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                              trial.status === 'approved' 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-yellow-100 text-yellow-800'
                            }`}>
                              {trial.status}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Cold Chain Monitoring Overview */}
                <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">Cold Chain Monitoring</h3>
                    <span className="text-sm text-gray-500">Real-time Temperature Tracking</span>
                  </div>
                  <div className="space-y-4">
                    {coldChainOverview.map((batch) => (
                      <div key={batch.batchID} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-4">
                          <div className={`p-2 rounded-full ${
                            batch.status === 'SAFE' ? 'bg-green-100' : 'bg-red-100'
                          }`}>
                            <Thermometer className={`w-4 h-4 ${
                              batch.status === 'SAFE' ? 'text-green-600' : 'text-red-600'
                            }`} />
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-900">{batch.batchID}</p>
                            <p className="text-xs text-gray-500">Avg: {batch.avgTemp}°C</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-4">
                          <div className="text-right">
                            <div className="flex items-center space-x-2">
                              <div className="w-20 bg-gray-200 rounded-full h-2">
                                <div 
                                  className={`h-2 rounded-full ${
                                    batch.riskScore < 0.3 ? 'bg-green-500' : 
                                    batch.riskScore < 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                                  }`}
                                  style={{ width: `${batch.riskScore * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-xs text-gray-600">{(batch.riskScore * 100).toFixed(0)}%</span>
                            </div>
                            <p className="text-xs text-gray-500 mt-1">Updated {batch.lastUpdate}</p>
                          </div>
                          <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                            batch.status === 'SAFE' 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {batch.status}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'storage' && (
              <div className="space-y-6">
                <div className="flex justify-between items-center">
                  <h2 className="text-2xl font-bold text-gray-900">Inventory Management</h2>
                  <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2">
                    <Plus size={16} />
                    <span>Add Item</span>
                  </button>
                </div>

                <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                  <div className="p-6 border-b border-gray-200">
                    <div className="flex items-center space-x-4">
                      <div className="flex-1 relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                        <input
                          type="text"
                          placeholder="Search storage..."
                          className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                        <Filter size={20} />
                      </button>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Item</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Stock</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Price</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {storageItems.map((item) => (
                          <tr key={item.id} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div>
                                <div className="text-sm font-medium text-gray-900">{item.name}</div>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-sm text-gray-900">{item.category}</div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-sm text-gray-900">{item.stock}/{item.maxStock}</div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-sm text-gray-900">${item.price}</div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(item.status)}`}>
                                {getStatusText(item.status)}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                              <div className="flex space-x-2">
                                <button className="text-blue-600 hover:text-blue-900">List</button>
                                <button className="text-gray-600 hover:text-gray-900">View</button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'alerts' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900">Active Alerts</h2>
                <div className="space-y-4">
                  {alerts.map((alert) => (
                    <div key={alert.id} className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start space-x-4">
                          <div className={`p-3 rounded-lg ${
                            alert.severity === 'critical' ? 'bg-red-100' : 'bg-yellow-100'
                          }`}>
                            <AlertTriangle className={`w-6 h-6 ${
                              alert.severity === 'critical' ? 'text-red-600' : 'text-yellow-600'
                            }`} />
                          </div>
                          <div className="flex-1">
                            <p className="text-sm font-medium text-gray-900">{alert.message}</p>
                            <p className="text-xs text-gray-500 mt-1">{alert.time}</p>
                          </div>
                        </div>
                        <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                          Resolve
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'verification' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-gray-900">Security Activity</h2>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-sm text-green-600 font-medium">System Active</span>
                  </div>
                </div>

                {/* Contract Information */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Smart Contract Details</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Contract Address</label>
                      <div className="flex items-center space-x-2">
                        <code className="px-3 py-2 bg-gray-100 rounded-md text-sm font-mono text-gray-900">
                          0xE2DFC07f329041a05f5257f27CE01e4329FC64Ef
                        </code>
                        <button className="p-1 text-gray-400 hover:text-gray-600">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                        </button>
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Network</label>
                      <div className="px-3 py-2 bg-blue-50 border border-blue-200 rounded-md">
                        <span className="text-sm text-blue-700 font-medium">Ethereum Mainnet</span>
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Contract Status</label>
                      <div className="px-3 py-2 bg-green-50 border border-green-200 rounded-md">
                        <span className="text-sm text-green-700 font-medium">Deployed & Verified</span>
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Total Transactions</label>
                      <div className="px-3 py-2 bg-gray-50 border border-gray-200 rounded-md">
                        <span className="text-sm text-gray-700 font-medium">1,389</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recent Transactions */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Transactions</h3>
                  <div className="space-y-4">
                    {verificationActivity.map((activity) => (
                      <div key={activity.id} className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                        <div className="flex items-start justify-between">
                          <div className="flex items-start space-x-4">
                            <div className="p-3 bg-blue-100 rounded-lg">
                              <Shield className="w-6 h-6 text-blue-600" />
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <span className="text-sm font-medium text-gray-900">{activity.type.replace('_', ' ').toUpperCase()}</span>
                                <span className="text-xs text-gray-500">•</span>
                                <span className="text-xs text-gray-500">{activity.time}</span>
                              </div>
                              <p className="text-sm text-gray-900 mb-1">{activity.item}</p>
                              <p className="text-sm text-gray-600">{activity.action}</p>
                              <div className="flex items-center space-x-4 mt-2">
                                <span className="text-xs text-blue-600 font-mono">{activity.hash}</span>
                                <span className="text-xs text-gray-500">Block #{activity.block}</span>
                                <span className="text-xs text-purple-600 font-mono">Contract: {activity.contract.slice(0, 8)}...</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'clinical-trials' && (
              <div className="space-y-8">
                {/* Form Section */}
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Log New Drug Batch</h2>
                  <form onSubmit={handleSubmitBatch} className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Drug Name
                        </label>
                        <input
                          type="text"
                          required
                          value={formData.drugName}
                          onChange={(e) => setFormData({...formData, drugName: e.target.value})}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder="Enter drug name"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Expiry Date
                        </label>
                        <input
                          type="date"
                          required
                          value={formData.expiry}
                          onChange={(e) => setFormData({...formData, expiry: e.target.value})}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Sender
                        </label>
                        <input
                          type="text"
                          required
                          value={formData.sender}
                          onChange={(e) => setFormData({...formData, sender: e.target.value})}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder="Enter sender name"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Receiver
                        </label>
                        <input
                          type="text"
                          required
                          value={formData.receiver}
                          onChange={(e) => setFormData({...formData, receiver: e.target.value})}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder="Enter receiver name"
                        />
                      </div>
                    </div>
                    <button
                      type="submit"
                      disabled={loading}
                      className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                    >
                      {loading ? 'Creating Batch...' : 'Create Batch'}
                    </button>
                  </form>
                </div>

                {/* Blockchain Ledger Section */}
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Blockchain Ledger - Approved Shipments</h2>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Batch ID
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Drug Name
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Expiry
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Sender
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Receiver
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Status
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Actions
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {batches.map((batch) => (
                          <tr key={batch.batchID} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              {batch.batchID}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {batch.drugName}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {new Date(batch.expiry).toLocaleDateString()}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {batch.sender}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {batch.receiver}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                batch.status === 'approved' 
                                  ? 'bg-green-100 text-green-800' 
                                  : 'bg-yellow-100 text-yellow-800'
                              }`}>
                                {batch.status}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {batch.status === 'pending' && (
                                <button
                                  onClick={() => handleApproveBatch(batch.batchID)}
                                  className="bg-green-600 text-white px-3 py-1 rounded-md text-xs hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500"
                                >
                                  Approve (Regulator)
                                </button>
                              )}
                              {batch.status === 'approved' && batch.approved_by && (
                                <span className="text-xs text-gray-500">
                                  Approved by {batch.approved_by}
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {batches.length === 0 && (
                      <div className="text-center py-8 text-gray-500">
                        No batches found. Create a new batch to see it here.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'coldchain' && (
              <div className="space-y-8">
                {/* Batch Selection */}
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-4">Cold-Chain Monitoring</h2>
                  <div className="flex space-x-4">
                    {batchIds.map((batchId) => (
                      <button
                        key={batchId}
                        onClick={() => setSelectedBatch(batchId)}
                        className={`px-4 py-2 rounded-md font-medium ${
                          selectedBatch === batchId
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                      >
                        {batchId}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Live Temperature Graph */}
                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-gray-900">Live Temperature Data</h3>
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                        <span className="text-sm text-gray-600">Temperature (°C)</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                        <span className="text-sm text-gray-600">Humidity (%)</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                        <span className="text-sm text-gray-600">Live</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="h-80">
                    {!graphLoaded ? (
                      <div className="h-full flex items-center justify-center">
                        <div className="text-center">
                          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                          <p className="text-gray-500">Initializing live monitoring...</p>
                        </div>
                      </div>
                    ) : (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                        className="h-full"
                      >
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={sensorData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                            <XAxis 
                              dataKey="time" 
                              tick={{ fontSize: 12, fill: '#666' }}
                              tickLine={false}
                              axisLine={false}
                            />
                            <YAxis 
                              yAxisId="left"
                              domain={[2, 8]} 
                              tick={{ fontSize: 12, fill: '#666' }}
                              tickLine={false}
                              axisLine={false}
                              label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#666' } }}
                            />
                            <YAxis 
                              yAxisId="right"
                              orientation="right"
                              domain={[30, 60]} 
                              tick={{ fontSize: 12, fill: '#666' }}
                              tickLine={false}
                              axisLine={false}
                              label={{ value: 'Humidity (%)', angle: 90, position: 'insideRight', style: { textAnchor: 'middle', fill: '#666' } }}
                            />
                            <Tooltip 
                              contentStyle={{ 
                                backgroundColor: 'white', 
                                border: '1px solid #e5e7eb',
                                borderRadius: '8px',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                              }}
                              labelStyle={{ fontWeight: 'bold', color: '#374151' }}
                            />
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="temperature"
                              stroke="#3B82F6"
                              strokeWidth={3}
                              dot={{ fill: '#3B82F6', strokeWidth: 2, r: 3 }}
                              activeDot={{ r: 5, stroke: '#3B82F6', strokeWidth: 2 }}
                              animationDuration={1000}
                              animationBegin={0}
                            />
                            <Line
                              yAxisId="right"
                              type="monotone"
                              dataKey="humidity"
                              stroke="#10B981"
                              strokeWidth={3}
                              dot={{ fill: '#10B981', strokeWidth: 2, r: 3 }}
                              activeDot={{ r: 5, stroke: '#10B981', strokeWidth: 2 }}
                              animationDuration={1000}
                              animationBegin={200}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </motion.div>
                    )}
                  </div>
                  
                  {/* Current Values Display */}
                  {graphLoaded && sensorData.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="mt-4 grid grid-cols-2 gap-4"
                    >
                      <div className="bg-blue-50 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-blue-700">Current Temperature</span>
                          <Thermometer className="w-5 h-5 text-blue-600" />
                        </div>
                        <div className="mt-2">
                          <span className="text-2xl font-bold text-blue-900">
                            {sensorData[sensorData.length - 1]?.temperature?.toFixed(1)}°C
                          </span>
                        </div>
                      </div>
                      <div className="bg-green-50 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-green-700">Current Humidity</span>
                          <div className="w-5 h-5 text-green-600">💧</div>
                        </div>
                        <div className="mt-2">
                          <span className="text-2xl font-bold text-green-700">
                            {sensorData[sensorData.length - 1]?.humidity?.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </div>

                {/* AI Risk Analysis */}
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">AI Risk Analysis - {selectedBatch}</h3>
                  
                  {!graphLoaded ? (
                    <div className="text-center py-8">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                      <p className="text-gray-500">Analyzing temperature patterns...</p>
                    </div>
                  ) : riskAnalysis.status ? (
                    <motion.div 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5 }}
                      className="space-y-4"
                    >
                      {/* Status Indicator */}
                      <div className="flex items-center space-x-4">
                        <span className="text-lg font-medium text-gray-700">Status:</span>
                        <motion.div 
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                          className={`px-4 py-2 rounded-full ${getStatusBgColorColdChain(riskAnalysis.status)}`}
                        >
                          <span className={`font-bold text-lg ${getStatusColorColdChain(riskAnalysis.status)}`}>
                            {riskAnalysis.status}
                          </span>
                        </motion.div>
                      </div>

                      {/* Risk Score */}
                      <div className="flex items-center space-x-4">
                        <span className="text-lg font-medium text-gray-700">Risk Score:</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-32 bg-gray-200 rounded-full h-2">
                            <motion.div 
                              initial={{ width: 0 }}
                              animate={{ width: `${riskAnalysis.risk_score * 100}%` }}
                              transition={{ delay: 0.4, duration: 1 }}
                              className={`h-2 rounded-full ${
                                riskAnalysis.risk_score < 0.3 ? 'bg-green-500' : 
                                riskAnalysis.risk_score < 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                              }`}
                            ></motion.div>
                          </div>
                          <span className="text-sm font-medium text-gray-600">
                            {(riskAnalysis.risk_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      {/* Recommendations */}
                      {riskAnalysis.recommendations && riskAnalysis.recommendations.length > 0 && (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.6 }}
                        >
                          <span className="text-lg font-medium text-gray-700 block mb-2">Recommendations:</span>
                          <ul className="list-disc list-inside space-y-1">
                            {riskAnalysis.recommendations.map((rec, index) => (
                              <motion.li 
                                key={index} 
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.8 + index * 0.1 }}
                                className="text-gray-600"
                              >
                                {rec}
                              </motion.li>
                            ))}
                          </ul>
                        </motion.div>
                      )}
                    </motion.div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      Loading risk analysis...
                    </div>
                  )}
                </div>

                {/* Real-time Data Feed */}
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">Real-time Sensor Data Feed</h3>
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {!graphLoaded ? (
                      <div className="text-center py-8">
                        <div className="animate-pulse space-y-3">
                          {[...Array(5)].map((_, i) => (
                            <div key={i} className="h-12 bg-gray-200 rounded-md"></div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      sensorData
                        .slice(-10)
                        .reverse()
                        .map((data, index) => (
                          <motion.div 
                            key={index} 
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1, duration: 0.3 }}
                            className="flex justify-between items-center p-3 bg-gray-50 rounded-md hover:bg-gray-100 transition-colors"
                          >
                            <div className="flex space-x-4">
                              <span className="text-sm text-gray-600">
                                {data.time}
                              </span>
                              <span className="text-sm font-medium">
                                Temp: {data.temperature.toFixed(1)}°C
                              </span>
                              <span className="text-sm font-medium">
                                Humidity: {data.humidity.toFixed(1)}%
                              </span>
                            </div>
                            <motion.div 
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              transition={{ delay: index * 0.1 + 0.2, type: "spring", stiffness: 200 }}
                              className={`px-2 py-1 rounded text-xs font-medium ${
                                data.temperature >= 2 && data.temperature <= 8 
                                  ? 'bg-green-100 text-green-800' 
                                  : 'bg-red-100 text-red-800'
                              }`}
                            >
                              {data.temperature >= 2 && data.temperature <= 8 ? 'SAFE' : 'WARNING'}
                            </motion.div>
                          </motion.div>
                        ))
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'analytics' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900">Analytics & Reports</h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Monthly Usage Trends</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={analyticsData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="value" fill="#3B82F6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Inventory Value Distribution</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {pieData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'users' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900">User Management</h2>
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                  <p className="text-gray-600">User management features coming soon...</p>
                </div>
              </div>
            )}

            {activeTab === 'settings' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">Settings</h2>
                    <p className="text-gray-600">Manage system-wide configurations and security</p>
                  </div>
                  <div className="flex space-x-3">
                    <button className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                      Reset to Default
                    </button>
                    <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                      Save Changes
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  {/* Settings Navigation */}
                  <div className="lg:col-span-1">
                    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                      <nav className="space-y-2">
                        <button className="w-full text-left px-3 py-2 text-blue-600 bg-blue-50 rounded-md font-medium">
                          General
                        </button>
                        <button className="w-full text-left px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-md transition-colors">
                          Notifications
                        </button>
                        <button className="w-full text-left px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-md transition-colors">
                          Appearance
                        </button>
                        <button className="w-full text-left px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-md transition-colors">
                          Security
                        </button>
                        <button className="w-full text-left px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-md transition-colors">
                          Role Settings
                        </button>
                      </nav>
                    </div>
                  </div>

                  {/* Settings Content */}
                  <div className="lg:col-span-2">
                    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                      <h3 className="text-lg font-semibold text-gray-900 mb-6">General Settings</h3>
                      
                      <div className="space-y-6">
                        {/* Account Information */}
                        <div>
                          <h4 className="text-md font-medium text-gray-900 mb-4">Account Information</h4>
                          <div className="space-y-4">
                            <div>
                              <label className="block text-sm font-medium text-gray-700 mb-2">
                                Email Address
                              </label>
                              <input
                                type="email"
                                defaultValue="user@medchain.com"
                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                              />
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-gray-700 mb-2">
                                Language
                              </label>
                              <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                                <option>English</option>
                                <option>Spanish</option>
                                <option>French</option>
                                <option>German</option>
                              </select>
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-gray-700 mb-2">
                                Timezone
                              </label>
                              <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                                <option>UTC</option>
                                <option>EST (UTC-5)</option>
                                <option>PST (UTC-8)</option>
                                <option>GMT (UTC+0)</option>
                                <option>CET (UTC+1)</option>
                              </select>
                            </div>
                          </div>
                        </div>

                        {/* Device Settings */}
                        <div>
                          <h4 className="text-md font-medium text-gray-900 mb-4">Device Settings</h4>
                          <div className="space-y-4">
                            <div className="flex items-center justify-between">
                              <div>
                                <h5 className="text-sm font-medium text-gray-900">Desktop Notifications</h5>
                                <p className="text-sm text-gray-600">Receive notifications on desktop</p>
                              </div>
                              <label className="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" defaultChecked className="sr-only peer" />
                                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                              </label>
                            </div>
                            
                            <div className="flex items-center justify-between">
                              <div>
                                <h5 className="text-sm font-medium text-gray-900">Mobile Notifications</h5>
                                <p className="text-sm text-gray-600">Receive notifications on mobile</p>
                              </div>
                              <label className="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" defaultChecked className="sr-only peer" />
                                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                              </label>
                            </div>
                          </div>
                        </div>

                        {/* Security Settings */}
                        <div>
                          <h4 className="text-md font-medium text-gray-900 mb-4">Security</h4>
                          <div className="space-y-4">
                            <div className="flex items-center justify-between">
                              <div>
                                <h5 className="text-sm font-medium text-gray-900">Auto-logout</h5>
                                <p className="text-sm text-gray-600">Automatically logout after inactivity</p>
                              </div>
                              <label className="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" className="sr-only peer" />
                                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                              </label>
                            </div>
                            
                            <div>
                              <label className="block text-sm font-medium text-gray-700 mb-2">
                                Auto-logout Timeout (minutes)
                              </label>
                              <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                                <option>15</option>
                                <option>30</option>
                                <option>60</option>
                                <option>120</option>
                              </select>
                            </div>
                          </div>
                        </div>

                        {/* Role Settings */}
                        <div>
                          <h4 className="text-md font-medium text-gray-900 mb-4">Role Settings</h4>
                          <div className="space-y-4">
                            <div>
                              <label className="block text-sm font-medium text-gray-700 mb-2">
                                Current Role
                              </label>
                              <div className="px-3 py-2 bg-gray-50 border border-gray-200 rounded-md">
                                <span className="text-sm text-gray-900 font-medium">Administrator</span>
                                <span className="ml-2 text-xs text-gray-500">Full system access</span>
                              </div>
                            </div>
                            
                            <div>
                              <label className="block text-sm font-medium text-gray-700 mb-2">
                                Permissions
                              </label>
                              <div className="space-y-2">
                                <div className="flex items-center">
                                  <input type="checkbox" defaultChecked className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded" />
                                  <label className="ml-2 text-sm text-gray-700">View Inventory</label>
                                </div>
                                <div className="flex items-center">
                                  <input type="checkbox" defaultChecked className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded" />
                                  <label className="ml-2 text-sm text-gray-700">Edit Inventory</label>
                                </div>
                                <div className="flex items-center">
                                  <input type="checkbox" defaultChecked className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded" />
                                  <label className="ml-2 text-sm text-gray-700">Approve Batches</label>
                                </div>
                                <div className="flex items-center">
                                  <input type="checkbox" defaultChecked className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded" />
                                  <label className="ml-2 text-sm text-gray-700">View Analytics</label>
                                </div>
                                <div className="flex items-center">
                                  <input type="checkbox" defaultChecked className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded" />
                                  <label className="ml-2 text-sm text-gray-700">Manage Users</label>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'ai-drug-verification' && (
              <AIDrugVerification />
            )}

            {activeTab === 'patient-adherence' && (
              <PatientAdherence />
            )}

            {activeTab === 'ai-analytics' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900">AI Analytics Dashboard</h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">ML Model Performance</h3>
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Drug Verification Accuracy</span>
                          <span className="font-semibold">94.2%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="bg-green-500 h-2 rounded-full" style={{ width: '94.2%' }}></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Anomaly Detection Precision</span>
                          <span className="font-semibold">89.7%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="bg-blue-500 h-2 rounded-full" style={{ width: '89.7%' }}></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Adherence Prediction</span>
                          <span className="font-semibold">91.3%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="bg-purple-500 h-2 rounded-full" style={{ width: '91.3%' }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Insights Summary</h3>
                    <div className="space-y-3">
                      <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
                        <CheckCircle className="w-5 h-5 text-green-600" />
                        <div>
                          <p className="text-sm font-medium text-green-800">System Health: Excellent</p>
                          <p className="text-xs text-green-600">All AI models performing optimally</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3 p-3 bg-blue-50 rounded-lg">
                        <Brain className="w-5 h-5 text-blue-600" />
                        <div>
                          <p className="text-sm font-medium text-blue-800">Learning Progress: Active</p>
                          <p className="text-xs text-blue-600">Models improving with new data</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3 p-3 bg-purple-50 rounded-lg">
                        <TrendingUp className="w-5 h-5 text-purple-600" />
                        <div>
                          <p className="text-sm font-medium text-purple-800">Efficiency: +23%</p>
                          <p className="text-xs text-purple-600">Automation reducing manual work</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Dashboard;
