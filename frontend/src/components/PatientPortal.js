import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';

const PatientPortal = () => {
  const { token } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');
  

  // Skin Analysis State
  const [skinTab, setSkinTab] = useState('nail');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [skinResult, setSkinResult] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Parkinson's detection states
  const [drawingType, setDrawingType] = useState('spiral');
  const [parkinsonsImage, setParkinsonsImage] = useState(null);
  const [parkinsonsResult, setParkinsonsResult] = useState(null);
  const [parkinsonsLoading, setParkinsonsLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  // Disease Prediction State
  const [diseaseSymptoms, setDiseaseSymptoms] = useState('');
  const [diseaseHistory, setDiseaseHistory] = useState('');
  const [diseasePrediction, setDiseasePrediction] = useState(null);
  const [predictingDisease, setPredictingDisease] = useState(false);

  // Chat State
  const [availableDoctors, setAvailableDoctors] = useState([]);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [chatSessions, setChatSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [websocket, setWebsocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const messagesEndRef = useRef(null);


  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setAnalysisResult(null);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleParkinsonsImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setParkinsonsImage(file);
      setParkinsonsResult(null);
    }
  };

  const analyzeSkin = async () => {
    if (!selectedImage) return;

    setAnalyzing(true);
    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch('http://localhost:8000/ai/skin/predict', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setAnalysisResult(result);
    } catch (error) {
      console.error('Error analyzing skin:', error);
      setAnalysisResult({ error: 'Failed to analyze image. Please try again.' });
    } finally {
      setAnalyzing(false);
    }
  };

  const analyzeParkinsonsDrawing = async () => {
    if (!parkinsonsImage) return;

    setParkinsonsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', parkinsonsImage);
      formData.append('drawing_type', drawingType);

      const response = await fetch('http://localhost:8000/ai/parkinsons/predict', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setParkinsonsResult(result);
    } catch (error) {
      console.error('Error analyzing drawing:', error);
      setParkinsonsResult({ error: 'Failed to analyze drawing. Please try again.' });
    } finally {
      setParkinsonsLoading(false);
    }
  };

  const analyzeNailImage = async () => {
    if (!selectedImage) return;

    setAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const endpoint = 'http://localhost:8000/ai/skin/predict';
      console.log(`Analyzing nail using endpoint: ${endpoint}`);
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });

      const data = await response.json();
      console.log(`Nail analysis result:`, data);
      setSkinResult(data);
      
      if (!response.ok) {
        console.error('Analysis failed:', data);
      }
    } catch (error) {
      console.error('Analysis error:', error);
    } finally {
      setAnalyzing(false);
    }
  };

  const predictDiseaseRisk = async () => {
    if (!diseaseSymptoms.trim()) return;
    
    setPredictingDisease(true);
    try {
      const response = await fetch('http://localhost:8000/ai/disease/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ 
          symptoms: diseaseSymptoms,
          history: diseaseHistory 
        })
      });

      if (response.ok) {
        const data = await response.json();
        setDiseasePrediction(data);
      }
    } catch (error) {
      console.error('Disease prediction error:', error);
    } finally {
      setPredictingDisease(false);
    }
  };

  // Chat Functions
  useEffect(() => {
    if (activeTab === 'chat') {
      fetchAvailableDoctors();
      fetchChatSessions();
    }
  }, [activeTab]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const fetchAvailableDoctors = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat/doctors/available', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setAvailableDoctors(data.doctors);
      }
    } catch (error) {
      console.error('Error fetching doctors:', error);
    }
  };

  const fetchChatSessions = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat/sessions/patient@medcare.com', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setChatSessions(data.sessions);
      }
    } catch (error) {
      console.error('Error fetching chat sessions:', error);
    }
  };

  const createChatSession = async (doctorId) => {
    try {
      const response = await fetch('http://localhost:8000/chat/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          patient_id: 'patient@medcare.com',
          doctor_id: doctorId
        })
      });

      if (response.ok) {
        const data = await response.json();
        await fetchChatSessions();
        joinChatSession(data.session_id, data.room_id);
      }
    } catch (error) {
      console.error('Error creating chat session:', error);
    }
  };

  const joinChatSession = async (sessionId, roomId) => {
    setActiveSession({ session_id: sessionId, room_id: roomId });
    
    // Fetch existing messages
    try {
      const response = await fetch(`http://localhost:8000/chat/messages/${sessionId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setMessages(data.messages);
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
    }

    // Connect to WebSocket
    connectWebSocket(roomId);
  };

  const connectWebSocket = (roomId) => {
    if (websocket) {
      websocket.close();
    }

    const ws = new WebSocket(`ws://localhost:8000/ws/chat/patient@medcare.com`);
    
    ws.onopen = () => {
      console.log('Connected to chat WebSocket');
      setIsConnected(true);
      
      // Join the room
      ws.send(JSON.stringify({
        action: 'join_room',
        room_id: roomId
      }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'message') {
        setMessages(prev => [...prev, data]);
      } else if (data.type === 'user_joined') {
        console.log(data.message);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from chat WebSocket');
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    setWebsocket(ws);
  };

  const sendMessage = () => {
    if (!newMessage.trim() || !websocket || !activeSession) return;

    const messageData = {
      action: 'send_message',
      room_id: activeSession.room_id,
      session_id: activeSession.session_id,
      message: newMessage
    };

    websocket.send(JSON.stringify(messageData));
    setNewMessage('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-purple-50 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute w-96 h-96 rounded-full blur-xl bg-gradient-to-r from-green-400/40 to-blue-400/40" 
             style={{top: '10%', left: '10%'}} />
        <div className="absolute w-80 h-80 rounded-full blur-xl bg-gradient-to-r from-blue-400/40 to-purple-400/40" 
             style={{top: '60%', right: '15%'}} />
      </div>

      {/* Header */}
      <div className="relative z-10 bg-white/20 backdrop-blur-md border-b border-gray-200/50 px-6 py-4">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
          Patient Portal
        </h1>
        <p className="text-gray-700 mt-1">Your Health & Wellness Dashboard</p>
      </div>

      {/* Main Content */}
      <div className="relative z-10 p-6">
      
      {/* Tab Navigation */}
      <div className="bg-white/30 backdrop-blur-sm border border-gray-200/50 rounded-xl p-2 mb-6">
        <nav className="flex space-x-2">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`py-3 px-6 rounded-lg font-medium text-sm transition-all ${
              activeTab === 'dashboard'
                ? 'bg-gradient-to-r from-green-600 to-blue-600 text-white shadow-lg'
                : 'text-gray-600 hover:bg-white/50'
            }`}
          >
            🏠 Dashboard
          </button>
          <button
            onClick={() => setActiveTab('skin')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'skin'
                ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                : 'bg-white/70 text-gray-700 hover:bg-white/90'
            }`}
          >
            💅 Nail Analysis
          </button>
          <button
            onClick={() => setActiveTab('parkinsons')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'parkinsons'
                ? 'bg-gradient-to-r from-blue-500 to-teal-500 text-white shadow-lg'
                : 'bg-white/70 text-gray-700 hover:bg-white/90'
            }`}
          >
            🧠 Parkinson's Test
          </button>
          <button
            onClick={() => setActiveTab('disease')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'disease'
                ? 'bg-gradient-to-r from-red-500 to-purple-500 text-white shadow-lg'
                : 'bg-white/70 text-gray-700 hover:bg-white/90'
            }`}
          >
            🔬 Disease Prediction
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`py-3 px-6 rounded-lg font-medium text-sm transition-all ${
              activeTab === 'chat'
                ? 'bg-gradient-to-r from-green-600 to-blue-600 text-white shadow-lg'
                : 'text-gray-600 hover:bg-white/50'
            }`}
          >
            💬 Chat with Doctor
          </button>
        </nav>
      </div>

      {/* Dashboard Tab */}
      {activeTab === 'dashboard' && (
        <div className="space-y-6">
          {/* Health Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Health Score</p>
                  <p className="text-3xl font-bold text-gray-900">92</p>
                  <p className="text-sm text-green-600">↗ Excellent</p>
                </div>
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center text-2xl">
                  💚
                </div>
              </div>
            </div>
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Medications</p>
                  <p className="text-3xl font-bold text-gray-900">3</p>
                  <p className="text-sm text-blue-600">87% adherence</p>
                </div>
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center text-2xl">
                  💊
                </div>
              </div>
            </div>
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Next Appointment</p>
                  <p className="text-lg font-bold text-gray-900">Sept 5</p>
                  <p className="text-sm text-purple-600">Dr. Smith</p>
                </div>
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center text-2xl">
                  📅
                </div>
              </div>
            </div>
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Checkups</p>
                  <p className="text-3xl font-bold text-gray-900">12</p>
                  <p className="text-sm text-green-600">This year</p>
                </div>
                <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center text-2xl">
                  🩺
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">🚀 Quick Actions</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <button 
                onClick={() => setActiveTab('skin')}
                className="bg-gradient-to-r from-green-500 to-blue-500 text-white p-4 rounded-lg hover:from-green-600 hover:to-blue-600 transition-all"
              >
                <div className="text-2xl mb-2">🔬</div>
                <div className="font-semibold">Nail Analysis</div>
                <div className="text-sm opacity-90">Upload photos for analysis</div>
              </button>
              <button 
                onClick={() => setActiveTab('disease')}
                className="bg-gradient-to-r from-red-500 to-purple-500 text-white p-4 rounded-lg hover:from-red-600 hover:to-purple-600 transition-all"
              >
                <div className="text-2xl mb-2">🤖</div>
                <div className="font-semibold">AI Disease Analysis</div>
                <div className="text-sm opacity-90">Powered by Groq (Llama3)</div>
              </button>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">📋 Recent Activity</h3>
            <div className="space-y-3">
              {[
                { type: 'medication', message: 'Took Metformin 500mg', time: '2 hours ago', status: 'success' },
                { type: 'appointment', message: 'Scheduled follow-up with Dr. Smith', time: '1 day ago', status: 'info' },
                { type: 'analysis', message: 'Completed skin analysis - results normal', time: '3 days ago', status: 'success' },
                { type: 'symptom', message: 'Reported mild headache symptoms', time: '5 days ago', status: 'warning' }
              ].map((activity, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-white/50 border border-gray-200 rounded-lg">
                  <div className="flex items-center gap-3">
                    <span className={`w-3 h-3 rounded-full ${
                      activity.status === 'success' ? 'bg-green-500' :
                      activity.status === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                    }`}></span>
                    <p className="text-gray-900">{activity.message}</p>
                  </div>
                  <p className="text-sm text-gray-500">{activity.time}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Parkinson's Detection Tab */}
      {activeTab === 'parkinsons' && (
        <div className="space-y-6">
          <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
            {/* Parkinson's Detection Header */}
            <div className="text-center mb-6">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">
                🧠 Parkinson's Disease Detection
              </h2>
              <p className="text-gray-600 mt-2">Upload a drawing (spiral or wave) for AI-powered Parkinson's screening</p>
            </div>

            {/* Drawing Type Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium mb-3 text-gray-700">
                📝 Select drawing type:
              </label>
              <div className="flex gap-4">
                <button
                  onClick={() => setDrawingType('spiral')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    drawingType === 'spiral'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  🌀 Spiral Drawing
                </button>
                <button
                  onClick={() => setDrawingType('wave')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    drawingType === 'wave'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  〰️ Wave Drawing
                </button>
              </div>
            </div>

            {/* Image Upload */}
            <div className="mb-6">
              <label className="block text-sm font-medium mb-3 text-gray-700">
                📸 Upload your {drawingType} drawing:
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center bg-white/50">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleParkinsonsImageSelect}
                  className="hidden"
                  id="parkinsons-upload"
                />
                <label htmlFor="parkinsons-upload" className="cursor-pointer">
                  <div className="text-gray-500">
                    <svg className="mx-auto h-12 w-12 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                      <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <p className="text-lg font-medium">Click to upload {drawingType} drawing</p>
                    <p className="text-sm">PNG, JPG up to 10MB</p>
                  </div>
                </label>
              </div>
            </div>

            {/* Selected Image Preview */}
            {parkinsonsImage && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3">📋 Selected Image:</h3>
                <div className="bg-white rounded-lg p-4 border">
                  <img 
                    src={URL.createObjectURL(parkinsonsImage)} 
                    alt="Selected drawing" 
                    className="max-w-full h-48 object-contain mx-auto rounded-lg"
                  />
                  <p className="text-center text-sm text-gray-600 mt-2">{parkinsonsImage.name}</p>
                </div>
              </div>
            )}

            {/* Analyze Button */}
            <div className="text-center mb-6">
              <button
                onClick={analyzeParkinsonsDrawing}
                disabled={!parkinsonsImage || parkinsonsLoading}
                className="bg-gradient-to-r from-blue-500 to-teal-500 text-white px-8 py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-teal-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {parkinsonsLoading ? '🔄 Analyzing...' : '🔍 Analyze Drawing'}
              </button>
            </div>

            {/* Results */}
            {parkinsonsResult && (
              <div className="bg-white rounded-lg p-6 border-l-4 border-blue-500">
                <h3 className="text-xl font-semibold text-gray-900 mb-4">📊 Analysis Results</h3>
                
                {parkinsonsResult.error ? (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p className="text-red-800">❌ {parkinsonsResult.error}</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Prediction Result */}
                    <div className={`p-4 rounded-lg border-l-4 ${
                      parkinsonsResult.prediction === 'parkinson' 
                        ? 'bg-orange-50 border-orange-400' 
                        : 'bg-green-50 border-green-400'
                    }`}>
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-semibold text-lg">
                            {parkinsonsResult.prediction === 'parkinson' ? '⚠️ Indicators Detected' : '✅ No Indicators'}
                          </h4>
                          <p className="text-sm text-gray-600">
                            Confidence: {(parkinsonsResult.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                          parkinsonsResult.risk_level === 'High Risk' ? 'bg-red-100 text-red-800' :
                          parkinsonsResult.risk_level === 'Moderate Risk' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {parkinsonsResult.risk_level}
                        </div>
                      </div>
                    </div>

                    {/* Probabilities */}
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h5 className="font-medium mb-2">📈 Detailed Probabilities:</h5>
                      <div className="space-y-2">
                        {Object.entries(parkinsonsResult.probabilities || {}).map(([key, value]) => (
                          <div key={key} className="flex justify-between items-center">
                            <span className="capitalize">{key}:</span>
                            <div className="flex items-center gap-2">
                              <div className="w-24 bg-gray-200 rounded-full h-2">
                                <div 
                                  className="bg-blue-500 h-2 rounded-full" 
                                  style={{ width: `${value * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-sm font-medium">{(value * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Recommendation */}
                    <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                      <h5 className="font-medium text-blue-900 mb-2">💡 Recommendation:</h5>
                      <p className="text-blue-800">{parkinsonsResult.recommendation}</p>
                    </div>

                    {/* Disclaimer */}
                    <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                      <h5 className="font-medium text-yellow-900 mb-2">⚠️ Important Notice:</h5>
                      <p className="text-yellow-800 text-sm">
                        This is an AI screening tool and should not replace professional medical diagnosis. 
                        Please consult a healthcare professional for proper evaluation and treatment.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Nail Analysis Tab */}
      {activeTab === 'skin' && (
        <div className="space-y-6">
          <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
            {/* Nail Analysis Header */}
            <div className="text-center mb-6">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                💅 Nail Health Analysis
              </h2>
              <p className="text-gray-600 mt-2">Upload a clear photo of your nails for AI-powered health analysis</p>
            </div>

            {/* Image Upload */}
            <div className="mb-6">
              <label className="block text-sm font-medium mb-3 text-gray-700">
                📸 Upload nail image for AI analysis:
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center bg-white/50">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                  id="image-upload"
                />
                <label htmlFor="image-upload" className="cursor-pointer">
                  <div className="text-4xl mb-2">📷</div>
                  <p className="text-gray-600 mb-2">Click to upload or drag and drop</p>
                  <p className="text-sm text-gray-500">PNG, JPG, GIF up to 10MB</p>
                </label>
              </div>
            </div>

            {/* Image Preview */}
            {imagePreview && (
              <div className="mb-6">
                <h3 className="text-sm font-medium mb-2 text-gray-700">📋 Image Preview:</h3>
                <div className="bg-white/50 border border-gray-200 rounded-xl p-4 inline-block">
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="max-w-xs max-h-64 object-contain rounded-lg shadow-sm"
                  />
                </div>
              </div>
            )}

            <button
              onClick={analyzeNailImage}
              disabled={!selectedImage || analyzing}
              className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 py-3 rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50"
            >
              {analyzing ? '🔄 Analyzing...' : '🔬 Analyze Nails'}
            </button>
          </div>

          {/* Analysis Results */}
          {skinResult && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Nail Analysis Results</h2>
              <div className="space-y-2">
                <p><strong>Classification:</strong> {skinResult.label}</p>
                <p><strong>Confidence:</strong> {(skinResult.confidence * 100).toFixed(1)}%</p>
                
                {/* Display message if available */}
                {skinResult.message && (
                  <div className={`mt-3 p-3 rounded-md ${
                    skinResult.label === 'Invalid image' || skinResult.label === 'Poor image quality' || skinResult.label === 'Uncertain classification'
                      ? 'bg-red-50 border border-red-200'
                      : 'bg-blue-50 border border-blue-200'
                  }`}>
                    <p className={`text-sm ${
                      skinResult.label === 'Invalid image' || skinResult.label === 'Poor image quality' || skinResult.label === 'Uncertain classification'
                        ? 'text-red-800'
                        : 'text-blue-800'
                    }`}>
                      <strong>Analysis:</strong> {skinResult.message}
                    </p>
                  </div>
                )}
                
                {skinResult.model_info && (
                  <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-md">
                    <p className="text-sm text-blue-800">
                      <strong>Model:</strong> {skinResult.model_info.type} (Accuracy: {skinResult.model_info.accuracy}%)
                    </p>
                  </div>
                )}
                <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-md">
                  <p className="text-sm text-yellow-800">
                    <strong>Disclaimer:</strong> This AI analysis is for informational purposes only. 
                    Please consult with a dermatologist for proper diagnosis and treatment.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Disease Prediction Tab */}
      {activeTab === 'disease' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Disease Risk Assessment</h2>
            
            {/* Symptoms Input */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                Describe your symptoms in detail:
              </label>
              <textarea
                value={diseaseSymptoms}
                onChange={(e) => setDiseaseSymptoms(e.target.value)}
                placeholder="Please describe all symptoms you are experiencing, including duration, severity, and any patterns you've noticed..."
                className="w-full p-3 border rounded-md h-32 resize-none"
              />
            </div>

            {/* Medical History */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                Medical History (Optional):
              </label>
              <textarea
                value={diseaseHistory}
                onChange={(e) => setDiseaseHistory(e.target.value)}
                placeholder="Any relevant medical history, current medications, family history, or previous conditions..."
                className="w-full p-3 border rounded-md h-24 resize-none"
              />
            </div>

            <button
              onClick={predictDiseaseRisk}
              disabled={!diseaseSymptoms.trim() || predictingDisease}
              className="bg-red-500 text-white px-6 py-2 rounded-md hover:bg-red-600 disabled:opacity-50"
            >
              {predictingDisease ? 'Analyzing Risk...' : 'Assess Disease Risk'}
            </button>
          </div>

          {/* Disease Prediction Results */}
          {diseasePrediction && (
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4 bg-gradient-to-r from-red-600 to-purple-600 bg-clip-text text-transparent">
                🎯 Risk Assessment Results
              </h2>
              <div className="space-y-4">
                {/* Primary Prediction */}
                {diseasePrediction.primary_prediction && (
                  <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-blue-900 mb-2">🏥 Primary Prediction</h3>
                    <p className="text-xl font-bold text-blue-800">{diseasePrediction.primary_prediction}</p>
                    <p className="text-md text-blue-700 mt-1">
                      Confidence: <span className="font-semibold">{diseasePrediction.confidence_percentage}%</span>
                    </p>
                  </div>
                )}

                {/* Top Predictions */}
                {diseasePrediction.top_predictions && diseasePrediction.top_predictions.length > 0 && (
                  <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-green-900 mb-3">📊 Top Predictions</h3>
                    <div className="space-y-2">
                      {diseasePrediction.top_predictions.slice(0, 3).map((pred, index) => (
                        <div key={index} className="flex justify-between items-center p-2 bg-white/50 rounded-md">
                          <span className="font-medium text-gray-800">
                            {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'} {pred.disease}
                          </span>
                          <span className="font-bold text-blue-600">{pred.probability}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Analysis Details */}
                {diseasePrediction.analysis && (
                  <div className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-purple-900 mb-3">🔬 Analysis Details</h3>
                    <div className="space-y-2 text-sm">
                      <p><strong>Method:</strong> {diseasePrediction.analysis.method || diseasePrediction.analysis.model_type}</p>
                      <p><strong>Symptoms Analyzed:</strong> {diseasePrediction.analysis.symptoms_analyzed}</p>
                      {diseasePrediction.analysis.total_diseases && (
                        <p><strong>Diseases in Database:</strong> {diseasePrediction.analysis.total_diseases}</p>
                      )}
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {diseasePrediction.recommendations && diseasePrediction.recommendations.length > 0 && (
                  <div className="p-4 bg-gradient-to-r from-green-50 to-teal-50 border border-green-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-green-900 mb-3">💡 Recommendations</h3>
                    <ul className="space-y-2">
                      {diseasePrediction.recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start gap-2">
                          <span className="text-green-600 mt-1">•</span>
                          <span className="text-gray-800">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Urgency Level */}
                {diseasePrediction.urgency_level && (
                  <div className={`p-4 border rounded-lg ${
                    diseasePrediction.urgency_level === 'high' ? 'bg-red-50 border-red-200' :
                    diseasePrediction.urgency_level === 'medium' ? 'bg-yellow-50 border-yellow-200' :
                    'bg-green-50 border-green-200'
                  }`}>
                    <h3 className="text-lg font-semibold mb-2">
                      {diseasePrediction.urgency_level === 'high' ? '🚨 High Priority' :
                       diseasePrediction.urgency_level === 'medium' ? '⚠️ Medium Priority' :
                       '✅ Low Priority'}
                    </h3>
                    <p className={
                      diseasePrediction.urgency_level === 'high' ? 'text-red-800' :
                      diseasePrediction.urgency_level === 'medium' ? 'text-yellow-800' :
                      'text-green-800'
                    }>
                      {diseasePrediction.seek_immediate_care ? 
                        '🏥 Consider seeking immediate medical attention' :
                        '📅 Schedule a routine consultation with your healthcare provider'
                      }
                    </p>
                  </div>
                )}

                {/* AI Provider Info */}
                {diseasePrediction.ai_provider && (
                  <div className="p-3 bg-gradient-to-r from-indigo-50 to-blue-50 border border-indigo-200 rounded-lg">
                    <p className="text-sm text-indigo-800">
                      <strong>🤖 Powered by:</strong> {diseasePrediction.ai_provider}
                    </p>
                  </div>
                )}

                {/* Medical Disclaimers */}
                <div className="space-y-3">
                  {diseasePrediction.disclaimer && (
                    <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                      <p className="text-sm text-red-800 font-medium">
                        {diseasePrediction.disclaimer}
                      </p>
                    </div>
                  )}
                  
                  {diseasePrediction.emergency_note && (
                    <div className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
                      <p className="text-sm text-orange-800 font-medium">
                        {diseasePrediction.emergency_note}
                      </p>
                    </div>
                  )}
                </div>

                {/* Error Handling */}
                {diseasePrediction.error && (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-red-900 mb-2">❌ Service Error</h3>
                    <p className="text-red-800">{diseasePrediction.message}</p>
                    {diseasePrediction.fallback_advice && (
                      <p className="text-red-700 mt-2 text-sm">{diseasePrediction.fallback_advice}</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Chat Tab */}
      {activeTab === 'chat' && (
        <div className="space-y-6">
          {!activeSession ? (
            <div className="space-y-6">
              {/* Available Doctors */}
              <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  💬 Available Doctors
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {availableDoctors.map((doctor) => (
                    <div key={doctor.id} className="bg-white/50 border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="font-semibold text-gray-900">{doctor.name}</h3>
                        <span className={`w-3 h-3 rounded-full ${doctor.online ? 'bg-green-500' : 'bg-gray-400'}`}></span>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{doctor.specialty}</p>
                      <p className="text-xs text-gray-500 mb-3">{doctor.response_time}</p>
                      <button
                        onClick={() => createChatSession(doctor.id)}
                        className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white py-2 px-4 rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all text-sm"
                      >
                        {doctor.online ? '💬 Chat Now' : '📧 Send Message'}
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Previous Chat Sessions */}
              {chatSessions.length > 0 && (
                <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                  <h2 className="text-xl font-semibold mb-4 bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
                    📋 Previous Conversations
                  </h2>
                  <div className="space-y-3">
                    {chatSessions.map((session) => (
                      <div key={session.session_id} className="bg-white/50 border border-gray-200 rounded-lg p-4 hover:bg-white/70 transition-all cursor-pointer"
                           onClick={() => joinChatSession(session.session_id, session.room_id)}>
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="font-medium text-gray-900">Chat with {session.other_user_id}</h3>
                            <p className="text-sm text-gray-600">{session.message_count} messages</p>
                          </div>
                          <div className="text-right">
                            <p className="text-xs text-gray-500">{new Date(session.created_at).toLocaleDateString()}</p>
                            <span className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                              {session.status}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Active Chat Interface */
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl overflow-hidden">
              {/* Chat Header */}
              <div className="bg-gradient-to-r from-blue-500 to-purple-500 text-white p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => {
                        setActiveSession(null);
                        setMessages([]);
                        if (websocket) websocket.close();
                      }}
                      className="text-white hover:bg-white/20 p-2 rounded-lg transition-all"
                    >
                      ←
                    </button>
                    <div>
                      <h3 className="font-semibold">Medical Chat</h3>
                      <p className="text-sm opacity-90">
                        {isConnected ? '🟢 Connected' : '🔴 Connecting...'}
                      </p>
                    </div>
                  </div>
                  <div className="text-sm opacity-90">
                    Session #{activeSession.session_id}
                  </div>
                </div>
              </div>

              {/* Messages Area */}
              <div className="h-96 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 ? (
                  <div className="text-center text-gray-500 py-8">
                    <div className="text-4xl mb-2">💬</div>
                    <p>Start a conversation with your doctor</p>
                    <p className="text-sm mt-1">If no doctor is available, our AI assistant will help you</p>
                  </div>
                ) : (
                  messages.map((message) => (
                    <div key={message.id} className={`flex ${message.sender_type === 'patient' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                        message.sender_type === 'patient' 
                          ? 'bg-blue-500 text-white' 
                          : message.is_ai_response
                          ? 'bg-purple-100 text-purple-900 border border-purple-200'
                          : 'bg-gray-100 text-gray-900'
                      }`}>
                        {message.is_ai_response && (
                          <div className="text-xs opacity-75 mb-1">🤖 AI Assistant</div>
                        )}
                        <div className="text-sm whitespace-pre-line">{message.message}</div>
                        <p className="text-xs opacity-75 mt-1">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Message Input */}
              <div className="border-t border-gray-200 p-4">
                <div className="flex gap-2">
                  <textarea
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your message... (Press Enter to send)"
                    className="flex-1 p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows="2"
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!newMessage.trim() || !isConnected}
                    className="bg-gradient-to-r from-blue-500 to-purple-500 text-white px-6 py-3 rounded-lg hover:from-blue-600 hover:to-purple-600 disabled:opacity-50 transition-all"
                  >
                    Send
                  </button>
                </div>
                <div className="text-xs text-gray-500 mt-2">
                  {isConnected ? (
                    '💡 If no doctor is available, our AI assistant will respond to help you'
                  ) : (
                    '⚠️ Connecting to chat service...'
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      </div>
    </div>
  );
};

export default PatientPortal;
