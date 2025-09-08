import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const DoctorPortal = () => {
  const { token } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // Adherence State
  const [patientId, setPatientId] = useState('');
  const [adherenceData, setAdherenceData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Chat State
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [chatSessions, setChatSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [websocket, setWebsocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const messagesEndRef = useRef(null);



  const fetchAdherence = async () => {
    if (!patientId.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/adherence/${patientId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setAdherenceData(data);
      } else {
        alert('Patient not found or access denied');
      }
    } catch (error) {
      console.error('Error fetching adherence:', error);
      alert('Error fetching data');
    } finally {
      setLoading(false);
    }
  };

  // Chat Functions
  useEffect(() => {
    if (activeTab === 'chat') {
      fetchPatients();
      fetchChatSessions();
      fetchNotifications();
    }
  }, [activeTab]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const fetchPatients = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat/patients', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setPatients(data.patients);
      }
    } catch (error) {
      console.error('Error fetching patients:', error);
    }
  };

  const fetchChatSessions = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat/sessions/doctor@medcare.com', {
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

  const fetchNotifications = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat/notifications/doctor@medcare.com', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setNotifications(data.notifications);
      }
    } catch (error) {
      console.error('Error fetching notifications:', error);
    }
  };

  const createChatSession = async (patientId) => {
    try {
      const response = await fetch('http://localhost:8000/chat/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          patient_id: patientId,
          doctor_id: 'doctor@medcare.com'
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

    const ws = new WebSocket(`ws://localhost:8000/ws/chat/doctor@medcare.com`);
    
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute w-96 h-96 rounded-full blur-xl bg-gradient-to-r from-blue-400/40 to-purple-400/40" 
             style={{top: '10%', left: '10%'}} />
        <div className="absolute w-80 h-80 rounded-full blur-xl bg-gradient-to-r from-purple-400/40 to-pink-400/40" 
             style={{top: '60%', right: '15%'}} />
      </div>

      {/* Header */}
      <div className="relative z-10 bg-white/20 backdrop-blur-md border-b border-gray-200/50 px-6 py-4">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Doctor Portal
        </h1>
        <p className="text-gray-700 mt-1">Patient Care & Monitoring Dashboard</p>
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
                ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                : 'text-gray-600 hover:bg-white/50'
            }`}
          >
            📊 Dashboard
          </button>
          <button
            onClick={() => setActiveTab('adherence')}
            className={`py-3 px-6 rounded-lg font-medium text-sm transition-all ${
              activeTab === 'adherence'
                ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                : 'text-gray-600 hover:bg-white/50'
            }`}
          >
            💊 Patient Adherence
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`py-3 px-6 rounded-lg font-medium text-sm transition-all ${
              activeTab === 'chat'
                ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                : 'text-gray-600 hover:bg-white/50'
            }`}
          >
            💬 Patient Chat
            {notifications.length > 0 && (
              <span className="ml-2 bg-red-500 text-white text-xs rounded-full px-2 py-1">
                {notifications.length}
              </span>
            )}
          </button>
        </nav>
      </div>

      {/* Dashboard Tab */}
      {activeTab === 'dashboard' && (
        <div className="space-y-6">
          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Total Patients</p>
                  <p className="text-3xl font-bold text-gray-900">127</p>
                  <p className="text-sm text-green-600">↗ +8% this month</p>
                </div>
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center text-2xl">
                  👥
                </div>
              </div>
            </div>
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Avg Adherence</p>
                  <p className="text-3xl font-bold text-gray-900">84.2%</p>
                  <p className="text-sm text-green-600">↗ +2.1% this week</p>
                </div>
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center text-2xl">
                  ✅
                </div>
              </div>
            </div>
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Critical Alerts</p>
                  <p className="text-3xl font-bold text-gray-900">3</p>
                  <p className="text-sm text-red-600">↗ +1 today</p>
                </div>
                <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center text-2xl">
                  🚨
                </div>
              </div>
            </div>
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Consultations</p>
                  <p className="text-3xl font-bold text-gray-900">42</p>
                  <p className="text-sm text-blue-600">↗ +5 today</p>
                </div>
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center text-2xl">
                  💬
                </div>
              </div>
            </div>
          </div>

          {/* Charts Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">📈 Patient Adherence Trends</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={[
                    { month: 'Jan', adherence: 78 },
                    { month: 'Feb', adherence: 82 },
                    { month: 'Mar', adherence: 79 },
                    { month: 'Apr', adherence: 85 },
                    { month: 'May', adherence: 88 },
                    { month: 'Jun', adherence: 84 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Line type="monotone" dataKey="adherence" stroke="#3b82f6" strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">🏥 Department Distribution</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Cardiology', value: 35, color: '#3b82f6' },
                        { name: 'Diabetes', value: 28, color: '#10b981' },
                        { name: 'Oncology', value: 22, color: '#f59e0b' },
                        { name: 'Neurology', value: 15, color: '#ef4444' }
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({name, value}) => `${name}: ${value}%`}
                    >
                      {[
                        { name: 'Cardiology', value: 35, color: '#3b82f6' },
                        { name: 'Diabetes', value: 28, color: '#10b981' },
                        { name: 'Oncology', value: 22, color: '#f59e0b' },
                        { name: 'Neurology', value: 15, color: '#ef4444' }
                      ].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">🔔 Recent Activity</h3>
            <div className="space-y-3">
              {[
                { type: 'alert', message: 'Patient P003 missed medication dose', time: '5 mins ago', priority: 'high' },
                { type: 'success', message: 'Patient P001 completed weekly check-in', time: '15 mins ago', priority: 'low' },
                { type: 'info', message: 'New lab results available for P002', time: '1 hour ago', priority: 'medium' },
                { type: 'alert', message: 'Patient P005 reported side effects', time: '2 hours ago', priority: 'high' }
              ].map((activity, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-white/50 border border-gray-200 rounded-lg">
                  <div className="flex items-center gap-3">
                    <span className={`w-3 h-3 rounded-full ${
                      activity.priority === 'high' ? 'bg-red-500' :
                      activity.priority === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
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

      {/* Patient Adherence Tab */}
      {activeTab === 'adherence' && (
        <div className="space-y-6">
          <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">🏥 Patient Adherence Monitoring</h2>
            
            <div className="flex gap-4 mb-4">
              <input
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="Enter Patient ID (P001, P002, P003, P004, P005) or patient@medcare.com"
                className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={fetchAdherence}
                disabled={loading}
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-3 rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 font-medium transition-all"
              >
                {loading ? '🔄 Loading...' : '🔍 Search Patient'}
              </button>
            </div>
            
            {/* Quick Patient Buttons */}
            <div className="flex flex-wrap gap-2 mb-4">
              <p className="text-sm text-gray-600 w-full mb-2">Quick Access:</p>
              {['P001', 'P002', 'P003', 'P004', 'P005'].map(id => (
                <button
                  key={id}
                  onClick={() => {
                    setPatientId(id);
                    setTimeout(() => {
                      const input = document.querySelector('input[placeholder*="Patient ID"]');
                      if (input) {
                        input.value = id;
                        setPatientId(id);
                        fetchAdherence();
                      }
                    }, 100);
                  }}
                  className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 text-sm font-medium transition-all"
                >
                  {id}
                </button>
              ))}
            </div>
          </div>

          {adherenceData && (
            <div className="space-y-6">
              {/* Patient Info Card */}
              <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900">{adherenceData.patient_name}</h3>
                    <p className="text-gray-600">Patient ID: {adherenceData.patient_id} • Age: {adherenceData.age} • {adherenceData.condition}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-blue-600">
                      {adherenceData.adherence_score ? `${adherenceData.adherence_score}%` : 'N/A'}
                    </div>
                    <div className="text-sm text-gray-500">Overall Adherence</div>
                  </div>
                </div>
                
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all duration-500 ${
                      adherenceData.adherence_score >= 80 ? 'bg-green-500' :
                      adherenceData.adherence_score >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${adherenceData.adherence_score || 0}%` }}
                  ></div>
                </div>
              </div>

              {/* Charts Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Weekly Trend Chart */}
                <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                  <h4 className="text-lg font-semibold mb-4">📈 Weekly Adherence Trend</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={adherenceData.weekly_trend}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="week" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip formatter={(value) => [`${value}%`, 'Adherence']} />
                        <Line 
                          type="monotone" 
                          dataKey="adherence" 
                          stroke="#3b82f6" 
                          strokeWidth={3}
                          dot={{ fill: '#3b82f6', strokeWidth: 2, r: 6 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Medication Breakdown Chart */}
                <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                  <h4 className="text-lg font-semibold mb-4">💊 Medication Adherence</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={adherenceData.medication_breakdown}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="medication" angle={-45} textAnchor="end" height={80} />
                        <YAxis domain={[0, 100]} />
                        <Tooltip formatter={(value) => [`${value}%`, 'Adherence']} />
                        <Bar dataKey="adherence" fill="#10b981" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Stats Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">Total Medications</p>
                      <p className="text-2xl font-bold text-gray-900">{adherenceData.total_medications}</p>
                    </div>
                    <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center text-2xl">💊</div>
                  </div>
                </div>
                <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">Critical Alerts</p>
                      <p className="text-2xl font-bold text-gray-900">{adherenceData.critical_alerts}</p>
                    </div>
                    <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center text-2xl">🚨</div>
                  </div>
                </div>
                <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">Recent Logs</p>
                      <p className="text-2xl font-bold text-gray-900">{adherenceData.logs.length}</p>
                    </div>
                    <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center text-2xl">📋</div>
                  </div>
                </div>
              </div>

              {/* Recent Logs Table */}
              <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                <h4 className="text-lg font-semibold mb-4">📋 Recent Medication Logs</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50/50">
                      <tr>
                        <th className="px-4 py-3 text-left font-semibold">Medication</th>
                        <th className="px-4 py-3 text-left font-semibold">Due Time</th>
                        <th className="px-4 py-3 text-left font-semibold">Status</th>
                        <th className="px-4 py-3 text-left font-semibold">Logged At</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {adherenceData.logs.slice(0, 15).map((log, index) => (
                        <tr key={index} className="hover:bg-gray-50/50">
                          <td className="px-4 py-3 font-medium">{log.medication}</td>
                          <td className="px-4 py-3">{new Date(log.due_time).toLocaleString()}</td>
                          <td className="px-4 py-3">
                            <span className={`px-3 py-1 text-xs font-medium rounded-full ${
                              log.taken ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                            }`}>
                              {log.taken ? '✅ Taken' : '❌ Missed'}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            {log.logged_at ? new Date(log.logged_at).toLocaleString() : '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
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
              {/* Chat Notifications */}
              {notifications.length > 0 && (
                <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                  <h2 className="text-xl font-semibold mb-4 bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent">
                    🔔 Chat Notifications
                  </h2>
                  <div className="space-y-3">
                    {notifications.map((notification) => (
                      <div key={notification.id} className="bg-red-50 border border-red-200 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-medium text-red-900">{notification.message}</p>
                            <p className="text-sm text-red-700">{new Date(notification.timestamp).toLocaleString()}</p>
                          </div>
                          <button
                            onClick={() => {
                              // Mark as read and join session
                              fetch(`http://localhost:8000/chat/notifications/${notification.id}/read`, {
                                method: 'PUT',
                                headers: { 'Authorization': `Bearer ${token}` }
                              });
                              joinChatSession(notification.session_id, `session_${notification.session_id}`);
                            }}
                            className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition-all text-sm"
                          >
                            Respond
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Patient List */}
              <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  👥 Your Patients
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {patients.map((patient) => (
                    <div key={patient.id} className="bg-white/50 border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="font-semibold text-gray-900">{patient.name}</h3>
                        <span className={`w-3 h-3 rounded-full ${patient.online ? 'bg-green-500' : 'bg-gray-400'}`}></span>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{patient.condition}</p>
                      <p className="text-xs text-gray-500 mb-3">Last visit: {patient.last_visit}</p>
                      <button
                        onClick={() => createChatSession(patient.id)}
                        className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white py-2 px-4 rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all text-sm"
                      >
                        {patient.online ? '💬 Chat Now' : '📧 Send Message'}
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Active Chat Sessions */}
              {chatSessions.length > 0 && (
                <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
                  <h2 className="text-xl font-semibold mb-4 bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
                    💬 Active Conversations
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
                            <span className="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">
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
                      <h3 className="font-semibold">Patient Consultation</h3>
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
                    <div className="text-4xl mb-2">👩‍⚕️</div>
                    <p>Start a conversation with your patient</p>
                    <p className="text-sm mt-1">Provide medical guidance and support</p>
                  </div>
                ) : (
                  messages.map((message) => (
                    <div key={message.id} className={`flex ${message.sender_type === 'doctor' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                        message.sender_type === 'doctor' 
                          ? 'bg-blue-500 text-white' 
                          : message.is_ai_response
                          ? 'bg-purple-100 text-purple-900 border border-purple-200'
                          : 'bg-gray-100 text-gray-900'
                      }`}>
                        {message.is_ai_response && (
                          <div className="text-xs opacity-75 mb-1">🤖 AI Assistant Response</div>
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
                    placeholder="Type your medical advice or response... (Press Enter to send)"
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
                    '👨‍⚕️ You are now connected to provide medical consultation'
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

export default DoctorPortal;
