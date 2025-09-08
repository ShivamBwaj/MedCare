import React, { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  Card,
  CardBody,
  CardHeader,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  useColorModeValue,
  Badge,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Grid,
  GridItem,
  Input,
  Textarea,
  FormControl,
  FormLabel,
  Select,
  useToast,
  Avatar,
  Flex,
  Divider,
  Alert,
  AlertIcon
} from '@chakra-ui/react';
import { useAuth } from '../context/AuthContext';
import PatientAdherence from './PatientAdherence';

const DoctorDashboard = ({ onLogout }) => {
  const [stats, setStats] = useState({});
  const [patients, setPatients] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [symptoms, setSymptoms] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [selectedImageFile, setSelectedImageFile] = useState(null);
  const [skinPrediction, setSkinPrediction] = useState(null);
  const { user, token } = useAuth();
  const toast = useToast();

  const bgColor = useColorModeValue('gray.50', 'gray.900');
  const cardBg = useColorModeValue('white', 'gray.800');

  useEffect(() => {
    fetchDashboardStats();
    fetchPatients();
    fetchConversations();
  }, []);

  const fetchDashboardStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/dashboard/stats', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchPatients = async () => {
    try {
      const response = await fetch('http://localhost:8000/users/patients', {
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

  const fetchConversations = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat/conversations', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setConversations(data.conversations);
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
    }
  };

  const fetchChatMessages = async (patientId) => {
    try {
      const response = await fetch(`http://localhost:8000/chat/messages/${patientId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setChatMessages(data.messages);
        setSelectedPatient(data.partner);
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  const sendMessage = async () => {
    if (!newMessage.trim() || !selectedPatient) return;

    try {
      const response = await fetch('http://localhost:8000/chat/send', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          receiver_id: selectedPatient.id,
          message: newMessage
        })
      });

      if (response.ok) {
        setNewMessage('');
        fetchChatMessages(selectedPatient.id);
        toast({
          title: 'Message sent',
          status: 'success',
          duration: 2000
        });
      }
    } catch (error) {
      toast({
        title: 'Failed to send message',
        status: 'error',
        duration: 3000
      });
    }
  };

  const predictDisease = async () => {
    if (!symptoms.trim()) {
      toast({
        title: 'Please enter symptoms',
        status: 'warning',
        duration: 3000
      });
      return;
    }

    try {
      const symptomsList = symptoms.split(',').map(s => s.trim());
      const response = await fetch('http://localhost:8000/predict/symptoms', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ symptoms: symptomsList })
      });

      if (response.ok) {
        const data = await response.json();
        setPredictionResult(data);
      }
    } catch (error) {
      toast({
        title: 'Prediction failed',
        status: 'error',
        duration: 3000
      });
    }
  };

  const analyzeSkinImage = async () => {
    if (!selectedImageFile) {
      toast({
        title: 'Please select an image',
        status: 'warning',
        duration: 3000
      });
      return;
    }

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64Image = e.target.result;
        
        const response = await fetch('http://localhost:8000/predict/skin', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ image_data: base64Image })
        });

        if (response.ok) {
          const data = await response.json();
          setSkinPrediction(data);
        }
      };
      reader.readAsDataURL(selectedImageFile);
    } catch (error) {
      toast({
        title: 'Analysis failed',
        status: 'error',
        duration: 3000
      });
    }
  };

  const PatientChat = () => (
    <Grid templateColumns="1fr 2fr" gap={6} h="600px">
      <Card bg={cardBg}>
        <CardHeader>
          <Heading size="md">Patient Conversations</Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={3} align="stretch">
            {patients.map((patient) => (
              <Box
                key={patient.id}
                p={3}
                border="1px"
                borderColor="gray.200"
                borderRadius="md"
                cursor="pointer"
                onClick={() => fetchChatMessages(patient.id)}
                _hover={{ bg: 'gray.50' }}
                bg={selectedPatient?.id === patient.id ? 'blue.50' : 'white'}
              >
                <HStack>
                  <Avatar size="sm" name={patient.name} />
                  <VStack align="start" spacing={0} flex={1}>
                    <Text fontWeight="bold">{patient.name}</Text>
                    <Text fontSize="sm" color="gray.500">
                      ID: {patient.patient_id}
                    </Text>
                  </VStack>
                  <Badge colorScheme="green" size="sm">Patient</Badge>
                </HStack>
              </Box>
            ))}
          </VStack>
        </CardBody>
      </Card>

      <Card bg={cardBg}>
        <CardHeader>
          <HStack justify="space-between">
            <Heading size="md">
              {selectedPatient ? `Chat with ${selectedPatient.name}` : 'Select a patient'}
            </Heading>
          </HStack>
        </CardHeader>
        <CardBody>
          {selectedPatient ? (
            <VStack spacing={4} h="full">
              <Box
                flex={1}
                w="full"
                border="1px"
                borderColor="gray.200"
                borderRadius="md"
                p={4}
                overflowY="auto"
                maxH="400px"
              >
                <VStack spacing={3} align="stretch">
                  {chatMessages.map((msg) => (
                    <Box
                      key={msg.id}
                      alignSelf={msg.is_own_message ? 'flex-end' : 'flex-start'}
                      maxW="70%"
                    >
                      <Box
                        p={3}
                        borderRadius="lg"
                        bg={msg.is_own_message ? 'blue.500' : 'gray.100'}
                        color={msg.is_own_message ? 'white' : 'black'}
                      >
                        <Text>{msg.message}</Text>
                        <Text fontSize="xs" opacity={0.8} mt={1}>
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </Text>
                      </Box>
                    </Box>
                  ))}
                </VStack>
              </Box>

              <HStack w="full">
                <Input
                  placeholder="Type your message..."
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                />
                <Button colorScheme="blue" onClick={sendMessage}>
                  Send
                </Button>
              </HStack>
            </VStack>
          ) : (
            <Flex align="center" justify="center" h="full">
              <Text color="gray.500">Select a patient to start chatting</Text>
            </Flex>
          )}
        </CardBody>
      </Card>
    </Grid>
  );

  const AIDiseasePrediction = () => (
    <Grid templateColumns="1fr 1fr" gap={6}>
      <Card bg={cardBg}>
        <CardHeader>
          <Heading size="md">🧠 Symptom-Based Diagnosis</Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={4} align="stretch">
            <FormControl>
              <FormLabel>Patient Symptoms</FormLabel>
              <Textarea
                placeholder="Enter symptoms separated by commas (e.g., fever, headache, cough)"
                value={symptoms}
                onChange={(e) => setSymptoms(e.target.value)}
                rows={4}
              />
            </FormControl>

            <Button colorScheme="green" onClick={predictDisease}>
              Analyze Symptoms
            </Button>

            {predictionResult && (
              <Alert status="info">
                <AlertIcon />
                <VStack align="start" spacing={2}>
                  <Text fontWeight="bold">Prediction Results:</Text>
                  {predictionResult.predicted_diseases.map((disease, index) => (
                    <Text key={index} fontSize="sm">
                      {disease.disease} - {disease.confidence}% confidence
                    </Text>
                  ))}
                </VStack>
              </Alert>
            )}
          </VStack>
        </CardBody>
      </Card>

      <Card bg={cardBg}>
        <CardHeader>
          <Heading size="md">🔬 Skin/Nail Disease Detection</Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={4} align="stretch">
            <FormControl>
              <FormLabel>Upload Patient Image</FormLabel>
              <Input
                type="file"
                accept="image/*"
                onChange={(e) => setSelectedImageFile(e.target.files[0])}
              />
            </FormControl>

            <Button colorScheme="purple" onClick={analyzeSkinImage}>
              Analyze Image
            </Button>

            {skinPrediction && (
              <Alert status="info">
                <AlertIcon />
                <VStack align="start" spacing={2}>
                  <Text fontWeight="bold">Analysis Results:</Text>
                  <Text fontSize="sm">
                    Condition: {skinPrediction.predicted_condition}
                  </Text>
                  <Text fontSize="sm">
                    Confidence: {skinPrediction.confidence_score}%
                  </Text>
                  <Text fontSize="sm">
                    Severity: {skinPrediction.severity}
                  </Text>
                </VStack>
              </Alert>
            )}
          </VStack>
        </CardBody>
      </Card>
    </Grid>
  );

  return (
    <Box minH="100vh" bg={bgColor}>
      <Box bg="white" shadow="sm" px={6} py={4}>
        <HStack justify="space-between">
          <VStack align="start" spacing={1}>
            <Heading size="lg" color="green.600">
              Doctor Portal
            </Heading>
            <Text color="gray.600">Welcome, {user?.full_name}</Text>
          </VStack>
          <Button colorScheme="red" variant="outline" onClick={onLogout}>
            Logout
          </Button>
        </HStack>
      </Box>

      <Box p={6}>
        {/* Stats Overview */}
        <Grid templateColumns="repeat(auto-fit, minmax(200px, 1fr))" gap={6} mb={8}>
          <Card bg={cardBg}>
            <CardBody>
              <Stat>
                <StatLabel>Total Patients</StatLabel>
                <StatNumber>{patients.length}</StatNumber>
                <StatHelpText>Under your care</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card bg={cardBg}>
            <CardBody>
              <Stat>
                <StatLabel>Unread Messages</StatLabel>
                <StatNumber color="blue.500">
                  {conversations.reduce((sum, conv) => sum + conv.unread_count, 0)}
                </StatNumber>
                <StatHelpText>From patients</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card bg={cardBg}>
            <CardBody>
              <Stat>
                <StatLabel>Pending Consultations</StatLabel>
                <StatNumber color="orange.500">{stats.pending_consultations || 0}</StatNumber>
                <StatHelpText>Today's schedule</StatHelpText>
              </Stat>
            </CardBody>
          </Card>
        </Grid>

        {/* Main Tabs */}
        <Tabs variant="enclosed" colorScheme="green">
          <TabList>
            <Tab>📊 Patient Adherence</Tab>
            <Tab>💬 Patient Chat</Tab>
            <Tab>🧠 AI Disease Prediction</Tab>
          </TabList>

          <TabPanels>
            <TabPanel>
              <PatientAdherence />
            </TabPanel>

            <TabPanel>
              <PatientChat />
            </TabPanel>

            <TabPanel>
              <AIDiseasePrediction />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
    </Box>
  );
};

export default DoctorDashboard;
