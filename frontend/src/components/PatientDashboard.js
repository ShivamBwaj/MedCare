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
  AlertIcon,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td
} from '@chakra-ui/react';
import { useAuth } from '../context/AuthContext';

const PatientDashboard = ({ onLogout }) => {
  const [stats, setStats] = useState({});
  const [doctors, setDoctors] = useState([]);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [symptoms, setSymptoms] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [selectedImageFile, setSelectedImageFile] = useState(null);
  const [skinPrediction, setSkinPrediction] = useState(null);
  const [prescriptions, setPrescriptions] = useState([]);
  const { user, token } = useAuth();
  const toast = useToast();

  const bgColor = useColorModeValue('gray.50', 'gray.900');
  const cardBg = useColorModeValue('white', 'gray.800');

  useEffect(() => {
    fetchDashboardStats();
    fetchDoctors();
    fetchPrescriptions();
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

  const fetchDoctors = async () => {
    try {
      const response = await fetch('http://localhost:8000/users/doctors', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setDoctors(data.doctors);
      }
    } catch (error) {
      console.error('Error fetching doctors:', error);
    }
  };

  const fetchPrescriptions = async () => {
    try {
      const response = await fetch(`http://localhost:8000/prescriptions/patient/${user.id}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setPrescriptions(data.prescriptions);
      }
    } catch (error) {
      console.error('Error fetching prescriptions:', error);
    }
  };

  const fetchChatMessages = async (doctorId) => {
    try {
      const response = await fetch(`http://localhost:8000/chat/messages/${doctorId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setChatMessages(data.messages);
        setSelectedDoctor(data.partner);
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  const sendMessage = async () => {
    if (!newMessage.trim() || !selectedDoctor) return;

    try {
      const response = await fetch('http://localhost:8000/chat/send', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          receiver_id: selectedDoctor.id,
          message: newMessage
        })
      });

      if (response.ok) {
        setNewMessage('');
        fetchChatMessages(selectedDoctor.id);
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



  const SymptomPrediction = () => (
    <Card bg={cardBg}>
      <CardHeader>
        <Heading size="md">🧠 Symptom-Based Disease Prediction</Heading>
      </CardHeader>
      <CardBody>
        <VStack spacing={4} align="stretch">
          <FormControl>
            <FormLabel>Describe Your Symptoms</FormLabel>
            <Textarea
              placeholder="Enter your symptoms separated by commas (e.g., fever, headache, cough)"
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
              rows={4}
            />
            <Text fontSize="sm" color="gray.500" mt={2}>
              Be as specific as possible for better predictions
            </Text>
          </FormControl>

          <Button colorScheme="purple" onClick={predictDisease} size="lg">
            Get AI Prediction
          </Button>

          {predictionResult && (
            <Alert status="info">
              <AlertIcon />
              <VStack align="start" spacing={3} w="full">
                <Text fontWeight="bold">AI Prediction Results:</Text>
                {predictionResult.predicted_diseases.map((disease, index) => (
                  <Box key={index} p={3} border="1px" borderColor="gray.200" borderRadius="md" w="full">
                    <HStack justify="space-between">
                      <Text fontWeight="bold">{disease.disease}</Text>
                      <Badge colorScheme={disease.confidence > 70 ? 'green' : 'yellow'}>
                        {disease.confidence}% confidence
                      </Badge>
                    </HStack>
                  </Box>
                ))}
                <Box mt={3}>
                  <Text fontWeight="bold" color="red.600">Recommendations:</Text>
                  {predictionResult.recommendations.map((rec, index) => (
                    <Text key={index} fontSize="sm" mt={1}>• {rec}</Text>
                  ))}
                </Box>
              </VStack>
            </Alert>
          )}
        </VStack>
      </CardBody>
    </Card>
  );

  const SkinDiseaseDetection = () => (
    <Card bg={cardBg}>
      <CardHeader>
        <Heading size="md">🔬 Nail Disease Detection</Heading>
      </CardHeader>
      <CardBody>
        <VStack spacing={4} align="stretch">
          <FormControl>
            <FormLabel>Upload Image of Affected Area</FormLabel>
            <Input
              type="file"
              accept="image/*"
              onChange={(e) => setSelectedImageFile(e.target.files[0])}
            />
            <Text fontSize="sm" color="gray.500" mt={2}>
              Upload clear photos of nail concerns
            </Text>
          </FormControl>

          <Button colorScheme="teal" onClick={analyzeSkinImage} size="lg">
            Analyze Image
          </Button>

          {skinPrediction && (
            <Alert status="info">
              <AlertIcon />
              <VStack align="start" spacing={3} w="full">
                <Text fontWeight="bold">AI Analysis Results:</Text>
                <Grid templateColumns="1fr 1fr" gap={4} w="full">
                  <Box>
                    <Text fontSize="sm"><strong>Condition:</strong> {skinPrediction.predicted_condition}</Text>
                    <Text fontSize="sm"><strong>Confidence:</strong> {skinPrediction.confidence_score}%</Text>
                    <Text fontSize="sm"><strong>Severity:</strong> {skinPrediction.severity}</Text>
                  </Box>
                </Grid>
                <Box>
                  <Text fontWeight="bold" color="red.600">Recommendations:</Text>
                  {skinPrediction.recommendations.map((rec, index) => (
                    <Text key={index} fontSize="sm" mt={1}>• {rec}</Text>
                  ))}
                </Box>
              </VStack>
            </Alert>
          )}
        </VStack>
      </CardBody>
    </Card>
  );

  const PrescriptionManager = () => (
    <Card bg={cardBg}>
      <CardHeader>
        <Heading size="md">💊 My Prescriptions</Heading>
      </CardHeader>
      <CardBody>
        {prescriptions.length > 0 ? (
          <Table variant="simple">
            <Thead>
              <Tr>
                <Th>Medication</Th>
                <Th>Dosage</Th>
                <Th>Frequency</Th>
                <Th>Doctor</Th>
                <Th>Date</Th>
              </Tr>
            </Thead>
            <Tbody>
              {prescriptions.map((prescription) => (
                <Tr key={prescription.id}>
                  <Td fontWeight="bold">{prescription.medication_name}</Td>
                  <Td>{prescription.dosage}</Td>
                  <Td>{prescription.frequency}</Td>
                  <Td>{prescription.doctor_name}</Td>
                  <Td>{new Date(prescription.created_at).toLocaleDateString()}</Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        ) : (
          <Text color="gray.500" textAlign="center" py={8}>
            No prescriptions found
          </Text>
        )}
      </CardBody>
    </Card>
  );

  const ChatWithDoctor = () => (
    <Grid templateColumns="1fr 2fr" gap={6} h="600px">
      <Card bg={cardBg}>
        <CardHeader>
          <Heading size="md">Available Doctors</Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={3} align="stretch">
            {doctors.map((doctor) => (
              <Box
                key={doctor.id}
                p={3}
                border="1px"
                borderColor="gray.200"
                borderRadius="md"
                cursor="pointer"
                onClick={() => fetchChatMessages(doctor.id)}
                _hover={{ bg: 'gray.50' }}
                bg={selectedDoctor?.id === doctor.id ? 'green.50' : 'white'}
              >
                <HStack>
                  <Avatar size="sm" name={doctor.name} />
                  <VStack align="start" spacing={0} flex={1}>
                    <Text fontWeight="bold">{doctor.name}</Text>
                    <Text fontSize="sm" color="gray.500">
                      ID: {doctor.employee_id}
                    </Text>
                  </VStack>
                  <Badge colorScheme="green" size="sm">Doctor</Badge>
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
              {selectedDoctor ? `Chat with ${selectedDoctor.name}` : 'Select a doctor'}
            </Heading>
          </HStack>
        </CardHeader>
        <CardBody>
          {selectedDoctor ? (
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
                        bg={msg.is_own_message ? 'purple.500' : 'gray.100'}
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
                <Button colorScheme="purple" onClick={sendMessage}>
                  Send
                </Button>
              </HStack>
            </VStack>
          ) : (
            <Flex align="center" justify="center" h="full">
              <Text color="gray.500">Select a doctor to start chatting</Text>
            </Flex>
          )}
        </CardBody>
      </Card>
    </Grid>
  );

  return (
    <Box minH="100vh" bg={bgColor}>
      <Box bg="white" shadow="sm" px={6} py={4}>
        <HStack justify="space-between">
          <VStack align="start" spacing={1}>
            <Heading size="lg" color="purple.600">
              Patient Portal
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
                <StatLabel>Active Prescriptions</StatLabel>
                <StatNumber>{prescriptions.length}</StatNumber>
                <StatHelpText>Current medications</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card bg={cardBg}>
            <CardBody>
              <Stat>
                <StatLabel>Health Score</StatLabel>
                <StatNumber color="green.500">{stats.health_score || 85}%</StatNumber>
                <StatHelpText>Overall wellness</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card bg={cardBg}>
            <CardBody>
              <Stat>
                <StatLabel>Upcoming Appointments</StatLabel>
                <StatNumber color="blue.500">{stats.upcoming_appointments || 0}</StatNumber>
                <StatHelpText>This week</StatHelpText>
              </Stat>
            </CardBody>
          </Card>
        </Grid>

        {/* Main Tabs */}
        <Tabs variant="enclosed" colorScheme="purple">
          <TabList>
            <Tab>🧠 Symptom Checker</Tab>
            <Tab>📸 Nail Analysis</Tab>
            <Tab>💊 My Prescriptions</Tab>
            <Tab>💬 Chat with Doctor</Tab>
          </TabList>

          <TabPanels>
            <TabPanel>
              <SymptomPrediction />
            </TabPanel>

            <TabPanel>
              <SkinDiseaseDetection />
            </TabPanel>

            <TabPanel>
              <PrescriptionManager />
            </TabPanel>

            <TabPanel>
              <ChatWithDoctor />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
    </Box>
  );
};

export default PatientDashboard;
