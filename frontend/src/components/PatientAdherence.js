import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Button, 
  Text, 
  Badge, 
  Progress, 
  Box, 
  VStack, 
  HStack, 
  Input, 
  Select, 
  useToast,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  FormControl,
  FormLabel,
  Textarea
} from '@chakra-ui/react';
import { 
  Plus, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  AlertTriangle, 
  CheckCircle,
  Mic,
  BarChart3
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const PatientAdherence = () => {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [medicationLog, setMedicationLog] = useState([]);
  const [adherenceData, setAdherenceData] = useState([]);
  const [isLogging, setIsLogging] = useState(false);
  const [voiceInput, setVoiceInput] = useState('');
  const toast = useToast();
  const { isOpen, onOpen, onClose } = useDisclosure();

  // Mock patient data
  const mockPatients = [
    { id: "P001", name: "John Smith", age: 65, medications: ["Metformin", "Lisinopril", "Atorvastatin"] },
    { id: "P002", name: "Sarah Johnson", age: 42, medications: ["Sertraline", "Bupropion"] },
    { id: "P003", name: "Michael Brown", age: 58, medications: ["Warfarin", "Metoprolol", "Furosemide"] },
    { id: "P004", name: "Emily Davis", age: 35, medications: ["Levothyroxine", "Iron Supplement"] },
    { id: "P005", name: "Robert Wilson", age: 71, medications: ["Donepezil", "Memantine", "Vitamin D"] }
  ];

  useEffect(() => {
    setPatients(mockPatients);
    if (mockPatients.length > 0) {
      setSelectedPatient(mockPatients[0]);
      generateAdherenceData(mockPatients[0].id);
    }
  }, []);

  const generateAdherenceData = (patientId) => {
    const data = [];
    for (let i = 29; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      data.push({
        date: date.toISOString().split('T')[0],
        adherence: Math.random() * 0.4 + 0.6, // 60-100% adherence
        medications: Math.floor(Math.random() * 3) + 1,
        total: 3
      });
    }
    setAdherenceData(data);
  };

  const handlePatientChange = (patientId) => {
    const patient = patients.find(p => p.id === patientId);
    setSelectedPatient(patient);
    generateAdherenceData(patientId);
  };

  const logMedication = async (formData) => {
    setIsLogging(true);
    
    // Simulate API call
    setTimeout(() => {
      const newLog = {
        id: Date.now(),
        patientId: selectedPatient.id,
        medication: formData.medication,
        dosage: formData.dosage,
        intakeTime: formData.intakeTime,
        method: formData.method,
        timestamp: new Date().toISOString()
      };
      
      setMedicationLog(prev => [newLog, ...prev]);
      
      // Simulate AI adherence analysis
      const adherenceScore = Math.random() * 0.4 + 0.6;
      const trend = adherenceScore > 0.8 ? "improving" : adherenceScore < 0.7 ? "declining" : "stable";
      
      toast({
        title: "Medication Logged",
        description: `Adherence: ${(adherenceScore * 100).toFixed(0)}% - Trend: ${trend}`,
        status: adherenceScore > 0.8 ? "success" : adherenceScore > 0.7 ? "warning" : "error",
        duration: 5000,
        isClosable: true,
      });
      
      setIsLogging(false);
      onClose();
    }, 1500);
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case "improving": return <TrendingUp color="green" />;
      case "declining": return <TrendingDown color="red" />;
      default: return <Minus color="gray" />;
    }
  };

  const getTrendColor = (trend) => {
    switch (trend) {
      case "improving": return "green";
      case "declining": return "red";
      default: return "gray";
    }
  };

  const getAdherenceColor = (score) => {
    if (score >= 0.9) return "green";
    if (score >= 0.8) return "blue";
    if (score >= 0.7) return "yellow";
    return "red";
  };

  return (
    <Box p={6} bg="gray.50" minH="100vh">
      <VStack spacing={6} align="stretch">
        {/* Header */}
        <Box textAlign="center" py={8}>
          <Text fontSize="3xl" fontWeight="bold" color="purple.600">
            AI Patient Adherence Tracker
          </Text>
          <Text fontSize="lg" color="gray.600">
            Monitor medication compliance with machine learning insights
          </Text>
        </Box>

        <HStack spacing={8} align="flex-start">
          {/* Left Panel - Patient Selection & Logging */}
          <Card p={6} flex={1} shadow="lg">
            <VStack spacing={6}>
              {/* Patient Selection */}
              <Box w="full">
                <Text fontSize="xl" fontWeight="semibold" mb={4}>
                  Select Patient
                </Text>
                <Select 
                  value={selectedPatient?.id || ''} 
                  onChange={(e) => handlePatientChange(e.target.value)}
                  size="lg"
                >
                  {patients.map(patient => (
                    <option key={patient.id} value={patient.id}>
                      {patient.name} (ID: {patient.id})
                    </option>
                  ))}
                </Select>
              </Box>

              {/* Patient Info */}
              {selectedPatient && (
                <Card p={4} w="full" variant="outline">
                  <VStack spacing={3} align="stretch">
                    <HStack justify="space-between">
                      <Text fontWeight="semibold">Name:</Text>
                      <Text>{selectedPatient.name}</Text>
                    </HStack>
                    <HStack justify="space-between">
                      <Text fontWeight="semibold">Age:</Text>
                      <Text>{selectedPatient.age} years</Text>
                    </HStack>
                    <HStack justify="space-between">
                      <Text fontWeight="semibold">Medications:</Text>
                      <Text>{selectedPatient.medications.length}</Text>
                    </HStack>
                    <VStack spacing={2} align="stretch">
                      {selectedPatient.medications.map((med, index) => (
                        <Badge key={index} colorScheme="blue" p={2}>
                          {med}
                        </Badge>
                      ))}
                    </VStack>
                  </VStack>
                </Card>
              )}

              {/* Log Medication Button */}
              <Button
                leftIcon={<Plus />}
                colorScheme="purple"
                size="lg"
                w="full"
                onClick={onOpen}
                isDisabled={!selectedPatient}
              >
                Log Medication Intake
              </Button>

              {/* Recent Logs */}
              <Box w="full">
                <Text fontSize="lg" fontWeight="semibold" mb={4}>
                  Recent Medication Logs
                </Text>
                <VStack spacing={3} align="stretch" maxH="300px" overflowY="auto">
                  {medicationLog.slice(0, 5).map(log => (
                    <Card key={log.id} p={3} variant="outline">
                      <HStack justify="space-between">
                        <VStack align="start" spacing={1}>
                          <Text fontWeight="semibold">{log.medication}</Text>
                          <Text fontSize="sm" color="gray.600">
                            {log.dosage} • {new Date(log.intakeTime).toLocaleString()}
                          </Text>
                        </VStack>
                        <Badge colorScheme="green">{log.method}</Badge>
                      </HStack>
                    </Card>
                  ))}
                  {medicationLog.length === 0 && (
                    <Text color="gray.500" textAlign="center">
                      No medication logs yet
                    </Text>
                  )}
                </VStack>
              </Box>
            </VStack>
          </Card>

          {/* Right Panel - Adherence Analytics */}
          <Card p={6} flex={1} shadow="lg">
            <VStack spacing={6}>
              <Text fontSize="xl" fontWeight="semibold">
                Adherence Analytics
              </Text>

              {/* Overall Adherence Score */}
              <Box w="full">
                <HStack justify="space-between" mb={2}>
                  <Text fontWeight="semibold">Overall Adherence:</Text>
                  <Text fontSize="xl" fontWeight="bold" color="purple.600">
                    {adherenceData.length > 0 ? 
                      (adherenceData.reduce((sum, d) => sum + d.adherence, 0) / adherenceData.length * 100).toFixed(0) + '%' 
                      : '0%'
                    }
                  </Text>
                </HStack>
                <Progress
                  value={adherenceData.length > 0 ? 
                    adherenceData.reduce((sum, d) => sum + d.adherence, 0) / adherenceData.length * 100 
                    : 0
                  }
                  colorScheme="purple"
                  size="lg"
                  borderRadius="full"
                />
              </Box>

              {/* Adherence Trend Chart */}
              <Box w="full" h="300px">
                <Text fontWeight="semibold" mb={4}>30-Day Adherence Trend</Text>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={adherenceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="adherence" 
                      stroke="#805AD5" 
                      strokeWidth={3}
                      dot={{ fill: '#805AD5', strokeWidth: 2, r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>

              {/* AI Insights */}
              <Box w="full">
                <Text fontWeight="semibold" mb={4}>AI Insights</Text>
                <VStack spacing={3} align="stretch">
                  <Card p={3} bg="green.50" borderColor="green.200">
                    <HStack spacing={3}>
                      <CheckCircle color="green" />
                      <VStack align="start" spacing={1}>
                        <Text fontWeight="semibold" color="green.700">
                          Adherence Trend: Improving
                        </Text>
                        <Text fontSize="sm" color="green.600">
                          Patient shows consistent medication intake over the past week
                        </Text>
                      </VStack>
                    </HStack>
                  </Card>
                  
                  <Card p={3} bg="yellow.50" borderColor="yellow.200">
                    <HStack spacing={3}>
                      <AlertTriangle color="orange" />
                      <VStack align="start" spacing={1}>
                        <Text fontWeight="semibold" color="orange.700">
                          Risk Factor Detected
                        </Text>
                        <Text fontSize="sm" color="orange.600">
                          Evening doses occasionally missed - consider reminder system
                        </Text>
                      </VStack>
                    </HStack>
                  </Card>
                </VStack>
              </Box>
            </VStack>
          </Card>
        </HStack>
      </VStack>

      {/* Log Medication Modal */}
      <Modal isOpen={isOpen} onClose={onClose} size="lg">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Log Medication Intake</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            <VStack spacing={4}>
              <FormControl>
                <FormLabel>Medication</FormLabel>
                <Select placeholder="Select medication">
                  {selectedPatient?.medications.map((med, index) => (
                    <option key={index} value={med}>{med}</option>
                  ))}
                </Select>
              </FormControl>
              
              <FormControl>
                <FormLabel>Dosage</FormLabel>
                <Input placeholder="e.g., 500mg" />
              </FormControl>
              
              <FormControl>
                <FormLabel>Intake Time</FormLabel>
                <Input type="datetime-local" defaultValue={new Date().toISOString().slice(0, 16)} />
              </FormControl>
              
              <FormControl>
                <FormLabel>Input Method</FormLabel>
                <HStack spacing={4}>
                  <Button leftIcon={<BarChart3 />} variant="outline" flex={1}>
                    Manual
                  </Button>
                  <Button leftIcon={<BarChart3 />} variant="outline" flex={1}>
                    Barcode
                  </Button>
                  <Button leftIcon={<Mic />} variant="outline" flex={1}>
                    Voice
                  </Button>
                </HStack>
              </FormControl>
              
              <FormControl>
                <FormLabel>Notes (Optional)</FormLabel>
                <Textarea placeholder="Any additional notes..." />
              </FormControl>
              
              <HStack spacing={4} w="full">
                <Button variant="outline" flex={1} onClick={onClose}>
                  Cancel
                </Button>
                <Button 
                  colorScheme="purple" 
                  flex={1} 
                  onClick={() => logMedication({
                    medication: "Metformin",
                    dosage: "500mg",
                    intakeTime: new Date().toISOString(),
                    method: "Manual"
                  })}
                  isLoading={isLogging}
                >
                  Log Intake
                </Button>
              </HStack>
            </VStack>
          </ModalBody>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default PatientAdherence;
