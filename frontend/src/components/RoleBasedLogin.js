import React, { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Input,
  Button,
  FormControl,
  FormLabel,
  Alert,
  AlertIcon,
  Select,
  Card,
  CardBody,
  useColorModeValue,
  Flex,
  Icon,
  Badge
} from '@chakra-ui/react';
import { useAuth } from '../context/AuthContext';

const RoleBasedLogin = ({ onBack }) => {
  const [selectedRole, setSelectedRole] = useState('');
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();

  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  const roles = [
    {
      id: 'manager',
      name: 'Hospital Manager',
      icon: '👔',
      color: 'blue',
      description: 'Access to inventory, trials, file management'
    },
    {
      id: 'doctor',
      name: 'Doctor',
      icon: '👨‍⚕️',
      color: 'green',
      description: 'Patient care, prescriptions, AI diagnosis'
    },
    {
      id: 'patient',
      name: 'Patient',
      icon: '🏥',
      color: 'purple',
      description: 'Health monitoring, chat with doctors'
    }
  ];

  const demoCredentials = {
    manager: { username: 'manager@medcare.com', password: 'manager123' },
    doctor: { username: 'doctor@medcare.com', password: 'doctor123' },
    patient: { username: 'patient@medcare.com', password: 'patient123' }
  };

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
    setFormData(demoCredentials[role]);
    setError('');
  };

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const result = await login(formData.username, formData.password);
    
    if (result.success) {
      // Login successful - App.js will handle navigation
    } else {
      setError(result.error);
    }
    
    setLoading(false);
  };

  return (
    <Box minH="100vh" bg="gray.50" py={12} px={4}>
      <VStack spacing={8} maxW="6xl" mx="auto">
        <VStack spacing={4} textAlign="center">
          <Heading size="2xl" color="blue.600">
            MedCare Hospital Management
          </Heading>
          <Text fontSize="lg" color="gray.600">
            Select your role to access the appropriate portal
          </Text>
        </VStack>

        {!selectedRole ? (
          <VStack spacing={6} w="full" maxW="4xl">
            <Heading size="lg" color="gray.700">
              Choose Your Portal
            </Heading>
            
            <HStack spacing={6} w="full" justify="center" flexWrap="wrap">
              {roles.map((role) => (
                <Card
                  key={role.id}
                  maxW="sm"
                  cursor="pointer"
                  onClick={() => handleRoleSelect(role.id)}
                  _hover={{ transform: 'translateY(-4px)', shadow: 'xl' }}
                  transition="all 0.2s"
                  border="2px"
                  borderColor="transparent"
                  _focus={{ borderColor: `${role.color}.500` }}
                >
                  <CardBody textAlign="center" p={8}>
                    <VStack spacing={4}>
                      <Box
                        p={4}
                        borderRadius="full"
                        bg={`${role.color}.100`}
                        color={`${role.color}.600`}
                      >
                        <Text fontSize="2xl">{role.icon}</Text>
                      </Box>
                      
                      <VStack spacing={2}>
                        <Heading size="md" color="gray.700">
                          {role.name}
                        </Heading>
                        <Text fontSize="sm" color="gray.500" textAlign="center">
                          {role.description}
                        </Text>
                        <Badge colorScheme={role.color} variant="subtle">
                          Click to Login
                        </Badge>
                      </VStack>
                    </VStack>
                  </CardBody>
                </Card>
              ))}
            </HStack>

            <Button variant="ghost" onClick={onBack} mt={4}>
              ← Back to Landing Page
            </Button>
          </VStack>
        ) : (
          <Card maxW="md" w="full" bg={bgColor} borderColor={borderColor}>
            <CardBody p={8}>
              <VStack spacing={6}>
                <VStack spacing={2} textAlign="center">
                  <Box
                    p={3}
                    borderRadius="full"
                    bg={`${roles.find(r => r.id === selectedRole)?.color}.100`}
                    color={`${roles.find(r => r.id === selectedRole)?.color}.600`}
                  >
                    <Icon as={roles.find(r => r.id === selectedRole)?.icon} boxSize={6} />
                  </Box>
                  <Heading size="lg">
                    {roles.find(r => r.id === selectedRole)?.name} Login
                  </Heading>
                  <Text fontSize="sm" color="gray.500">
                    Demo credentials are pre-filled
                  </Text>
                </VStack>

                {error && (
                  <Alert status="error">
                    <AlertIcon />
                    {error}
                  </Alert>
                )}

                <Box as="form" onSubmit={handleLogin} w="full">
                  <VStack spacing={4}>
                    <FormControl>
                      <FormLabel>Username</FormLabel>
                      <Input
                        name="username"
                        value={formData.username}
                        onChange={handleInputChange}
                        placeholder="Enter username"
                        required
                      />
                    </FormControl>

                    <FormControl>
                      <FormLabel>Password</FormLabel>
                      <Input
                        name="password"
                        type="password"
                        value={formData.password}
                        onChange={handleInputChange}
                        placeholder="Enter password"
                        required
                      />
                    </FormControl>

                    <VStack spacing={3} w="full">
                      <Button
                        type="submit"
                        colorScheme={roles.find(r => r.id === selectedRole)?.color}
                        size="lg"
                        w="full"
                        isLoading={loading}
                        loadingText="Logging in..."
                      >
                        Login as {roles.find(r => r.id === selectedRole)?.name}
                      </Button>

                      <Button
                        variant="ghost"
                        onClick={() => setSelectedRole('')}
                        size="sm"
                      >
                        ← Choose Different Role
                      </Button>
                    </VStack>
                  </VStack>
                </Box>

                <Box p={4} bg="gray.50" borderRadius="md" w="full">
                  <Text fontSize="xs" color="gray.600" textAlign="center">
                    <strong>Demo Credentials:</strong><br />
                    Username: {demoCredentials[selectedRole]?.username}<br />
                    Password: {demoCredentials[selectedRole]?.password}
                  </Text>
                </Box>
              </VStack>
            </CardBody>
          </Card>
        )}
      </VStack>
    </Box>
  );
};

export default RoleBasedLogin;
