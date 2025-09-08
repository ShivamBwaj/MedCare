import React from 'react';
import { ChakraProvider } from '@chakra-ui/react';
import { AuthProvider, useAuth } from './context/AuthContext';
import LandingPage from './components/LandingPage';
import RoleBasedLogin from './components/RoleBasedLogin';
import ManagerPortal from './components/ManagerPortal';
import DoctorPortal from './components/DoctorPortal';
import PatientPortal from './components/PatientPortal';

function AppContent() {
  const { user, loading, logout } = useAuth();
  const [showLogin, setShowLogin] = React.useState(false);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  if (!user && !showLogin) {
    return <LandingPage onLogin={() => setShowLogin(true)} />;
  }

  if (!user && showLogin) {
    return <RoleBasedLogin onBack={() => setShowLogin(false)} />;
  }

  // Role-based portal rendering
  const renderPortal = () => {
    switch (user.role) {
      case 'manager':
        return <ManagerPortal />;
      case 'doctor':
        return <DoctorPortal />;
      case 'patient':
        return <PatientPortal />;
      default:
        return <LandingPage onLogin={() => setShowLogin(true)} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">MedCare</h1>
              <span className="ml-4 px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full">
                {user.role.charAt(0).toUpperCase() + user.role.slice(1)}
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-gray-700">{user.email}</span>
              <button
                onClick={logout}
                className="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main>
        {renderPortal()}
      </main>
    </div>
  );
}

function App() {
  return (
    <ChakraProvider>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </ChakraProvider>
  );
}

export default App;
