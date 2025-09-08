import React from 'react';
import { useAuth } from '../context/AuthContext';
import ManagerDashboard from './ManagerDashboard';

const ManagerPortal = () => {
  const { logout } = useAuth();

  return <ManagerDashboard onLogout={logout} />;
};

export default ManagerPortal;
