// Frontend API Authentication Service
'use client';

import { useState, useEffect, createContext, useContext } from 'react';

// API Keys for different components
export const API_KEYS = {
  MAP_PROCESSING: 'geoai_map_proc_key_2024_v1_secure',
  ADAPTIVE_FUSION: 'geoai_fusion_key_2024_v1_secure', 
  SATELLITE_ANALYSIS: 'geoai_sat_analysis_key_2024_v1_secure',
  VECTOR_CONVERSION: 'geoai_vector_key_2024_v1_secure',
  GRAPH_VISUALIZATION: 'geoai_graph_viz_key_2024_v1_secure',
  LIVE_TRAINING: 'geoai_live_train_key_2024_v1_secure'
};

// Authentication Context
const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [authenticatedComponents, setAuthenticatedComponents] = useState(new Set());
  const [authStatus, setAuthStatus] = useState({});

  const authenticateComponent = async (componentName, apiKey) => {
    try {
      const response = await fetch('http://localhost:8002/api/v1/auth/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey
        },
        body: JSON.stringify({
          component: componentName,
          apiKey: apiKey,
          timestamp: new Date().toISOString()
        })
      });

      if (response.ok) {
        const result = await response.json();
        setAuthenticatedComponents(prev => new Set(Array.from(prev).concat(componentName)));
        setAuthStatus(prev => ({
          ...prev,
          [componentName]: { 
            status: 'authenticated', 
            timestamp: new Date().toISOString(),
            permissions: result.permissions || []
          }
        }));
        return true;
      } else {
        setAuthStatus(prev => ({
          ...prev,
          [componentName]: { 
            status: 'failed', 
            timestamp: new Date().toISOString(),
            error: 'Authentication failed'
          }
        }));
        return false;
      }
    } catch (error) {
      setAuthStatus(prev => ({
        ...prev,
        [componentName]: { 
          status: 'error', 
          timestamp: new Date().toISOString(),
          error: error.message
        }
      }));
      return false;
    }
  };

  const isComponentAuthenticated = (componentName) => {
    return authenticatedComponents.has(componentName);
  };

  return (
    <AuthContext.Provider value={{
      authenticateComponent,
      isComponentAuthenticated,
      authStatus,
      authenticatedComponents: Array.from(authenticatedComponents)
    }}>
      {children}
    </AuthContext.Provider>
  );
};

// HOC for component authentication
export const withAuth = (WrappedComponent: React.ComponentType<any>, componentName: string, requiredApiKey: string) => {
  return function AuthenticatedComponent(props: any) {
    const { authenticateComponent, isComponentAuthenticated, authStatus } = useAuth();
    const [isAuthenticating, setIsAuthenticating] = useState(false);

    useEffect(() => {
      if (!isComponentAuthenticated(componentName)) {
        setIsAuthenticating(true);
        authenticateComponent(componentName, requiredApiKey)
          .finally(() => setIsAuthenticating(false));
      }
    }, [componentName, requiredApiKey, authenticateComponent, isComponentAuthenticated]);

    if (isAuthenticating) {
      return (
        <div className="flex items-center justify-center p-8 bg-gradient-to-r from-blue-900/20 to-purple-900/20 rounded-lg border border-blue-500/30">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400 mx-auto mb-4"></div>
            <p className="text-blue-300">Authenticating {componentName}...</p>
          </div>
        </div>
      );
    }

    if (!isComponentAuthenticated(componentName)) {
      const status = authStatus[componentName];
      return (
        <div className="flex items-center justify-center p-8 bg-gradient-to-r from-red-900/20 to-orange-900/20 rounded-lg border border-red-500/30">
          <div className="text-center">
            <div className="text-red-400 text-2xl mb-4">🔒</div>
            <p className="text-red-300">Authentication Failed for {componentName}</p>
            {status?.error && <p className="text-red-400 text-sm mt-2">{status.error}</p>}
          </div>
        </div>
      );
    }

    return <WrappedComponent {...props} />;
  };
};