import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchApiStatus = async () => {
  try {
    const response = await api.get('/');
    return response.data;
  } catch (error) {
    console.error('API status check failed:', error);
    throw error;
  }
};

export const fetchHealthCheck = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

export const startProcessing = async (processingData) => {
  const response = await api.post('/api/process', processingData);
  return response.data;
};

export const getJobStatus = async (jobId) => {
  const response = await api.get(`/api/jobs/${jobId}`);
  return response.data;
};

export const listJobs = async () => {
  const response = await api.get('/api/jobs');
  return response.data;
};

export const getResults = async (jobId) => {
  const response = await api.get(`/api/results/${jobId}`);
  return response.data;
};

export const listStates = async () => {
  const response = await api.get('/api/states');
  return response.data;
};

export const getSystemStats = async () => {
  const response = await api.get('/api/stats');
  return response.data;
};

// Websocket connection for real-time updates
export const createWebSocketConnection = (jobId, onMessage, onError) => {
  const wsUrl = `${API_URL.replace('http', 'ws')}/ws/${jobId}`;
  const socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log('WebSocket connection established');
    // Send ping every 30s to keep connection alive
    const pingInterval = setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send('ping');
      } else {
        clearInterval(pingInterval);
      }
    }, 30000);
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  };

  socket.onerror = (error) => {
    console.error('WebSocket error:', error);
    if (onError) {
      onError(error);
    }
  };

  socket.onclose = () => {
    console.log('WebSocket connection closed');
  };

  return {
    close: () => {
      socket.close();
    },
  };
};

export default api;