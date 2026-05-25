import axios from 'axios';

const getApiUrl = () => {
  try {
    if (typeof process !== 'undefined' && process.env.REACT_APP_API_URL) {
      return process.env.REACT_APP_API_URL;
    }
  } catch (e) {}
  try {
    if (import.meta && import.meta.env && import.meta.env.VITE_API_URL) {
      return import.meta.env.VITE_API_URL;
    }
  } catch (e) {}
  return 'http://127.0.0.1:5000';
};

const client = axios.create({
  baseURL: getApiUrl(),
  timeout: 30000,
});

client.interceptors.request.use((config) => {
  config.headers['Content-Type'] = 'application/json';
  return config;
});

client.interceptors.response.use(
  (response) => response,
  (error) => {
    let message = error.message;
    if (error.response && error.response.data && error.response.data.message) {
      message = error.response.data.message;
    } else if (error.response && error.response.data && error.response.data.error) {
      message = error.response.data.error;
    }
    return Promise.reject(new Error(message));
  }
);

export const healthCheck = async () => {
  try {
    const { data } = await client.get('/api/health');
    return data;
  } catch (err) {
    throw err;
  }
};

export const predictMatch = async (homeTeam, awayTeam) => {
  try {
    const { data } = await client.post('/api/predict', { 
      home_team: homeTeam, 
      away_team: awayTeam 
    });
    return data;
  } catch (err) {
    throw err;
  }
};

export const getMetrics = async () => {
  try {
    const { data } = await client.get('/api/metrics');
    return data;
  } catch (err) {
    throw err;
  }
};

export const getTrainingHistory = async () => {
  try {
    const { data } = await client.get('/api/training-history');
    return data;
  } catch (err) {
    throw err;
  }
};

export const getFeatureImportance = async () => {
  try {
    const { data } = await client.get('/api/feature-importance');
    return data;
  } catch (err) {
    throw err;
  }
};

export const getShapValues = async () => {
  try {
    const { data } = await client.get('/api/shap');
    return data;
  } catch (err) {
    throw err;
  }
};

export const getTeamSequence = async (teamName) => {
  try {
    const { data } = await client.get(`/api/team-sequence/${encodeURIComponent(teamName)}`);
    return data;
  } catch (err) {
    throw err;
  }
};

export const getDatasetStats = async () => {
  try {
    const { data } = await client.get('/api/dataset-stats');
    return data;
  } catch (err) {
    throw err;
  }
};
