import React, { createContext, useState, useContext } from 'react';

const PredictionContext = createContext(null);

export const PredictionProvider = ({ children }) => {
  const [lastPrediction, setLastPrediction] = useState(null);

  return (
    <PredictionContext.Provider value={{ lastPrediction, setLastPrediction }}>
      {children}
    </PredictionContext.Provider>
  );
};

export const usePredictionContext = () => useContext(PredictionContext);
