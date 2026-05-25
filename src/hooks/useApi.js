import { useState, useEffect, useCallback } from 'react';
import { predictMatch } from '../api/client';

export function useApi(apiFn, params = null, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchApi = useCallback(async () => {
    if (!apiFn) return;
    setLoading(true);
    setError(null);
    try {
      const res = params ? await apiFn(params) : await apiFn();
      setData(res);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [apiFn, params]);

  useEffect(() => {
    fetchApi();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, loading, error, refetch: fetchApi };
}

export function usePredict() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = async (homeTeam, awayTeam) => {
    setLoading(true);
    setError(null);
    try {
      const res = await predictMatch(homeTeam, awayTeam);
      setResult(res);
      return res; // useful if caller wants it immediately
    } catch (err) {
      setError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { predict, result, loading, error };
}
