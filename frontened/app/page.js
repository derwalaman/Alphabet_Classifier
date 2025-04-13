"use client";

import { useState } from 'react';

export default function Home() {
  const [grid, setGrid] = useState(Array(30).fill(0));
  const [prediction, setPrediction] = useState('');

  const toggleCell = (index) => {
    const updated = [...grid];
    updated[index] = grid[index] === 1 ? 0 : 1;
    setGrid(updated);
  };

  const handleSubmit = async () => {
    try {
      const res = await fetch('https://alphabetclassifier-production.up.railway.app/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: grid }),
      });

      if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);

      const result = await res.json();
      setPrediction(result.prediction);
    } catch (error) {
      alert("Error: " + error.message); // This will show error on phone
      console.error("Fetch error:", error);
    }
  };

  const resetGrid = () => {
    setGrid(Array(30).fill(0));
    setPrediction('');
  };

  return (
    <div style={{ fontFamily: 'Poppins, sans-serif', backgroundColor: '#f1f5f9', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ backgroundColor: '#0f172a', padding: '1rem 2rem', color: '#fff', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ fontSize: '1.75rem', fontWeight: '700', color: '#38bdf8' }}>Alphabet Predictor</h1>
        <a href="https://github.com/derwalaman" target="_blank" rel="noopener noreferrer" style={{ color: '#facc15', textDecoration: 'none', fontWeight: '500' }}>
          Aman Derwal ↗
        </a>
      </header>

      <main style={{ flex: 1, padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#0f172a' }}>Draw a letter on the 5x6 grid below</h2>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(6, 40px)',
            gap: '6px',
            marginBottom: '1rem',
          }}
        >
          {grid.map((cell, index) => (
            <div
              key={index}
              onClick={() => toggleCell(index)}
              style={{
                width: '40px',
                height: '40px',
                backgroundColor: cell ? '#2563eb' : '#e2e8f0',
                borderRadius: '6px',
                boxShadow: 'inset 0 0 5px rgba(0,0,0,0.2)',
                cursor: 'pointer',
                transition: 'background-color 0.2s ease-in-out',
              }}
            ></div>
          ))}
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <button
            onClick={handleSubmit}
            style={{
              backgroundColor: '#10b981',
              color: '#fff',
              padding: '0.5rem 1.25rem',
              marginRight: '10px',
              borderRadius: '8px',
              border: 'none',
              fontWeight: '600',
              fontSize: '1rem',
              cursor: 'pointer',
            }}
          >
            Predict
          </button>
          <button
            onClick={resetGrid}
            style={{
              backgroundColor: '#ef4444',
              color: '#fff',
              padding: '0.5rem 1.25rem',
              borderRadius: '8px',
              border: 'none',
              fontWeight: '600',
              fontSize: '1rem',
              cursor: 'pointer',
            }}
          >
            Clear
          </button>
        </div>

        {prediction && (
          <div style={{ textAlign: 'center', marginTop: '1.5rem', backgroundColor: '#fff', padding: '1rem', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#0f172a' }}>
              Prediction: <span style={{ color: '#2563eb', fontWeight: 'bold' }}>{prediction}</span>
            </h3>
          </div>
        )}
      </main>

      <footer style={{ backgroundColor: '#0f172a', padding: '1rem', color: '#94a3b8', textAlign: 'center' }}>
        © 2025 Aman Derwal — <a href="https://github.com/derwalaman" target="_blank" rel="noopener noreferrer" style={{ color: '#facc15', fontWeight: '500' }}>GitHub</a>
      </footer>
    </div>
  );
}
