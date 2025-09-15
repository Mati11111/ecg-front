import React, { useEffect, useRef } from 'react';
import { Chart, CategoryScale, LinearScale, LineController, LineElement, PointElement, Title, Tooltip, Legend } from 'chart.js';

Chart.register(
  CategoryScale,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

interface PlotSignalsProps {
  ecgData: { labels: string[]; values: number[] };
}

const PlotSignals: React.FC<PlotSignalsProps> = ({ ecgData }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    if (chartRef.current) chartRef.current.destroy();

    chartRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: ecgData.labels,
        datasets: [{
          label: 'ECG Signal',
          data: ecgData.values,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false }
        },
        scales: {
          x: { display: false },
          y: {
            display: true,
            min: -1,      
            max: 1,      
            grid: { color: 'rgba(0,0,0,0.1)' }
          }
        },
        interaction: { intersect: false },
        animation: { duration: 0 }
      }
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [ecgData]); 

  return (
    <div style={{ width: '100%', height: '100%', padding: '10px' }}>
      <canvas ref={canvasRef} />
    </div>
  );
};

export default PlotSignals;
