import React, { useEffect, useRef } from "react";
import {
  Chart,
  CategoryScale,
  LinearScale,
  BarController,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

Chart.register(
  CategoryScale,
  LinearScale,
  BarController,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface ProcessPredictionProps {
  ecgData: {
    predictedClasses: (number | string)[];
    allClasses?: (number | string)[]; 
  };
}

const ProcessPrediction: React.FC<ProcessPredictionProps> = ({ ecgData }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    if (chartRef.current) chartRef.current.destroy();

    const allClasses =
      ecgData.allClasses ??
      Array.from(new Set(ecgData.predictedClasses)).sort((a, b) =>
        a.toString().localeCompare(b.toString())
      );

    const classCounts = allClasses.map(
      (cls) =>
        ecgData.predictedClasses.filter((c) => c.toString() === cls.toString())
          .length
    );

    chartRef.current = new Chart(ctx, {
      type: "bar",
      data: {
        labels: allClasses.map((c) => c.toString()),
        datasets: [
          {
            label: "Cantidad por clase",
            data: classCounts,
            backgroundColor: "rgba(75, 192, 192, 0.6)",
            borderColor: "rgba(75, 192, 192, 1)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true },
          tooltip: { enabled: true },
        },
        scales: {
          x: { title: { display: true, text: "Clase" } },
          y: {
            title: { display: true, text: "Cantidad" },
            beginAtZero: true,
            min: 0,
            max: 187,
            ticks: { stepSize: 10 },
          },
        },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [ecgData]);

  return (
    <div style={{ width: "100%", height: "400px", padding: "10px" }}>
      <canvas ref={canvasRef} />
    </div>
  );
};

export default ProcessPrediction;
