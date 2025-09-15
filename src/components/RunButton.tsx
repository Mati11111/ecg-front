"use client";
interface RunButtonProps {
  backendUrl: string;
}

export default function RunButton({ backendUrl }: RunButtonProps) {
  const handlePrediction = async () => {
    try {
      console.log("RUN prediction triggered");

      const response = await fetch(`${backendUrl}/doPrediction`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      console.log("Prediction result:", data);

    } catch (error) {
      console.error("Error making prediction:", error);
    }
  };

  return (
    <button
      className="bg-black/60 font-bold text-white px-6 py-2"
      onClick={handlePrediction}
    >
      RUN
    </button>
  );
}