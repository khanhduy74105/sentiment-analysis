import React, { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      console.log("Response status:", response);

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setResult({ error: "Prediction failed. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="card">
        <h1>Sentiment Predictor</h1>
        <textarea
          placeholder="Type your review here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        ></textarea>
        <button onClick={handlePredict} disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>

        {result && (
          <div className="result">
            {result.error ? (
              <p className="error">{result.error}</p>
            ) : (
              <>
                <p><strong>Label:</strong> {result.label}</p>
                <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
