import { useEffect, useState } from 'react'
import axios from 'axios'
import './App.css'

// Popup UI. Based on the original CMPE209FinalProject popup; changed to
// (1) ask the Gmail tab for the open email when the popup opens, and
// (2) call the hybrid detector (/api/hybrid-scan) and show which stage decided.

const API = 'http://127.0.0.1:5000/api/hybrid-scan'

function App() {
  const [email, setEmail] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [emailLoading, setEmailLoading] = useState(true);

  useEffect(() => {
    // Ask the content script in the active tab for the email that is open now.
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs[0];
      if (!tab || !tab.url || !tab.url.startsWith('https://mail.google.com')) {
        setEmailLoading(false);
        return;
      }
      chrome.tabs.sendMessage(tab.id, { type: 'GET_EMAIL' }, (response) => {
        if (chrome.runtime.lastError) {
          // content script not loaded yet (page opened before the extension): refresh Gmail
          setEmailLoading(false);
          return;
        }
        if (response?.payload) setEmail(response.payload);
        setEmailLoading(false);
      });
    });

    // Still accept pushes from the content script (original behaviour).
    const listener = (request) => {
      if (request.type === 'EMAIL_DATA') {
        setEmail(request.payload);
        setEmailLoading(false);
      }
    };
    chrome.runtime.onMessage.addListener(listener);
    return () => chrome.runtime.onMessage.removeListener(listener);
  }, []);

  const handleScan = async () => {
    if (!email || !email.body) {
      setResult({ label: 'Error', explanation: 'Could not read the email body. Open the email fully and try again.' });
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post(API, email);
      const d = res.data;
      setResult({
        label: d.is_phishing ? '⚠️ Phishing Detected' : '✅ Not Phishing',
        explanation: d.explanation,
        detail: `Decided by ${d.decided_by} · ML phishing probability ${(d.phishing_probability * 100).toFixed(1)}%`
      });
    } catch (err) {
      setResult({ label: 'Error', explanation: err.response?.data?.error || err.message + ' (is the backend running?)' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Hybrid Phishing Detector</h1>

      {emailLoading ? (
        <div className="load-wrapper">
          <div className="loader"></div>
          <p className="load-text">Loading email...</p>
        </div>
      ) : email ? (
        <div className="card">
          <h3>{email.subject}</h3>
          <p>From: {email.sender}</p>
          <p>{email.body}</p>
        </div>
      ) : (
        <p>Open an email in Gmail, then click the extension again. If Gmail was open before the extension was installed, refresh the Gmail tab.</p>
      )}

      {!emailLoading && email && (
        <button onClick={handleScan} disabled={loading}>Scan Email</button>
      )}

      {loading && <div className="loader"></div>}
      {result && (
        <div className="result-card">
          <h2>{result.label}</h2>
          <p>{result.explanation}</p>
          {result.detail && <p className="result-detail">{result.detail}</p>}
        </div>
      )}
    </div>
  )
}

export default App
