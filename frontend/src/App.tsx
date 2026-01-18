import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Activity, Cpu, Server, Terminal, Zap, HardDrive,
  BarChart3, AlertTriangle, CheckCircle2, AlertOctagon, Info
} from 'lucide-react';

// --- Types ---
interface DiagnosticResponse {
  diagnosis: string;
  inference_time_sec: number;
  model_version: string;
}

interface BenchmarkStats {
  avg_tps: number;
  peak_memory_mb: number;
  load_time_sec: number;
  device: string;
}

// --- Configuration ---
const API_URL = "http://localhost:8000";

// --- Helper: Severity Logic ---
const getSeverity = (text: string) => {
  const lower = text.toLowerCase();
  if (lower.includes("critical") || lower.includes("fail") || lower.includes("error") || lower.includes("emergency")) {
    return { color: "bg-red-50 border-red-200 text-red-900", icon: <AlertOctagon className="w-5 h-5 text-red-600" />, label: "CRITICAL FAULT" };
  }
  if (lower.includes("warning") || lower.includes("check") || lower.includes("inspect") || lower.includes("unstable")) {
    return { color: "bg-amber-50 border-amber-200 text-amber-900", icon: <AlertTriangle className="w-5 h-5 text-amber-600" />, label: "WARNING" };
  }
  return { color: "bg-blue-50 border-blue-200 text-slate-800", icon: <Info className="w-5 h-5 text-blue-600" />, label: "SYSTEM INFO" };
};

function App() {
  const [status, setStatus] = useState<"offline" | "online" | "loading">("loading");
  const [query, setQuery] = useState("");
  const [context, setContext] = useState("");
  const [response, setResponse] = useState<DiagnosticResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [stats, setStats] = useState<BenchmarkStats | null>(null);

  // 1. Check System Health & Fetch Stats
  useEffect(() => {
    const initSystem = async () => {
      try {
        await axios.get(`${API_URL}/health`);
        setStatus("online");
        const benchRes = await axios.get(`${API_URL}/benchmark`);
        if (benchRes.data.status === "success") {
          setStats(benchRes.data);
        }
      } catch (e) {
        setStatus("offline");
      }
    };
    initSystem();
  }, []);

  // 2. Handle Diagnosis Request
  const handleDiagnose = async () => {
    if (!query) return;
    setIsAnalyzing(true);
    setResponse(null);

    try {
      const res = await axios.post(`${API_URL}/diagnose`, {
        query,
        context: context || undefined,
        max_tokens: 50
      }, { timeout: 600000 });
      setResponse(res.data);
    } catch (err) {
      alert("Analysis failed. Check backend console.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const severity = response ? getSeverity(response.diagnosis) : null;

  return (
    <div className="min-h-screen p-8 font-sans bg-slate-50 text-slate-900">

      {/* Header */}
      <header className="max-w-6xl mx-auto mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Cpu className="w-8 h-8 text-blue-600" />
            NanoSentri <span className="text-slate-400 font-light">Edge</span>
          </h1>
          <p className="text-slate-500 mt-1">AI Powered Industrial Diagnostics Module</p>
        </div>
        <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${status === 'online' ? 'bg-green-50 border-green-200 text-green-700' : 'bg-red-50 border-red-200 text-red-700'
          }`}>
          <Activity className="w-4 h-4" />
          <span className="text-sm font-medium uppercase tracking-wider">{status}</span>
        </div>
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* LEFT COLUMN: Controls & Inputs */}
        <div className="lg:col-span-2 space-y-6">

          {/* Hardware Stats Row */}
          {stats && (
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col items-center justify-center">
                <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase mb-1">
                  <Zap className="w-3 h-3" /> Speed
                </div>
                <div className="text-2xl font-bold text-slate-800 tracking-tight">{stats.avg_tps} <span className="text-sm font-normal text-slate-400">tok/s</span></div>
              </div>
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col items-center justify-center">
                <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase mb-1">
                  <HardDrive className="w-3 h-3" /> RAM Usage
                </div>
                <div className="text-2xl font-bold text-slate-800 tracking-tight">{stats.peak_memory_mb} <span className="text-sm font-normal text-slate-400">MB</span></div>
              </div>
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col items-center justify-center">
                <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase mb-1">
                  <BarChart3 className="w-3 h-3" /> Load Time
                </div>
                <div className="text-2xl font-bold text-slate-800 tracking-tight">{stats.load_time_sec} <span className="text-sm font-normal text-slate-400">s</span></div>
              </div>
            </div>
          )}

          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <label className="block text-sm font-semibold text-slate-700 mb-2">Technical Query</label>
            <input
              type="text"
              className="w-full p-4 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition text-lg"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. Why is the wind sensor reporting error 503?"
            />
          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <label className="block text-sm font-semibold text-slate-700 mb-2">Sensor Logs / Context</label>
            <textarea
              className="w-full h-48 p-4 bg-slate-900 text-green-400 font-mono text-sm border border-slate-800 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition leading-relaxed"
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder="// Paste raw sensor logs here..."
            />
          </div>

          <button
            onClick={handleDiagnose}
            disabled={status !== 'online' || isAnalyzing}
            className={`w-full py-4 rounded-xl font-bold text-white shadow-lg transition-all flex flex-col items-center justify-center gap-1 ${status === 'online' && !isAnalyzing ? 'bg-slate-900 hover:bg-blue-600 active:scale-[0.98]' : 'bg-slate-300 cursor-not-allowed'
              }`}
          >
            {isAnalyzing ? (
              <>
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Processing Neural Graph...</span>
                </div>
                <span className="text-[10px] opacity-80 animate-pulse">Running on Local CPU (INT4 Quantized)</span>
              </>
            ) : (
              <div className="flex items-center gap-2">
                <Terminal className="w-5 h-5" />
                <span>Run Diagnostics</span>
              </div>
            )}
          </button>
        </div>

        {/* RIGHT COLUMN: The "Beautified" Result Card */}
        <div className="lg:col-span-1">
          <div className="bg-white h-full rounded-xl shadow-sm border border-slate-200 flex flex-col overflow-hidden">

            {/* Card Header */}
            <div className="p-5 border-b border-slate-100 bg-slate-50/50">
              <h3 className="text-sm font-bold text-slate-700 flex items-center gap-2">
                <Server className="w-4 h-4 text-blue-500" />
                AI DIAGNOSTIC REPORT
              </h3>
            </div>

            <div className="flex-1 p-6 flex flex-col">
              {response && severity ? (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 flex flex-col h-full">

                  {/* Status Badge */}
                  <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wide w-fit mb-4 ${severity.color.replace('border', '')} border`}>
                    {severity.icon}
                    {severity.label}
                  </div>

                  {/* Diagnosis Text */}
                  <div className={`flex-1 p-4 rounded-lg border text-sm leading-relaxed font-medium mb-6 ${severity.color}`}>
                    <p className="whitespace-pre-wrap">{response.diagnosis}</p>
                  </div>

                  {/* Metadata Footer */}
                  <div className="mt-auto space-y-3 pt-4 border-t border-slate-100">
                    <div className="flex justify-between items-center text-xs text-slate-500">
                      <span>Inference Time</span>
                      <span className="font-mono bg-slate-100 px-2 py-1 rounded text-slate-700">{response.inference_time_sec}s</span>
                    </div>
                    <div className="flex justify-between items-center text-xs text-slate-500">
                      <span>Model Version</span>
                      <span className="font-mono text-blue-600 bg-blue-50 px-2 py-1 rounded">{response.model_version}</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-green-600 font-bold mt-2 justify-center bg-green-50 py-2 rounded border border-green-100">
                      <CheckCircle2 className="w-3 h-3" />
                      INT4 QUANTIZED
                    </div>
                  </div>

                </div>
              ) : (
                /* Empty State */
                <div className="flex-1 flex flex-col items-center justify-center text-slate-300">
                  <div className="w-16 h-16 rounded-full bg-slate-50 flex items-center justify-center mb-4">
                    <Activity className="w-8 h-8 opacity-20" />
                  </div>
                  <p className="text-sm font-medium">Ready for input</p>
                  <p className="text-xs text-slate-400 mt-2 max-w-[200px] text-center">Enter a query and logs to generate a diagnostic report.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;