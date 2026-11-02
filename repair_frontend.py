"""Mechanical edits in the isolated production clone; preserve visual layout."""
from pathlib import Path
p=Path('urbanflow/frontend/src/App.jsx')
s=p.read_text(encoding='utf-8')
a=s.index('const MOCK_ALERTS ='); b=s.index('// --- CORRECTED GAUGE WIDGET ---')
s=s[:a]+s[b:]
s=s.replace('{value || 0}%',"{value == null ? 'N/A' : `${value}%`}")
s=s.replace('useState("")', 'useState("2016-03-05T23:00")',1)
s=s.replace("confidence: 95",'confidence: null')
s=s.replace('useState([MOCK_ALERTS[0], MOCK_ALERTS[1]])','useState([])')
s=s.replace('useState({ ms: 42, gpu: 64 })','useState({ ms: null })')
a=s.index('  useEffect(() => { shiftTime');b=s.index('    try {',a)
s=s[:a]+'''  useEffect(() => { handleCompute(); }, []);

  const handleCompute = async () => {
    setLoading(true);
    const started = performance.now();
'''+s[b:]
s=s.replace("'http://140.238.240.219:8000'","'http://127.0.0.1:8000'")
s=s.replace('datetime: timeValue || new Date().toISOString(),','datetime: timeValue, mode: "historical",')
s=s.replace('if (!response.ok) throw new Error("API Offline");','if (!response.ok) { const failure = await response.json(); throw new Error(typeof failure.detail === "string" ? failure.detail : "Invalid forecasting request"); }')
s=s.replace('const pred = item.value || item.yhat || item.prediction || Object.values(item)[1] || 0;','const pred = item.prediction ?? item.value ?? item.yhat ?? 0;')
s=s.replace('confidenceBand: [item.lower_bound || pred * 0.85, item.upper_bound || pred * 1.15]','confidenceBand: null')
a=s.index('// Dynamic confidence');b=s.index('setGauges({',a)
s=s[:a]+'const confidence = null;\n\n'+s[b:]
s=s.replace('temp: realData.weather?.temp || weather.temp,','temp: realData.weather?.temperature ?? "N/A",')
s=s.replace('condition: realData.weather?.condition || weather.condition,','condition: realData.weather?.condition ?? "Unavailable",')
s=s.replace('precip: weather.precip,','precip: realData.weather?.precipitation ?? "N/A",')
s=s.replace('snow: weather.snow,','snow: realData.weather?.snowfall ?? "N/A",')
s=s.replace("flow: scenario > 1.2 ? 'Demand Surge' : scenario < 0.8 ? 'Suppressed Flow' : 'Standard Flow',","flow: 'Historical simulation',")
s=s.replace('mape: realData.metrics?.mape || (4.2 + (Math.abs(1 - scenario) * 5)).toFixed(2),','mape: "N/A",')
s=s.replace('rmse: realData.metrics?.rmse || 12.4,','rmse: "N/A",')
s=s.replace('mae: realData.metrics?.mae || 8.1','mae: "N/A"')
s=s.replace('Math.max(\n            5,\n            Math.round((f.score / 1200) * 100)\n          )','Math.round(100 * f.score / Math.max(1, ...(realData.feature_importance || []).map(v => v.score)))')
a=s.index('    } catch (error) {');b=s.index('    } finally {',a)
s=s[:a]+'''      setActiveAlerts([]);
      setTelemetry({ ms: Math.round(performance.now() - started) });
    } catch (error) {
      setGraphData([]); setFeatures([]); setHeatmap([]);
      setGauges({ momentum: 0, momentumTrend: 'Unavailable', confidence: null });
      setKpis({ reqs: 'N/A', temp: 'N/A', condition: 'Unavailable', precip: 'N/A', snow: 'N/A', flow: 'Unavailable', mape: 'N/A', rmse: 'N/A', mae: 'N/A' });
      setActiveAlerts([{ id: 'request-error', icon: AlertTriangle, color: 'text-amber-400', title: 'Forecast unavailable', desc: error.message }]);
'''+s[b:]
a=s.index('    const d = new Date(); d.setHours');b=s.index('\n  };',a)
s=s[:a]+'''    const d = new Date(`${timeValue || '2016-03-05T23:00'}Z`);
    d.setUTCHours(d.getUTCHours() + hours);
    setTimeValue(d.toISOString().slice(0, 16));'''+s[b:]
s=s.replace('VRAM: {telemetry.gpu}%','GPU telemetry unavailable')
s=s.replace('label="Bound" subLabel={gauges.confidence > 90 ? "High Reliability" : "Moderate Variance"}','label="Uncertainty" subLabel="Intervals unavailable"')
s=s.replace('<select value={scenario}', '<select disabled value={scenario}')
s=s.replace('Scenario Injection','Scenarios unavailable')
s=s.replace('{horizon}h','{horizon} steps · {horizon / 4}h')
s=s.replace('>Live</button>','>Origin</button>')
s=s.replace('>Peak</button>','>+5h</button>')
s=s.replace('Adjust temporal memory parameters & trigger LightGBM step-ahead pipeline.','Historical simulation using original 2016 observations. Times are America/New_York. Operational alerts and confidence intervals are unavailable.')
p.write_text(s,encoding='utf-8')
