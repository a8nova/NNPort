import { useState, useEffect } from 'react'

function App() {
  const [sourceFile, setSourceFile] = useState(null)
  const [targetType, setTargetType] = useState('DSP')
  const [deviceConfig, setDeviceConfig] = useState({ mock: false, use_adb: false, adb_device_id: '' })
  const [inputShape, setInputShape] = useState('1, 10')
  const [logs, setLogs] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [systemStatus, setSystemStatus] = useState(null)

  const [activeTab, setActiveTab] = useState('auto') // 'auto' or 'manual'
  const [manualSourceFile, setManualSourceFile] = useState(null)
  const [manualRefFile, setManualRefFile] = useState(null)

  // Fetch system status on mount
  useEffect(() => {
    fetch('/status')
      .then(res => res.json())
      .then(data => setSystemStatus(data))
      .catch(err => console.error('Failed to fetch system status:', err))
  }, [])

  const handleUploadGeneric = async (file, setter) => {
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)
    try {
      setLoading(true)
      const res = await fetch('/upload', { method: 'POST', body: formData })
      if (!res.ok) throw new Error('Upload failed')
      const data = await res.json()
      setter(data.filename)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handlePort = async () => {
    if (!sourceFile) return
    try {
      setLoading(true)
      setError(null)
      setLogs(null)
      const res = await fetch('/port', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_filename: sourceFile,
          target_type: targetType,
          device_config: deviceConfig,
          max_iterations: 3,
          input_shape: inputShape.split(',').map(n => parseInt(n.trim())),
        }),
      })
      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Porting failed')
      }
      const data = await res.json()
      setLogs(data.logs)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleVerify = async () => {
    if (!manualSourceFile || !manualRefFile) return
    try {
      setLoading(true)
      setError(null)
      setLogs(null)
      const res = await fetch('/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_filename: manualSourceFile,
          reference_filename: manualRefFile,
          target_type: targetType,
          device_config: deviceConfig,
          input_shape: inputShape.split(',').map(n => parseInt(n.trim())),
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Verification failed')
      }
      const data = await res.json()
      setLogs(data.logs)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="layout">
      {/* Sidebar Configuration */}
      <aside className="sidebar">
        <h1 className="section-title">NNPort</h1>

        {/* Tab Switcher */}
        <div style={{ display: 'flex', marginBottom: '1.5rem', borderBottom: '1px solid var(--border)' }}>
          <button
            onClick={() => { setActiveTab('auto'); setLogs(null); setError(null); }}
            style={{
              flex: 1,
              background: activeTab === 'auto' ? 'rgba(255,255,255,0.1)' : 'transparent',
              border: 'none',
              borderBottom: activeTab === 'auto' ? '2px solid var(--accent)' : 'none',
              borderRadius: 0,
              padding: '0.5rem',
              cursor: 'pointer',
              color: activeTab === 'auto' ? 'var(--text-primary)' : 'var(--text-secondary)'
            }}
          >
            Auto-Port (AI)
          </button>
          <button
            onClick={() => { setActiveTab('manual'); setLogs(null); setError(null); }}
            style={{
              flex: 1,
              background: activeTab === 'manual' ? 'rgba(255,255,255,0.1)' : 'transparent',
              border: 'none',
              borderBottom: activeTab === 'manual' ? '2px solid var(--accent)' : 'none',
              borderRadius: 0,
              padding: '0.5rem',
              cursor: 'pointer',
              color: activeTab === 'manual' ? 'var(--text-primary)' : 'var(--text-secondary)'
            }}
          >
            Vibe Debug
          </button>
        </div>

        {activeTab === 'auto' ? (
          <div className="control-group">
            <label className="control-label">Source Model (Reference)</label>
            <input
              type="file"
              accept=".py"
              onChange={(e) => handleUploadGeneric(e.target.files[0], setSourceFile)}
            />
            {sourceFile && <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--success)' }}>✓ {sourceFile}</div>}
          </div>
        ) : (
          <>
            <div className="control-group">
              <label className="control-label">C++ Source Code</label>
              <input
                type="file"
                accept=".cpp,.c,.cl,.cu"
                onChange={(e) => handleUploadGeneric(e.target.files[0], setManualSourceFile)}
              />
              {manualSourceFile && <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--success)' }}>✓ {manualSourceFile}</div>}
            </div>

            <div className="control-group">
              <label className="control-label">Reference Model (.py)</label>
              <input
                type="file"
                accept=".py"
                onChange={(e) => handleUploadGeneric(e.target.files[0], setManualRefFile)}
              />
              {manualRefFile && <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--success)' }}>✓ {manualRefFile}</div>}
            </div>
          </>
        )}

        <div className="control-group">
          <label className="control-label">Target Hardware</label>
          <select
            value={targetType}
            onChange={(e) => setTargetType(e.target.value)}
          >
            <option value="DSP">DSP (C6x)</option>
            <option value="CUDA">NVIDIA CUDA</option>
            <option value="OpenCL">OpenCL Generic</option>
          </select>
        </div>

        {/* ADB Configuration */}
        <div style={{ marginBottom: '1.5rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '1rem' }}>
          <div style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem', color: 'var(--text-primary)' }}>Device Configuration</div>

          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
            <input
              type="checkbox"
              id="useAdb"
              checked={deviceConfig.use_adb}
              onChange={(e) => setDeviceConfig({ ...deviceConfig, use_adb: e.target.checked })}
              style={{ width: 'auto', marginRight: '0.5rem' }}
            />
            <label htmlFor="useAdb" style={{ fontSize: '0.875rem' }}>Use ADB (Android Device)</label>
          </div>

          {deviceConfig.use_adb && (
            <div style={{ marginBottom: '0.5rem' }}>
              <label className="control-label" style={{ fontSize: '0.8rem' }}>ADB Device ID</label>
              <input
                type="text"
                placeholder="Serial (optional)"
                value={deviceConfig.adb_device_id || ''}
                onChange={(e) => setDeviceConfig({ ...deviceConfig, adb_device_id: e.target.value })}
              />
            </div>
          )}
          
          {!deviceConfig.use_adb && (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
              Running locally on this machine
            </div>
          )}
        </div>

        <div className="control-group">
          <label className="control-label">Input Shape (comma separated)</label>
          <input
            type="text"
            placeholder="e.g. 1, 10"
            value={inputShape}
            onChange={(e) => setInputShape(e.target.value)}
          />
        </div>

        {activeTab === 'auto' ? (
          <button
            onClick={handlePort}
            disabled={!sourceFile || loading}
            style={{ width: '100%' }}
          >
            {loading ? 'Running Porting Loop...' : 'Start Porting'}
          </button>
        ) : (
          <button
            onClick={handleVerify}
            disabled={!manualSourceFile || !manualRefFile || loading}
            style={{ width: '100%' }}
          >
            {loading ? 'Vibing...' : 'Vibe Debug'}
          </button>
        )}

        {error && (
          <div style={{ marginTop: '1rem', color: 'var(--error)', fontSize: '0.875rem' }}>
            Error: {error}
          </div>
        )}

        {/* System Status */}
        {systemStatus && (
          <div style={{ marginTop: '1.5rem', padding: '1rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)', fontSize: '0.75rem' }}>
            <div style={{ fontWeight: '600', marginBottom: '0.5rem', color: 'var(--text-primary)' }}>System Status</div>

            <div style={{ marginBottom: '0.5rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>NDK: </span>
              <span style={{ color: systemStatus.ndk.available ? 'var(--success)' : 'var(--error)' }}>
                {systemStatus.ndk.available ? '✓ Available' : '✗ Not Found'}
              </span>
              {systemStatus.ndk.available && (
                <div style={{ marginLeft: '1rem', color: 'var(--text-secondary)', fontSize: '0.7rem' }}>
                  {systemStatus.ndk.path}
                </div>
              )}
            </div>

            <div>
              <span style={{ color: 'var(--text-secondary)' }}>ADB Devices: </span>
              <span style={{ color: systemStatus.adb.devices.length > 0 ? 'var(--success)' : 'var(--text-secondary)' }}>
                {systemStatus.adb.devices.length > 0 ? `${systemStatus.adb.devices.length} connected` : 'None'}
              </span>
              {systemStatus.adb.devices.length > 0 && (
                <div style={{ marginLeft: '1rem', color: 'var(--text-secondary)', fontSize: '0.7rem' }}>
                  {systemStatus.adb.devices.join(', ')}
                </div>
              )}
            </div>
          </div>
        )}
      </aside>

      {/* Main Content Results */}
      <main className="main-content">
        <div className="card">
          <h2 className="section-title">Porting Progress</h2>
          {!logs && !loading ? (
            <div style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '2rem' }}>
              Upload a model and select target hardware to begin vibe debuging.
            </div>
          ) : (
            <div className="logs-container">
              {loading && !logs && <div>Initializing...</div>}
              {logs && logs.map((log, i) => (
                <div key={i} style={{ marginBottom: '1rem', borderLeft: log.status === 'Success' ? '3px solid var(--success)' : '3px solid var(--border)', paddingLeft: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <span style={{ fontWeight: '600', color: 'var(--accent)' }}>{log.stage}</span>
                      <span style={{
                        marginLeft: '1rem',
                        color: log.status === 'Success' ? 'var(--success)' :
                          log.status === 'Failed' ? 'var(--error)' : 'var(--text-primary)'
                      }}>{log.status}</span>
                    </div>
                    {log.status === 'Success' && log.source_preview && (
                      <button
                        onClick={() => {
                          const blob = new Blob([log.source_preview], { type: 'text/plain' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `ported_model_${targetType}.cpp`;
                          a.click();
                        }}
                        style={{
                          fontSize: '0.75rem',
                          padding: '0.25rem 0.5rem',
                          height: 'auto',
                          background: 'var(--accent)',
                        }}
                      >
                        Download Code
                      </button>
                    )}
                  </div>
                  {log.details && <div style={{ fontSize: '0.875rem', marginTop: '0.25rem', color: 'var(--text-secondary)' }}>{log.details}</div>}
                  {log.source_preview && (
                    <pre style={{
                      fontSize: '0.75rem',
                      backgroundColor: 'rgba(0,0,0,0.3)',
                      padding: '0.5rem',
                      borderRadius: '4px',
                      marginTop: '0.5rem',
                      overflowX: 'auto',
                      maxHeight: '300px'
                    }}>
                      {log.source_preview}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
