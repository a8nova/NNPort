import { useState } from 'react'

export default function ConnectionConfig({ config, onChange }) {
  const [expanded, setExpanded] = useState(true)
  const [findingAdb, setFindingAdb] = useState(false)
  const [adbResult, setAdbResult] = useState(null)

  const handleChange = (field, value) => {
    onChange({ ...config, [field]: value })
  }

  const handleFindAdb = async () => {
    try {
      setFindingAdb(true)
      setAdbResult(null)
      const res = await fetch('/discover-adb')
      const data = await res.json()
      
      if (data.found) {
        handleChange('adb_path', data.path)
        setAdbResult({
          success: true,
          message: `Found ADB: ${data.path}`,
          details: data.version,
          devices: data.devices || []
        })
      } else {
        setAdbResult({
          success: false,
          message: 'ADB not found',
          details: data.error
        })
      }
    } catch (err) {
      console.error('Failed to find ADB:', err)
      setAdbResult({
        success: false,
        message: 'Failed to find ADB',
        details: err.message
      })
    } finally {
      setFindingAdb(false)
    }
  }

  return (
    <div className="config-section">
      <div 
        style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          cursor: 'pointer',
          marginBottom: '0.5rem'
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <h3 style={{ margin: 0, fontSize: '0.9rem' }}>Connection Settings</h3>
        <span style={{ fontSize: '0.8rem' }}>{expanded ? '▼' : '▶'}</span>
      </div>

      {expanded && (
        <>
          <label className="control-label">Connection Type</label>
          <select
            value={config.connection_type || 'local'}
            onChange={(e) => handleChange('connection_type', e.target.value)}
          >
            <option value="local">Local (Host Machine)</option>
            <option value="adb">ADB (Android Device)</option>
            <option value="ssh">SSH (Remote Device)</option>
          </select>

          {config.connection_type === 'adb' && (
            <>
              <label className="control-label">ADB Path</label>
              <input
                type="text"
                placeholder="/path/to/adb"
                value={config.adb_path || 'adb'}
                onChange={(e) => handleChange('adb_path', e.target.value)}
              />
              
              <button 
                onClick={handleFindAdb}
                disabled={findingAdb}
                style={{ 
                  width: '100%', 
                  marginBottom: '1rem', 
                  fontSize: '0.85rem',
                  padding: '0.5rem',
                  background: '#4A90E2',
                  border: 'none',
                  color: 'white',
                  cursor: findingAdb ? 'wait' : 'pointer'
                }}
              >
                {findingAdb ? '🔍 Finding ADB...' : '🔍 Find ADB on Host'}
              </button>

              {adbResult && (
                <div style={{ 
                  marginBottom: '1rem', 
                  padding: '0.75rem', 
                  background: adbResult.success ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)',
                  borderRadius: 'var(--radius)',
                  fontSize: '0.8rem',
                  border: `1px solid ${adbResult.success ? 'rgba(76, 175, 80, 0.3)' : 'rgba(244, 67, 54, 0.3)'}`
                }}>
                  <div style={{ fontWeight: '600', marginBottom: '0.25rem', color: adbResult.success ? '#4CAF50' : '#F44336' }}>
                    {adbResult.message}
                  </div>
                  {adbResult.details && (
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                      {adbResult.details}
                    </div>
                  )}
                  {adbResult.success && adbResult.devices && adbResult.devices.length > 0 && (
                    <div style={{ fontSize: '0.75rem', marginTop: '0.5rem' }}>
                      <strong>Connected devices:</strong> {adbResult.devices.join(', ')}
                    </div>
                  )}
                </div>
              )}

              <label className="control-label">ADB Device ID (optional)</label>
              <input
                type="text"
                placeholder="e.g., emulator-5554"
                value={config.adb_device_id || ''}
                onChange={(e) => handleChange('adb_device_id', e.target.value)}
              />
            </>
          )}

          {config.connection_type === 'ssh' && (
            <>
              <label className="control-label">SSH Host</label>
              <input
                type="text"
                placeholder="192.168.1.100"
                value={config.ssh_host || ''}
                onChange={(e) => handleChange('ssh_host', e.target.value)}
              />

              <label className="control-label">SSH Port</label>
              <input
                type="number"
                placeholder="22"
                value={config.ssh_port || 22}
                onChange={(e) => handleChange('ssh_port', parseInt(e.target.value))}
              />

              <label className="control-label">SSH Username</label>
              <input
                type="text"
                placeholder="root"
                value={config.ssh_user || ''}
                onChange={(e) => handleChange('ssh_user', e.target.value)}
              />

              <label className="control-label">SSH Password</label>
              <input
                type="password"
                placeholder="password"
                value={config.ssh_password || ''}
                onChange={(e) => handleChange('ssh_password', e.target.value)}
              />

              <label className="control-label">SSH Key Path (optional)</label>
              <input
                type="text"
                placeholder="~/.ssh/id_rsa"
                value={config.ssh_key_path || ''}
                onChange={(e) => handleChange('ssh_key_path', e.target.value)}
              />
            </>
          )}

          <label className="control-label">Remote Work Directory</label>
          <input
            type="text"
            placeholder="/tmp/nnport"
            value={config.remote_work_dir || '/data/local/tmp'}
            onChange={(e) => handleChange('remote_work_dir', e.target.value)}
          />
        </>
      )}
    </div>
  )
}
