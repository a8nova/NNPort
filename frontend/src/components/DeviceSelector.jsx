import { useState, useEffect } from 'react'

export default function DeviceSelector({ targetType, config, onChange }) {
  const [devices, setDevices] = useState(null)
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(false)
  
  // Only show device selector for GPU/accelerator targets
  const shouldShow = ['OpenCL', 'CUDA', 'GPU'].some(t => 
    targetType.toUpperCase().includes(t.toUpperCase())
  )
  
  useEffect(() => {
    if (shouldShow && expanded && !devices) {
      discoverDevices()
    }
  }, [shouldShow, expanded])
  
  const discoverDevices = async () => {
    try {
      setLoading(true)
      const res = await fetch('/discover-devices')
      const data = await res.json()
      setDevices(data)
    } catch (err) {
      console.error('Failed to discover devices:', err)
    } finally {
      setLoading(false)
    }
  }
  
  const handleSelectDevice = (backend, device) => {
    onChange({
      ...config,
      compute_backend: backend,
      compute_device_type: device.type,
      compute_platform_id: device.platform_id || 0,
      compute_device_id: device.device_id || 0
    })
    setExpanded(false)
  }
  
  if (!shouldShow) return null
  
  const currentDevice = config.compute_backend !== 'auto' 
    ? `${config.compute_backend.toUpperCase()} - ${config.compute_device_type}`
    : 'Auto-select'
  
  return (
    <div className="config-section" style={{ marginTop: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
        <h3 style={{ fontSize: '0.9rem', fontWeight: '600', margin: 0 }}>
          🖥️ Compute Device
        </h3>
        <button 
          onClick={() => setExpanded(!expanded)}
          style={{ 
            background: 'transparent', 
            border: 'none', 
            cursor: 'pointer',
            fontSize: '0.85rem',
            color: 'var(--text-secondary)'
          }}
        >
          {expanded ? '▼' : '▶'}
        </button>
      </div>
      
      <div style={{ 
        fontSize: '0.8rem', 
        color: 'var(--text-secondary)', 
        marginBottom: '0.5rem',
        padding: '0.5rem',
        background: 'rgba(74, 144, 226, 0.1)',
        borderRadius: '4px'
      }}>
        Selected: <strong style={{ color: '#4A90E2' }}>{currentDevice}</strong>
      </div>
      
      {expanded && (
        <div style={{ 
          padding: '0.75rem', 
          background: 'rgba(0, 0, 0, 0.2)',
          borderRadius: 'var(--radius)',
          marginTop: '0.5rem'
        }}>
          {loading ? (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              🔍 Discovering devices...
            </div>
          ) : devices ? (
            <>
              <button
                onClick={() => onChange({ ...config, compute_backend: 'auto' })}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  marginBottom: '0.5rem',
                  background: config.compute_backend === 'auto' ? 'rgba(74, 144, 226, 0.3)' : 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(74, 144, 226, 0.3)',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                  textAlign: 'left'
                }}
              >
                🤖 Auto-select (recommended)
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  Automatically choose the best available device
                </div>
              </button>
              
              {devices.opencl && devices.opencl.length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                    OpenCL Devices ({devices.opencl.length})
                  </div>
                  {devices.opencl.map((device, idx) => (
                    <div
                      key={idx}
                      onClick={() => handleSelectDevice('opencl', device)}
                      style={{
                        padding: '0.5rem',
                        marginBottom: '0.5rem',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(74, 144, 226, 0.3)',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.8rem'
                      }}
                    >
                      <div style={{ fontWeight: '600', color: '#4A90E2' }}>
                        {device.type === 'GPU' ? '🎮' : device.type === 'CPU' ? '💻' : '⚡'} {device.name}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                        {device.platform} • {device.type}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              {devices.cuda && devices.cuda.length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                    CUDA Devices ({devices.cuda.length})
                  </div>
                  {devices.cuda.map((device, idx) => (
                    <div
                      key={idx}
                      onClick={() => handleSelectDevice('cuda', device)}
                      style={{
                        padding: '0.5rem',
                        marginBottom: '0.5rem',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(118, 185, 0, 0.3)',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.8rem'
                      }}
                    >
                      <div style={{ fontWeight: '600', color: '#76B900' }}>
                        🎮 {device.name}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                        {device.platform} • Compute {device.compute_capability}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              {(!devices.opencl || devices.opencl.length === 0) && 
               (!devices.cuda || devices.cuda.length === 0) && (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                  No GPU devices found. Will use CPU fallback.
                </div>
              )}
            </>
          ) : (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Click to discover available devices
            </div>
          )}
        </div>
      )}
    </div>
  )
}
