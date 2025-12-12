import { useState, useEffect } from 'react'
import ConnectionConfig from './components/ConnectionConfig'
import ToolchainConfig from './components/ToolchainConfig'
import DeviceSelector from './components/DeviceSelector'

function App() {
  const [sourceFile, setSourceFile] = useState(null)
  const [targetType, setTargetType] = useState('OpenCL')
  const [deviceConfig, setDeviceConfig] = useState({ 
    connection_type: 'local',
    mock: false, 
    use_adb: false, 
    adb_device_id: '',
    remote_work_dir: '/data/local/tmp',
    toolchain: {
      compiler_path: 'gcc',
      sysroot: '',
      include_paths: [],
      library_paths: [],
      compiler_flags: [],
      linker_flags: [],
      architecture: 'x86_64',
      abi: '',
      endianness: 'little'
    },
    compiler_cmd: 'gcc',
    compute_backend: 'auto',
    compute_device_type: 'GPU',
    compute_platform_id: 0,
    compute_device_id: 0
  })
  const [inputShape, setInputShape] = useState('1, 10')
  const [logs, setLogs] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [systemStatus, setSystemStatus] = useState(null)

  const [activeTab, setActiveTab] = useState('auto') // 'auto' or 'manual'
  const [manualSourceFile, setManualSourceFile] = useState(null)
  const [manualRefFile, setManualRefFile] = useState(null)
  const [manualProjectMode, setManualProjectMode] = useState('upload') // 'upload' | 'local'
  const [localProjectPath, setLocalProjectPath] = useState('')
  const [localEntrypoint, setLocalEntrypoint] = useState('')
  const [allowLocalWrite, setAllowLocalWrite] = useState(false)
  const [localPickerOpen, setLocalPickerOpen] = useState(false)
  const [localPickerQuery, setLocalPickerQuery] = useState('')
  const [localPickerLoading, setLocalPickerLoading] = useState(false)
  const [localPickerResults, setLocalPickerResults] = useState([])
  const [localDetectedEntrypoint, setLocalDetectedEntrypoint] = useState(null)
  const [testResult, setTestResult] = useState(null)
  const [testLoading, setTestLoading] = useState(false)
  const [maxIterations, setMaxIterations] = useState(5)
  const [debugInstructions, setDebugInstructions] = useState('')
  const [recentFiles, setRecentFiles] = useState(() => {
    const saved = localStorage.getItem('nnport-recent-files')
    return saved ? JSON.parse(saved) : []
  })

  const addToRecent = (filename, type) => {
    const recent = [{ filename, type, timestamp: Date.now() }]
      .concat(recentFiles.filter(f => f.filename !== filename))
      .slice(0, 5)
    setRecentFiles(recent)
    localStorage.setItem('nnport-recent-files', JSON.stringify(recent))
  }

  // Fetch system status on mount
  useEffect(() => {
    fetch('/status')
      .then(res => res.json())
      .then(data => setSystemStatus(data))
      .catch(err => console.error('Failed to fetch system status:', err))
  }, [])

  const handleUploadGeneric = async (file, setter, type) => {
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)
    try {
      setLoading(true)
      const res = await fetch('/upload', { method: 'POST', body: formData })
      if (!res.ok) throw new Error('Upload failed')
      const data = await res.json()
      setter(data.filename)
      if (type) addToRecent(data.filename, type)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleFolderUpload = async (e) => {
    const files = Array.from(e.target.files)
    if (files.length === 0) return
    
    // Filter for relevant GPU project files
    const validFiles = files.filter(f => {
      const ext = f.name.split('.').pop().toLowerCase()
      return ['cpp', 'c', 'h', 'hpp', 'cu', 'cl', 'cuh', 'metal', 'glsl'].includes(ext)
    })
    
    if (validFiles.length === 0) {
      setError('No valid GPU source files found in folder (.cpp, .c, .h, .cu, .cl, etc.)')
      return
    }
    
    const formData = new FormData()
    // Get folder name from first file's path
    const folderPath = validFiles[0].webkitRelativePath || validFiles[0].name
    const folderName = folderPath.split('/')[0] || 'project'
    formData.append('folder_name', folderName)
    
    // Add all files with their relative paths
    validFiles.forEach(file => {
      const relativePath = file.webkitRelativePath || file.name
      formData.append('files', file, relativePath)
    })
    
    try {
      setLoading(true)
      const res = await fetch('/upload-project', { method: 'POST', body: formData })
      if (!res.ok) throw new Error('Folder upload failed')
      const data = await res.json()
      
      // Set the main source file (first .cpp/.cu/.cl file)
      const mainFile = data.main_file || data.files[0]
      setManualSourceFile(mainFile)
      addToRecent(mainFile, 'cpp')
      
      // Show success message with file count
      setError(null)
      console.log(`✅ Uploaded ${data.files.length} files from project: ${folderName}`)
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
          max_iterations: maxIterations,
          input_shape: inputShape.split(',').map(n => parseInt(n.trim())),
          debug_instructions: debugInstructions,
        }),
      })
      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Porting failed')
      }
      const data = await res.json()
      
      if (data.job_id) {
        // Connect to WebSocket for real-time updates
        const wsUrl = `ws://localhost:8000/ws/port/${data.job_id}`
        const ws = new WebSocket(wsUrl)
        
        // Safety timeout: force reset after 10 minutes
        const timeoutId = setTimeout(() => {
          console.warn('⚠️ Job timeout - forcing reset')
          setLoading(false)
          ws.close()
        }, 600000) // 10 minutes
        
        ws.onopen = () => console.log('WebSocket connected')
        ws.onmessage = (event) => {
          const message = JSON.parse(event.data)
          if (message.type === 'job_complete') {
            clearTimeout(timeoutId)
            setLogs(message.logs || [])
            setLoading(false)
            ws.close()
          } else {
            setLogs(prevLogs => [...(prevLogs || []), message])
          }
        }
        ws.onerror = () => {
          clearTimeout(timeoutId)
          setError('WebSocket connection error')
          setLoading(false)
        }
        ws.onclose = () => {
          clearTimeout(timeoutId)
          console.log('🔌 WebSocket closed')
          // Always reset loading state when connection closes
          setLoading(false)
        }
      } else {
        setLogs(data.logs)
        setLoading(false)
      }
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const handleVerify = async () => {
    if (!manualRefFile) return
    if (manualProjectMode === 'upload' && !manualSourceFile) return
    if (manualProjectMode === 'local' && (!localProjectPath || !allowLocalWrite)) return
    try {
      setLoading(true)
      setError(null)
      setLogs([])

      const payloadCommon = {
        target_type: targetType,
        device_config: deviceConfig,
        input_shape: inputShape.split(',').map(n => parseInt(n.trim())),
        max_iterations: maxIterations,
        debug_instructions: debugInstructions,
      }

      const endpoint = manualProjectMode === 'local' ? '/verify-local' : '/verify'
      const body =
        manualProjectMode === 'local'
          ? {
              ...payloadCommon,
              project_path: localProjectPath.trim(),
              entrypoint: localEntrypoint.trim() || null,
              reference_filename: manualRefFile,
              allow_write: true,
            }
          : {
              ...payloadCommon,
              source_filename: manualSourceFile,
              reference_filename: manualRefFile,
            }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Verification failed')
      }
      const data = await res.json()
      
      if (data.job_id) {
        // Connect to WebSocket for real-time updates
        const wsUrl = `ws://localhost:8000/ws/port/${data.job_id}`
        const ws = new WebSocket(wsUrl)
        
        // Safety timeout: force reset after 10 minutes
        const timeoutId = setTimeout(() => {
          console.warn('⚠️ Job timeout - forcing reset')
          setLoading(false)
          ws.close()
        }, 600000) // 10 minutes
        
        ws.onopen = () => {
          console.log('✅ WebSocket connected for job:', data.job_id)
        }
        
        ws.onmessage = (event) => {
          const message = JSON.parse(event.data)
          console.log('📨 Received WebSocket message:', message)
          
          if (message.type === 'job_complete') {
            clearTimeout(timeoutId)
            console.log('✅ Job complete, final logs:', message.logs)
            setLogs(message.logs || [])
            setLoading(false)
            ws.close()
          } else {
            // It's a log entry, append it
            console.log('📝 Appending log entry:', message)
            setLogs(prevLogs => {
              const newLogs = [...(prevLogs || []), message]
              console.log('📊 Updated logs array length:', newLogs.length)
              return newLogs
            })
          }
        }
        
        ws.onerror = (error) => {
          clearTimeout(timeoutId)
          console.error('❌ WebSocket error:', error)
          setError('WebSocket connection error')
          setLoading(false)
        }
        
        ws.onclose = () => {
          clearTimeout(timeoutId)
          console.log('🔌 WebSocket closed')
          // Always reset loading state when connection closes
          setLoading(false)
        }
      } else {
        setLogs(data.logs)
        setLoading(false)
      }
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const openLocalPicker = async () => {
    try {
      setLocalPickerOpen(true)
      setLocalPickerLoading(true)
      setLocalPickerResults([])
      const res = await fetch(`/local-projects?query=${encodeURIComponent(localPickerQuery || '')}&limit=100`)
      const data = await res.json()
      setLocalPickerResults(data.projects || [])
    } catch (e) {
      setError(e.message || 'Failed to load local projects')
    } finally {
      setLocalPickerLoading(false)
    }
  }

  const analyzeAndSetEntrypoint = async (projectPath) => {
    try {
      const res = await fetch('/analyze-local-project', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_path: projectPath }),
      })
      if (!res.ok) return
      const data = await res.json()
      setLocalDetectedEntrypoint(data.selected_entrypoint || null)
      // Keep the entrypoint field optional; we show the detected one and only fill if user wants.
      if (!localEntrypoint.trim() && data.selected_entrypoint) {
        setLocalEntrypoint('')
      }
    } catch {
      // non-fatal
    }
  }

  const handleTestConnection = async () => {
    try {
      setTestLoading(true)
      setTestResult(null)
      const res = await fetch('/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(deviceConfig),
      })
      const data = await res.json()
      setTestResult(data)
    } catch (err) {
      setTestResult({ success: false, message: 'Test failed', details: err.message })
    } finally {
      setTestLoading(false)
    }
  }

  const handleTestToolchain = async () => {
    try {
      setTestLoading(true)
      setTestResult(null)
      const res = await fetch('/test-toolchain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(deviceConfig),
      })
      const data = await res.json()
      setTestResult(data)
    } catch (err) {
      setTestResult({ success: false, message: 'Test failed', details: err.message })
    } finally {
      setTestLoading(false)
    }
  }

  const handleExportConfig = () => {
    // Generate smart filename
    const toolchainName = deviceConfig.toolchain?.compiler_path?.split('/').pop()?.split('-')[0] || 'unknown'
    const arch = deviceConfig.toolchain?.architecture || 'x64'
    const connection = deviceConfig.connection_type || 'local'
    const timestamp = new Date().toISOString().split('T')[0]
    const filename = `nnport-${toolchainName}-${arch}-${connection}-${timestamp}.json`
    
    const json = JSON.stringify(deviceConfig, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleImportConfig = (event) => {
    const file = event.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const imported = JSON.parse(e.target.result)
          setDeviceConfig(imported)
          alert('Configuration loaded successfully!')
        } catch (error) {
          alert('Failed to import config: ' + error.message)
        }
      }
      reader.readAsText(file)
    }
  }

  return (
    <div className="layout">
      {/* Sidebar Configuration */}
      <aside className="sidebar">
        <h1 className="section-title">NNPort</h1>

        {/* Config Management */}
        <div style={{ marginBottom: '1.5rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
          <button 
            onClick={handleExportConfig}
            style={{ fontSize: '0.85rem', padding: '0.5rem', background: '#4A90E2', border: 'none', color: 'white', cursor: 'pointer', borderRadius: 'var(--radius)' }}
          >
            💾 Export Config
          </button>
          <label style={{ 
            fontSize: '0.85rem', 
            padding: '0.5rem', 
            borderRadius: 'var(--radius)',
            cursor: 'pointer',
            textAlign: 'center',
            background: '#4A90E2',
            border: 'none',
            color: 'white',
            margin: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            📂 Import Config
            <input
              type="file"
              accept=".json"
              onChange={handleImportConfig}
              style={{ display: 'none' }}
            />
          </label>
        </div>

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
            Auto Port
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
          <div
            style={{
              padding: '0.75rem',
              borderRadius: 'var(--radius)',
              border: '1px solid var(--border)',
              background: 'rgba(245, 158, 11, 0.08)',
              color: 'var(--text-secondary)',
              marginBottom: '1rem',
              pointerEvents: 'none',
            }}
          >
            <div style={{ fontWeight: 800, color: '#F59E0B', marginBottom: '0.35rem', letterSpacing: '0.2px' }}>
              Coming soon
            </div>
            <div style={{ fontSize: '0.85rem' }}>
              Auto Port is disabled in this build.
            </div>
          </div>
        ) : (
          <>
            {/* Project mode switcher */}
            <div className="control-group">
              <label className="control-label">Project Mode</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  type="button"
                  onClick={() => {
                    setManualProjectMode('upload')
                    setAllowLocalWrite(false)
                    setError(null)
                  }}
                  style={{
                    flex: 1,
                    padding: '0.6rem',
                    borderRadius: 'var(--radius)',
                    border: '1px solid var(--border)',
                    background: manualProjectMode === 'upload' ? 'rgba(255,255,255,0.12)' : 'transparent',
                    cursor: 'pointer',
                    fontWeight: 600,
                  }}
                >
                  Upload folder (safe)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setManualProjectMode('local')
                    setError(null)
                  }}
                  style={{
                    flex: 1,
                    padding: '0.6rem',
                    borderRadius: 'var(--radius)',
                    border: '1px solid #ff6b6b',
                    background: manualProjectMode === 'local' ? 'rgba(255, 107, 107, 0.12)' : 'transparent',
                    cursor: 'pointer',
                    fontWeight: 700,
                    color: manualProjectMode === 'local' ? '#ff6b6b' : 'var(--text-primary)',
                  }}
                >
                  Local path (edits your project)
                </button>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.4rem' }}>
                Upload mode edits a temp copy. Local path mode edits your real folder and keeps backups under <code>.nnport_backups/</code>.
              </div>
            </div>

            <div className="control-group">
              <label className="control-label">
                GPU Project / Source Code
                <span style={{ fontSize: '0.75rem', marginLeft: '0.5rem', color: 'var(--text-secondary)' }}>
                  (Upload single file OR entire folder)
                </span>
              </label>
              
              {manualProjectMode === 'upload' ? (
                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  {/* Single file upload */}
                  <label style={{ 
                    flex: 1,
                    padding: '0.5rem',
                    borderRadius: 'var(--radius)',
                    cursor: 'pointer',
                    textAlign: 'center',
                    background: 'rgba(255,255,255,0.1)',
                    border: '1px solid var(--border)',
                    fontSize: '0.85rem'
                  }}>
                    📄 Upload File
                    <input
                      type="file"
                      accept=".cpp,.c,.cl,.cu,.h,.hpp"
                      onChange={(e) => handleUploadGeneric(e.target.files[0], setManualSourceFile, 'cpp')}
                      style={{ display: 'none' }}
                    />
                  </label>
                  
                  {/* Folder upload */}
                  <label style={{ 
                    flex: 1,
                    padding: '0.5rem',
                    borderRadius: 'var(--radius)',
                    cursor: 'pointer',
                    textAlign: 'center',
                    background: 'rgba(106, 90, 205, 0.2)',
                    border: '1px solid #6A5ACD',
                    fontSize: '0.85rem',
                    color: '#A494F0'
                  }}>
                    📁 Upload Folder
                    <input
                      type="file"
                      webkitdirectory=""
                      directory=""
                      multiple
                      onChange={handleFolderUpload}
                      style={{ display: 'none' }}
                    />
                  </label>
                </div>
              ) : (
                <div style={{ marginBottom: '0.75rem' }}>
                  <div style={{
                    padding: '0.75rem',
                    borderRadius: 'var(--radius)',
                    border: '1px solid #ff6b6b',
                    background: 'rgba(255, 107, 107, 0.08)',
                    marginBottom: '0.6rem',
                    fontSize: '0.85rem',
                    lineHeight: 1.35
                  }}>
                    <div style={{ fontWeight: 800, marginBottom: '0.25rem' }}>Edits your real project folder</div>
                    <div style={{ color: 'var(--text-secondary)' }}>
                      NNPort will write fixes directly into the folder you specify and store rollback backups in <code>.nnport_backups/</code>.
                      <br />
                      Tip (macOS): Finder → right click folder → hold Option → “Copy as Pathname”.
                    </div>
                  </div>

                  <label className="control-label" style={{ marginTop: '0.25rem' }}>Project path</label>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <input
                      type="text"
                      placeholder="Click Find… to pick a local folder"
                      value={localProjectPath}
                      onChange={(e) => setLocalProjectPath(e.target.value)}
                      style={{ fontFamily: 'monospace', fontSize: '0.85rem', flex: 1 }}
                    />
                    <button
                      type="button"
                      onClick={openLocalPicker}
                      style={{
                        padding: '0.55rem 0.75rem',
                        borderRadius: 'var(--radius)',
                        border: '1px solid var(--border)',
                        background: 'rgba(255,255,255,0.08)',
                        cursor: 'pointer',
                        fontWeight: 700,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      Find…
                    </button>
                  </div>

                  {localPickerOpen && (
                    <div style={{
                      marginTop: '0.6rem',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius)',
                      padding: '0.75rem',
                      background: 'rgba(0,0,0,0.15)'
                    }}>
                      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                        <input
                          type="text"
                          placeholder="Filter (e.g. examples, myproj)…"
                          value={localPickerQuery}
                          onChange={(e) => setLocalPickerQuery(e.target.value)}
                          style={{ flex: 1, fontFamily: 'monospace', fontSize: '0.85rem' }}
                        />
                        <button
                          type="button"
                          onClick={openLocalPicker}
                          disabled={localPickerLoading}
                          style={{ padding: '0.5rem 0.75rem', fontWeight: 700 }}
                        >
                          {localPickerLoading ? 'Searching…' : 'Search'}
                        </button>
                        <button
                          type="button"
                          onClick={() => setLocalPickerOpen(false)}
                          style={{ padding: '0.5rem 0.75rem' }}
                        >
                          Close
                        </button>
                      </div>

                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                        Showing projects from <code>~/Projects</code> on the backend host.
                      </div>

                      <div style={{ maxHeight: '220px', overflow: 'auto', borderTop: '1px solid var(--border)', paddingTop: '0.5rem' }}>
                        {localPickerLoading && <div style={{ fontSize: '0.85rem' }}>Loading…</div>}
                        {!localPickerLoading && localPickerResults.length === 0 && (
                          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                            No results. Try a different filter.
                          </div>
                        )}
                        {!localPickerLoading && localPickerResults.map((p) => (
                          <div
                            key={p}
                            onClick={async () => {
                              setLocalProjectPath(p)
                              setLocalPickerOpen(false)
                              setError(null)
                              await analyzeAndSetEntrypoint(p)
                            }}
                            style={{
                              cursor: 'pointer',
                              padding: '0.35rem 0.25rem',
                              fontFamily: 'monospace',
                              fontSize: '0.8rem',
                              color: 'var(--accent)',
                              borderRadius: '6px',
                            }}
                          >
                            {p}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <label className="control-label" style={{ marginTop: '0.5rem' }}>
                    Entrypoint (optional)
                    <span style={{ fontSize: '0.75rem', marginLeft: '0.5rem', color: 'var(--text-secondary)' }}>
                      (relative path, leave blank to auto-detect main())
                    </span>
                  </label>
                  <input
                    type="text"
                    placeholder="src/main.cpp"
                    value={localEntrypoint}
                    onChange={(e) => setLocalEntrypoint(e.target.value)}
                    style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}
                  />
                  {localDetectedEntrypoint && !localEntrypoint.trim() && (
                    <div style={{ marginTop: '0.35rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      Detected entrypoint: <code>{localDetectedEntrypoint}</code>
                    </div>
                  )}

                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.6rem' }}>
                    <input
                      type="checkbox"
                      checked={allowLocalWrite}
                      onChange={(e) => setAllowLocalWrite(e.target.checked)}
                    />
                    <span style={{ fontSize: '0.85rem' }}>
                      I understand NNPort will modify files in this folder
                    </span>
                  </label>
                </div>
              )}
              
              {manualSourceFile && <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--success)' }}>✓ {manualSourceFile}</div>}
              {recentFiles.filter(f => f.type === 'cpp').slice(0, 3).length > 0 && (
                <div style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
                  <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Recent:</div>
                  {recentFiles.filter(f => f.type === 'cpp').slice(0, 3).map((f, i) => (
                    <div 
                      key={i} 
                      onClick={() => setManualSourceFile(f.filename)}
                      style={{ 
                        cursor: 'pointer', 
                        color: manualSourceFile === f.filename ? 'var(--success)' : 'var(--accent)', 
                        padding: '0.15rem 0',
                        textDecoration: manualSourceFile === f.filename ? 'underline' : 'none'
                      }}
                    >
                      • {f.filename}
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="control-group">
              <label className="control-label">Reference Model (.py)</label>
              <input
                type="file"
                accept=".py"
                onChange={(e) => handleUploadGeneric(e.target.files[0], setManualRefFile, 'py')}
              />
              {manualRefFile && <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--success)' }}>✓ {manualRefFile}</div>}
              {recentFiles.filter(f => f.type === 'py').slice(0, 3).length > 0 && (
                <div style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
                  <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Recent:</div>
                  {recentFiles.filter(f => f.type === 'py').slice(0, 3).map((f, i) => (
                    <div 
                      key={i} 
                      onClick={() => setManualRefFile(f.filename)}
                      style={{ 
                        cursor: 'pointer', 
                        color: manualRefFile === f.filename ? 'var(--success)' : 'var(--accent)', 
                        padding: '0.15rem 0',
                        textDecoration: manualRefFile === f.filename ? 'underline' : 'none'
                      }}
                    >
                      • {f.filename}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}

        <div className="control-group">
          <label className="control-label">Target Hardware</label>
          <select
            value={targetType}
            onChange={(e) => setTargetType(e.target.value)}
          >
            <option value="DSP" disabled style={{ opacity: 0.55 }}>
              DSP (C6x) (coming soon)
            </option>
            <option value="CUDA" disabled style={{ opacity: 0.55 }}>
              NVIDIA CUDA (coming soon)
            </option>
            <option value="OpenCL">OpenCL</option>
          </select>
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

        {/* Vibe Debugging Configuration */}
        <div style={{ 
          marginBottom: '1.5rem', 
          border: '2px solid #9B59B6', 
          borderRadius: 'var(--radius)', 
          padding: '1rem',
          background: 'rgba(155, 89, 182, 0.05)'
        }}>
          <div style={{ 
            fontSize: '0.9rem', 
            fontWeight: '700', 
            color: '#9B59B6', 
            marginBottom: '0.75rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            🔄 Iterative Debugging
          </div>

          <div className="control-group" style={{ marginBottom: '0.75rem' }}>
            <label className="control-label">Max Iterations</label>
            <input
              type="number"
              min="1"
              max="20"
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value) || 5)}
              style={{ width: '100%' }}
            />
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
              Number of cross-compile → deploy → test → fix cycles (1-20, default: 5)
            </div>
          </div>

          <div className="control-group">
            <label className="control-label">Debug Instructions (optional)</label>
            <textarea
              rows="4"
              placeholder="Guide the AI debugging process...&#10;&#10;Example:&#10;- Focus on fixing precision issues&#10;- The output is off by a factor of 2&#10;- Check memory alignment on target device"
              value={debugInstructions}
              onChange={(e) => setDebugInstructions(e.target.value)}
              style={{ 
                width: '100%', 
                fontSize: '0.85rem',
                fontFamily: 'monospace',
                resize: 'vertical'
              }}
            />
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
              Additional context to help AI fix mismatches between host and target
            </div>
          </div>
        </div>

        {/* Target Device Configuration */}
        <div style={{ 
          border: '2px solid #50C878', 
          borderRadius: 'var(--radius)', 
          padding: '0.75rem',
          marginBottom: '1rem',
          background: 'rgba(80, 200, 120, 0.03)'
        }}>
          <div style={{ 
            fontSize: '0.85rem', 
            fontWeight: '700', 
            color: '#50C878', 
            marginBottom: '0.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            🎯 TARGET DEVICE
          </div>
          <ConnectionConfig 
            config={deviceConfig} 
            onChange={setDeviceConfig} 
          />
        </div>

        {/* Host Toolchain Configuration */}
        <div style={{ 
          border: '2px solid #4A90E2', 
          borderRadius: 'var(--radius)', 
          padding: '0.75rem',
          marginBottom: '1rem',
          background: 'rgba(74, 144, 226, 0.03)'
        }}>
          <div style={{ 
            fontSize: '0.85rem', 
            fontWeight: '700', 
            color: '#4A90E2', 
            marginBottom: '0.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            ⚙️ HOST TOOLCHAIN (Cross-Compilation)
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
            Compiler and tools on your local machine used to build code for target
          </div>
          <ToolchainConfig 
            config={deviceConfig} 
            onChange={setDeviceConfig} 
          />
          
          <DeviceSelector
            targetType={targetType}
            config={deviceConfig}
            onChange={setDeviceConfig}
          />
        </div>

        {/* Testing Buttons */}
        <div style={{ marginBottom: '1.5rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
          <button
            onClick={handleTestConnection}
            disabled={testLoading}
            style={{ fontSize: '0.85rem', padding: '0.5rem' }}
          >
            {testLoading ? 'Testing...' : 'Test Connection'}
          </button>
          <button
            onClick={handleTestToolchain}
            disabled={testLoading}
            style={{ fontSize: '0.85rem', padding: '0.5rem' }}
          >
            {testLoading ? 'Testing...' : 'Test Toolchain'}
          </button>
        </div>

        {testResult && (
          <div style={{ 
            marginBottom: '1.5rem', 
            padding: '0.75rem', 
            borderRadius: 'var(--radius)',
            background: testResult.success ? 'rgba(0, 255, 0, 0.1)' : 'rgba(255, 0, 0, 0.1)',
            border: `1px solid ${testResult.success ? 'var(--success)' : 'var(--error)'}`,
            fontSize: '0.85rem'
          }}>
            <div style={{ fontWeight: '600', marginBottom: '0.25rem', color: testResult.success ? 'var(--success)' : 'var(--error)' }}>
              {testResult.message}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {testResult.details}
            </div>
          </div>
        )}

        {activeTab === 'auto' ? (
          <button
            disabled={true}
            style={{
              width: '100%',
              padding: '0.75rem',
              fontSize: '1rem',
              fontWeight: '600',
              opacity: 0.75,
              cursor: 'not-allowed',
              color: '#F59E0B',
            }}
            title="Coming soon"
          >
            ⏳ Auto Port (coming soon)
          </button>
        ) : (
          <button
            onClick={handleVerify}
            disabled={
              loading ||
              !manualRefFile ||
              (manualProjectMode === 'upload' && !manualSourceFile) ||
              (manualProjectMode === 'local' && (!localProjectPath.trim() || !allowLocalWrite))
            }
            style={{ width: '100%', padding: '0.75rem', fontSize: '1rem', fontWeight: '600' }}
          >
            {loading
              ? `🔄 Vibing... (0/${maxIterations})`
              : manualProjectMode === 'local'
                ? '🛠️ Vibe Debug (Local Project)'
                : '🔧 Vibe Debug Manual Code'}
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
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              {/* HOST Side */}
              <div>
                <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--accent)' }}>🖥️ HOST (Reference)</h3>
                <div className="logs-container">
                  {loading && !logs && <div>Initializing...</div>}
                  {logs && logs.filter(log => log.stage?.includes('HOST') || log.stage?.includes('Reference') || log.stage?.includes('Initialization') || log.stage?.includes('Source Code')).map((log, i) => (
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
                              background: 'var(--accent)',
                              border: 'none',
                              borderRadius: 'var(--radius)',
                              cursor: 'pointer'
                            }}
                          >
                            Download
                          </button>
                        )}
                      </div>
                      {log.details && (
                        <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
                          {log.details}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* TARGET Side */}
              <div>
                <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--accent)' }}>📱 TARGET (Device)</h3>
                <div className="logs-container">
                  {logs && logs.filter(log => log.stage?.includes('TARGET') || log.stage?.includes('Iteration') || log.stage?.includes('Compil') || log.stage?.includes('Result') || log.stage?.includes('Device Output')).map((log, i) => (
                    <div key={i} style={{ marginBottom: '1rem', borderLeft: log.status === 'Success' ? '3px solid var(--success)' : log.stage === 'Device Output' ? '3px solid #00d4aa' : '3px solid var(--border)', paddingLeft: '1rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <span style={{ fontWeight: '600', color: log.stage === 'Device Output' ? '#00d4aa' : 'var(--accent)' }}>{log.stage}</span>
                          {log.execution_time && <span style={{ marginLeft: '0.5rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>⏱️ {log.execution_time}</span>}
                          <span style={{
                            marginLeft: '1rem',
                            color: log.status === 'Success' ? 'var(--success)' :
                              log.status === 'Failed' ? 'var(--error)' : 
                              log.status === 'Info' ? '#00d4aa' : 'var(--text-primary)'
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
                              background: 'var(--accent)',
                              border: 'none',
                              borderRadius: 'var(--radius)',
                              cursor: 'pointer'
                            }}
                          >
                            Download
                          </button>
                        )}
                      </div>
                      {log.details && (
                        <div style={{ 
                          marginTop: '0.5rem', 
                          fontSize: log.stage === 'Device Output' ? '0.8rem' : '0.85rem', 
                          color: log.stage === 'Device Output' ? '#00ff00' : 'var(--text-secondary)', 
                          whiteSpace: 'pre-wrap',
                          backgroundColor: log.stage === 'Device Output' ? '#0a0a0a' : 'transparent',
                          padding: log.stage === 'Device Output' ? '0.75rem' : '0',
                          borderRadius: log.stage === 'Device Output' ? '4px' : '0',
                          fontFamily: log.stage === 'Device Output' ? 'monospace' : 'inherit',
                          maxHeight: log.stage === 'Device Output' ? '400px' : 'none',
                          overflowY: log.stage === 'Device Output' ? 'auto' : 'visible',
                          border: log.stage === 'Device Output' ? '1px solid #00d4aa' : 'none'
                        }}>
                          {log.details}
                        </div>
                      )}
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
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
