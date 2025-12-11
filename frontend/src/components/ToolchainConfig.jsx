import { useState } from 'react'

export default function ToolchainConfig({ config, onChange }) {
  const [expanded, setExpanded] = useState(false)
  const [discovering, setDiscovering] = useState(false)
  const [discoveredToolchains, setDiscoveredToolchains] = useState(null)
  const [showDiscoveryResults, setShowDiscoveryResults] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const handleChange = (field, value) => {
    const toolchain = { ...(config.toolchain || {}), [field]: value }
    onChange({ ...config, toolchain })
  }

  const handleArrayChange = (field, text) => {
    const values = text.split('\n').map(line => line.trim()).filter(line => line)
    handleChange(field, values)
  }

  const toolchain = config.toolchain || {}

  const handleDiscoverToolchains = async () => {
    try {
      setDiscovering(true)
      const res = await fetch('/discover-toolchains')
      
      // Check if response is OK
      if (!res.ok) {
        throw new Error(`Server error: ${res.status} ${res.statusText}`)
      }
      
      // Get response text first to debug
      const text = await res.text()
      
      // Try to parse as JSON
      let data
      try {
        data = JSON.parse(text)
      } catch (jsonErr) {
        console.error('Invalid JSON response:', text)
        throw new Error('Server returned invalid JSON. Check backend logs.')
      }
      
      setDiscoveredToolchains(data)
      setShowDiscoveryResults(true)
    } catch (err) {
      console.error('Failed to discover toolchains:', err)
      alert('Failed to discover toolchains: ' + err.message)
    } finally {
      setDiscovering(false)
    }
  }

  const handleSelectToolchain = (tc) => {
    const newToolchain = {
      compiler_path: tc.compiler_path || '',
      sysroot: tc.sysroot || '',
      include_paths: tc.include_paths || [],
      library_paths: tc.library_paths || [],
      compiler_flags: [],
      linker_flags: [],
      architecture: tc.architecture || 'x86_64',
      abi: tc.abi || '',
      endianness: 'little'
    }
    // Update the entire config with the new toolchain at once
    onChange({ ...config, toolchain: newToolchain })
    setShowDiscoveryResults(false)
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
        <h3 style={{ margin: 0, fontSize: '0.9rem' }}>Toolchain Settings</h3>
        <span style={{ fontSize: '0.8rem' }}>{expanded ? '▼' : '▶'}</span>
      </div>

      {expanded && (
        <>
          <button 
            onClick={handleDiscoverToolchains}
            disabled={discovering}
            style={{ 
              width: '100%', 
              marginBottom: '1rem', 
              fontSize: '0.85rem',
              padding: '0.5rem',
              background: '#4A90E2',
              border: 'none',
              color: 'white',
              cursor: discovering ? 'wait' : 'pointer'
            }}
          >
            {discovering ? '🔍 Discovering...' : '🔍 Find Toolchains on Host'}
          </button>

          {showDiscoveryResults && discoveredToolchains && (
            <div style={{ 
              marginBottom: '1rem', 
              padding: '0.75rem', 
              background: 'rgba(74, 144, 226, 0.1)',
              borderRadius: 'var(--radius)',
              maxHeight: '400px',
              overflowY: 'auto'
            }}>
              <div style={{ fontSize: '0.85rem', fontWeight: '600', marginBottom: '0.75rem' }}>
                Discovered Toolchains
              </div>
              
              {discoveredToolchains.toolchains ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {/* Native Toolchains */}
                  {discoveredToolchains.toolchains.native && discoveredToolchains.toolchains.native.length > 0 && (
                    <div>
                      <div style={{ fontSize: '0.8rem', fontWeight: '600', marginBottom: '0.5rem', color: '#FFB84D' }}>
                        💻 Native (Host: {discoveredToolchains.host_info?.architecture})
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {discoveredToolchains.toolchains.native.map((tc, idx) => (
                          <div 
                            key={`native-${idx}`}
                            onClick={() => handleSelectToolchain(tc)}
                            style={{ 
                              padding: '0.5rem', 
                              background: 'rgba(255, 184, 77, 0.1)',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              border: '1px solid rgba(255, 184, 77, 0.3)',
                              fontSize: '0.8rem'
                            }}
                          >
                            <div style={{ fontWeight: '600', color: '#FFB84D' }}>{tc.name}</div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                              {tc.compiler_path}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Android NDK Toolchains */}
                  {discoveredToolchains.toolchains.android_ndk && discoveredToolchains.toolchains.android_ndk.length > 0 && (
                    <div>
                      <div style={{ fontSize: '0.8rem', fontWeight: '600', marginBottom: '0.5rem', color: '#A4C639' }}>
                        🤖 Android NDK
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {discoveredToolchains.toolchains.android_ndk.map((tc, idx) => (
                          <div 
                            key={`ndk-${idx}`}
                            onClick={() => handleSelectToolchain(tc)}
                            style={{ 
                              padding: '0.5rem', 
                              background: 'rgba(164, 198, 57, 0.1)',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              border: '1px solid rgba(164, 198, 57, 0.3)',
                              fontSize: '0.8rem'
                            }}
                          >
                            <div style={{ fontWeight: '600', color: '#A4C639' }}>{tc.name}</div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                              {tc.compiler_path}
                            </div>
                            {tc.architecture && (
                              <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                                Arch: {tc.architecture}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Cross-Compilation Toolchains */}
                  {discoveredToolchains.toolchains.cross && discoveredToolchains.toolchains.cross.length > 0 && (
                    <div>
                      <div style={{ fontSize: '0.8rem', fontWeight: '600', marginBottom: '0.5rem', color: '#4A90E2' }}>
                        🔄 Cross-Compilation (ARM, MIPS, RISC-V, etc.)
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {discoveredToolchains.toolchains.cross.map((tc, idx) => (
                          <div 
                            key={`cross-${idx}`}
                            onClick={() => handleSelectToolchain(tc)}
                            style={{ 
                              padding: '0.5rem', 
                              background: 'rgba(74, 144, 226, 0.1)',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              border: '1px solid rgba(74, 144, 226, 0.3)',
                              fontSize: '0.8rem'
                            }}
                          >
                            <div style={{ fontWeight: '600', color: '#4A90E2' }}>{tc.name}</div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                              {tc.compiler_path}
                            </div>
                            {tc.architecture && (
                              <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                                Arch: {tc.architecture} {tc.abi ? `(${tc.abi})` : ''}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Embedded Toolchains */}
                  {discoveredToolchains.toolchains.embedded && discoveredToolchains.toolchains.embedded.length > 0 && (
                    <div>
                      <div style={{ fontSize: '0.8rem', fontWeight: '600', marginBottom: '0.5rem', color: '#E85D75' }}>
                        🔌 Embedded (Bare-metal)
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {discoveredToolchains.toolchains.embedded.map((tc, idx) => (
                          <div 
                            key={`embedded-${idx}`}
                            onClick={() => handleSelectToolchain(tc)}
                            style={{ 
                              padding: '0.5rem', 
                              background: 'rgba(232, 93, 117, 0.1)',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              border: '1px solid rgba(232, 93, 117, 0.3)',
                              fontSize: '0.8rem'
                            }}
                          >
                            <div style={{ fontWeight: '600', color: '#E85D75' }}>{tc.name}</div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                              {tc.compiler_path}
                            </div>
                            {tc.architecture && (
                              <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                                Arch: {tc.architecture}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* No toolchains found */}
                  {(!discoveredToolchains.toolchains.native || discoveredToolchains.toolchains.native.length === 0) &&
                   (!discoveredToolchains.toolchains.cross || discoveredToolchains.toolchains.cross.length === 0) &&
                   (!discoveredToolchains.toolchains.android_ndk || discoveredToolchains.toolchains.android_ndk.length === 0) &&
                   (!discoveredToolchains.toolchains.embedded || discoveredToolchains.toolchains.embedded.length === 0) && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      No toolchains found. Install a toolchain or manually configure.
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  No toolchains found. Install a toolchain or manually configure.
                </div>
              )}

              {discoveredToolchains.gpu_sdks && (
                <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(74, 144, 226, 0.3)' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: '600', marginBottom: '0.5rem' }}>GPU SDKs</div>
                  {discoveredToolchains.gpu_sdks.cuda && (
                    <div style={{ fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                      ✓ CUDA: {discoveredToolchains.gpu_sdks.cuda.path}
                    </div>
                  )}
                  {discoveredToolchains.gpu_sdks.opencl && (
                    <div style={{ fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                      ✓ OpenCL headers found
                    </div>
                  )}
                  {discoveredToolchains.gpu_sdks.rocm && (
                    <div style={{ fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                      ✓ ROCm: {discoveredToolchains.gpu_sdks.rocm.path}
                    </div>
                  )}
                </div>
              )}

              {discoveredToolchains.ndk && (
                <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(74, 144, 226, 0.3)' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: '600', marginBottom: '0.25rem' }}>
                    ✓ Android NDK {discoveredToolchains.ndk.version}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    {discoveredToolchains.ndk.path}
                  </div>
                </div>
              )}

              <button 
                onClick={() => setShowDiscoveryResults(false)}
                style={{ 
                  width: '100%', 
                  marginTop: '0.75rem',
                  fontSize: '0.8rem',
                  padding: '0.4rem',
                  background: 'transparent',
                  border: '1px solid rgba(74, 144, 226, 0.5)'
                }}
              >
                Close
              </button>
            </div>
          )}

          <label className="control-label">Compiler Path (on host)</label>
          <input
            type="text"
            placeholder="/usr/bin/gcc"
            value={toolchain.compiler_path || 'gcc'}
            onChange={(e) => handleChange('compiler_path', e.target.value)}
          />

          <label className="control-label">Sysroot (host filesystem, optional)</label>
          <input
            type="text"
            placeholder="/opt/sysroot/arm-linux-gnueabihf"
            value={toolchain.sysroot || ''}
            onChange={(e) => handleChange('sysroot', e.target.value)}
          />

          <label className="control-label">Architecture</label>
          <select
            value={toolchain.architecture || 'x86_64'}
            onChange={(e) => handleChange('architecture', e.target.value)}
          >
            <option value="x86_64">x86_64</option>
            <option value="arm">ARM (32-bit)</option>
            <option value="aarch64">ARM64 (64-bit)</option>
            <option value="mips">MIPS</option>
            <option value="riscv">RISC-V</option>
          </select>

          <label className="control-label">ABI</label>
          <select
            value={toolchain.abi || ''}
            onChange={(e) => handleChange('abi', e.target.value)}
          >
            <option value="">Default</option>
            <option value="hard-float">Hard Float</option>
            <option value="soft-float">Soft Float</option>
          </select>

          <label className="control-label">Endianness</label>
          <select
            value={toolchain.endianness || 'little'}
            onChange={(e) => handleChange('endianness', e.target.value)}
          >
            <option value="little">Little Endian</option>
            <option value="big">Big Endian</option>
          </select>

          <div 
            style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              cursor: 'pointer',
              marginTop: '1rem',
              marginBottom: '0.5rem',
              padding: '0.5rem',
              background: 'rgba(74, 144, 226, 0.05)',
              borderRadius: '4px'
            }}
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <span style={{ fontSize: '0.85rem', fontWeight: '600' }}>Advanced Options (Optional)</span>
            <span style={{ fontSize: '0.8rem' }}>{showAdvanced ? '▼' : '▶'}</span>
          </div>

          {showAdvanced && (
            <>
              <label className="control-label">Include Paths (host, one per line)</label>
              <textarea
                rows="3"
                placeholder="/usr/include&#10;/opt/includes"
                value={(toolchain.include_paths || []).join('\n')}
                onChange={(e) => handleArrayChange('include_paths', e.target.value)}
              />

              <label className="control-label">Library Paths (host, one per line)</label>
              <textarea
                rows="3"
                placeholder="/usr/lib&#10;/opt/lib"
                value={(toolchain.library_paths || []).join('\n')}
                onChange={(e) => handleArrayChange('library_paths', e.target.value)}
              />

              <label className="control-label">Compiler Flags</label>
              <textarea
                rows="2"
                placeholder="-march=armv7-a -mfpu=neon"
                value={(toolchain.compiler_flags || []).join(' ')}
                onChange={(e) => handleChange('compiler_flags', e.target.value.split(/\s+/).filter(f => f))}
              />

              <label className="control-label">Linker Flags</label>
              <textarea
                rows="2"
                placeholder="-lpthread -lm"
                value={(toolchain.linker_flags || []).join(' ')}
                onChange={(e) => handleChange('linker_flags', e.target.value.split(/\s+/).filter(f => f))}
              />
            </>
          )}
        </>
      )}
    </div>
  )
}
