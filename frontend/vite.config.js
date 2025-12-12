import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/upload': 'http://localhost:8000',
      '/upload-project': 'http://localhost:8000',
      '/port': 'http://localhost:8000',
      '/verify': 'http://localhost:8000',
      '/verify-local': 'http://localhost:8000',
      '/local-projects': 'http://localhost:8000',
      '/analyze-local-project': 'http://localhost:8000',
      '/status': 'http://localhost:8000',
      '/discover-toolchains': 'http://localhost:8000',
      '/discover-devices': 'http://localhost:8000',
      '/discover-adb': 'http://localhost:8000',
      '/discover': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/test-connection': 'http://localhost:8000',
      '/test-toolchain': 'http://localhost:8000',
      '/jobs': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      }
    }
  }
})
