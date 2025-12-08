# NNPort

A web application for porting and debugging neural network architectures.

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- ADB (optional, for Android porting)

### 1. Setup Backend
```bash
# Create virtual env
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run Server
# IMPORTANT: Run from the project root (NNPort directory), NOT inside backend/
uvicorn backend.main:app --reload --port 8000
```

### 2. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```

### 3. Usage
- Open `http://localhost:5173`
- Upload a source PyTorch model (must define `model` variable).
- Select Target (DSP, CUDA, OpenCL).
- Configure ADB if needed.
- Click "Start Porting".
