# PyTorch Tensor Marathon - Web Application

A modern web interface for practicing PyTorch tensor operations with real-time feedback.

## Features

- 🎨 **Beautiful UI**: Modern dark theme with glassmorphism effects
- 🌐 **Bilingual**: Switch between Japanese and English seamlessly
- ⚡ **Real-time Validation**: Instant feedback on your solutions
- 📊 **Progress Tracking**: Track solved problems with localStorage
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 💡 **Hints & Solutions**: Get help when you're stuck
- ✨ **Syntax Highlighting**: Prism.js for beautiful code display

## Quick Start

### Running Locally

```bash
# From project root
cd /path/to/torch_tensor_marathon

# Start the server
uv run uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload

# Open in browser
open http://localhost:8000
```

### Production Mode

```bash
# Run with multiple workers
uv run uvicorn web.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Architecture

### Backend (FastAPI)

**File**: `web/app.py`

**API Endpoints**:
- `GET /` - Serve main HTML page
- `GET /api/categories` - List all problem categories with stats
- `GET /api/problems/{category}` - Get problems for a category
- `GET /api/problem/{problem_id}` - Get problem details
- `POST /api/check` - Check user's solution
- `GET /api/solution/{problem_id}` - Get expected solution
- `GET /api/stats` - Get overall statistics

**Key Features**:
- CORS enabled for development
- Reuses existing `CorrectnessChecker` and `ProblemBank`
- JSON-based API responses

### Frontend

**Files**:
- `web/static/index.html` - Main HTML structure
- `web/static/style.css` - Styling with CSS variables and animations
- `web/static/app.js` - Application logic and state management
- `web/static/i18n.js` - Internationalization translations

**State Management**:
```javascript
const state = {
    currentLang: 'en',
    currentCategory: null,
    currentProblems: [],
    currentProblem: null,
    currentProblemIndex: 0,
    solvedProblems: new Set(),
    filteredDifficulty: 'all',
};
```

**Key Features**:
- Vanilla JavaScript (no frameworks)
- LocalStorage for progress persistence
- Prism.js for syntax highlighting
- Responsive grid layouts

## Deployment

### Docker (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync

EXPOSE 8000
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t tensor-marathon-web .
docker run -p 8000:8000 tensor-marathon-web
```

### Cloud Deployment

**Recommended platforms**:
- **Railway**: Connect GitHub repo, auto-deploy
- **Render**: Free tier available, easy setup
- **Google Cloud Run**: Serverless, auto-scaling
- **Heroku**: Simple deployment with Procfile

**Environment Variables**:
- None required for basic operation
- Optional: `GEMINI_API_KEY` for dynamic problem generation

## Development

### File Structure

```
web/
├── app.py              # FastAPI application
└── static/
    ├── index.html      # Main HTML
    ├── style.css       # Styles
    ├── app.js          # Application logic
    └── i18n.js         # Translations
```

### Adding Features

1. **New API Endpoint**: Edit `web/app.py`
2. **UI Changes**: Edit `web/static/index.html` and `web/static/style.css`
3. **Logic Changes**: Edit `web/static/app.js`
4. **Translations**: Edit `web/static/i18n.js`

### Debugging

Enable hot reload:
```bash
uv run uvicorn web.app:app --reload --log-level debug
```

Check browser console for frontend errors.

## API Documentation

### Check Solution

**Endpoint**: `POST /api/check`

**Request**:
```json
{
  "problem_id": "reshape_001",
  "user_code": "result = x.reshape(2, 6)"
}
```

**Response**:
```json
{
  "is_correct": true,
  "message": "Correct! Shape matches expected result.",
  "error_type": null,
  "expected_shape": null,
  "actual_shape": null
}
```

### Get Problem Details

**Endpoint**: `GET /api/problem/{problem_id}`

**Response**:
```json
{
  "id": "reshape_001",
  "category": "reshape_permute",
  "difficulty": "beginner",
  "title_ja": "...",
  "title_en": "...",
  "description_ja": "...",
  "description_en": "...",
  "setup_code": "import torch\nx = torch.randn(3, 4)",
  "hint_ja": "...",
  "hint_en": "..."
}
```

## Troubleshooting

### Port Already in Use

```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Problems Not Loading

Check that `initialize_problems()` is called in `web/app.py`:
```python
problem_bank = ProblemBank()
initialize_problems(problem_bank)
```

### CORS Errors

CORS is enabled by default. If issues persist, check browser console and verify `CORSMiddleware` configuration in `web/app.py`.

## Performance

- **Static Files**: Served directly by FastAPI
- **Caching**: Browser caches static assets
- **Code Execution**: Runs in isolated namespace on server
- **Response Time**: Typical < 100ms for solution checking

## Security Considerations

- **Code Execution**: User code runs in restricted namespace
- **No Authentication**: Current version has no user accounts
- **Input Validation**: All inputs are validated before execution
- **No File Access**: Code cannot access filesystem

## Future Enhancements

- User authentication and accounts
- Code execution history
- Social sharing features
- Advanced code editor (Monaco Editor)
- Real-time collaboration
- Leaderboards and achievements

## License

MIT License - see LICENSE file for details
