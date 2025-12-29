// PyTorch Tensor Marathon - Main Application Logic

// State
const state = {
    currentLang: 'en',
    currentCategory: null,
    currentProblems: [],
    currentProblem: null,
    currentProblemIndex: 0,
    solvedProblems: new Set(),
    filteredDifficulty: 'all',
};

// API Base URL
const API_BASE = '';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    loadProgress();
    setupEventListeners();
    await loadCategories();
    updateI18n(state.currentLang);
}

// Event Listeners
function setupEventListeners() {
    // Home logo click
    document.getElementById('home-logo').addEventListener('click', () => {
        showWelcome();
    });

    // Reset progress button
    // Reset progress button
    document.getElementById('reset-progress').addEventListener('click', (e) => {
        e.preventDefault();
        showConfirm(
            state.currentLang === 'ja' ? '進捗をリセットしますか？' : 'Reset all progress?',
            () => {
                state.solvedProblems.clear();
                saveProgress();
                location.reload();
            }
        );
    });

    // Language selector
    document.getElementById('lang-selector').addEventListener('change', (e) => {
        state.currentLang = e.target.value;
        updateI18n(state.currentLang);
        if (state.currentProblem) {
            displayProblemDetail(state.currentProblem);
        }
    });

    // Back button
    document.getElementById('back-to-list').addEventListener('click', () => {
        showProblemList();
    });

    // Difficulty filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.filteredDifficulty = btn.dataset.filter;
            displayProblems(state.currentProblems);
        });
    });

    // Hint button
    document.getElementById('hint-btn').addEventListener('click', () => {
        const hintBox = document.getElementById('hint-box');
        // Always reset to original hint text when toggling manually
        if (state.currentProblem) {
            const hintText = state.currentProblem['hint_' + state.currentLang];
            // Parse markdown if marked is available
            const hintContent = typeof marked !== 'undefined' ? marked.parse(hintText) : hintText;
            hintBox.innerHTML = `<strong>${state.currentLang === 'ja' ? 'ヒント' : 'Hint'}:</strong><br><div class="markdown-content">${hintContent}</div>`;
        }
        hintBox.classList.toggle('hidden');
    });

    // Run button
    document.getElementById('run-btn').addEventListener('click', () => {
        runCode();
    });

    // Solution button
    document.getElementById('solution-btn').addEventListener('click', async () => {
        await showSolution();
    });

    // Navigation buttons
    document.getElementById('prev-problem').addEventListener('click', () => {
        navigateProblem(-1);
    });

    document.getElementById('next-problem').addEventListener('click', () => {
        navigateProblem(1);
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            runCode();
        }
    });
}

// API Calls
async function loadCategories() {
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/api/categories`);
        const data = await response.json();

        displayCategories(data.categories);
        updateStats(data.total_problems);
    } catch (error) {
        console.error('Error loading categories:', error);
    } finally {
        showLoading(false);
    }
}

async function loadProblems(category) {
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/api/problems/${category}`);
        const data = await response.json();

        state.currentCategory = category;
        state.currentProblems = data.problems;
        displayProblems(data.problems);
        showProblemList();
    } catch (error) {
        console.error('Error loading problems:', error);
    } finally {
        showLoading(false);
    }
}

async function loadProblemDetail(problemId) {
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/api/problem/${problemId}`);
        const problem = await response.json();

        state.currentProblem = problem;
        state.currentProblemIndex = state.currentProblems.findIndex(p => p.id === problemId);
        displayProblemDetail(problem);
        showProblemDetail();
    } catch (error) {
        console.error('Error loading problem:', error);
    } finally {
        showLoading(false);
    }
}

async function runCode() {
    const userCode = document.getElementById('user-code').value.trim();

    if (!userCode) {
        showResult(false, state.currentLang === 'ja' ? 'コードを入力してください' : 'Please enter your code');
        return;
    }

    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/api/check`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                problem_id: state.currentProblem.id,
                user_code: userCode,
            }),
        });

        const result = await response.json();
        showResult(result.is_correct, result.message, result);

        if (result.is_correct) {
            markAsSolved(state.currentProblem.id);
        }
    } catch (error) {
        console.error('Error checking solution:', error);
        showResult(false, state.currentLang === 'ja' ? '実行エラーが発生しました' : 'Execution error occurred');
    } finally {
        showLoading(false);
    }
}

async function showSolution() {
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/api/solution/${state.currentProblem.id}`);
        const data = await response.json();

        const solutionBox = document.getElementById('solution-box');
        const solutionCode = document.getElementById('solution-code');

        solutionCode.textContent = data.solution_code;
        Prism.highlightElement(solutionCode);
        solutionBox.classList.remove('hidden');
    } catch (error) {
        console.error('Error loading solution:', error);
    } finally {
        showLoading(false);
    }
}

// Display Functions
function displayCategories(categories) {
    const listEl = document.getElementById('category-list');
    listEl.innerHTML = '';

    categories.forEach(cat => {
        const el = document.createElement('div');
        el.className = 'category-item';
        el.innerHTML = `
            <span class="category-name">${t('cat_' + cat.id, state.currentLang)}</span>
            <span class="category-count">${cat.stats.total}</span>
        `;
        el.addEventListener('click', () => {
            document.querySelectorAll('.category-item').forEach(item => item.classList.remove('active'));
            el.classList.add('active');
            loadProblems(cat.id);
        });
        listEl.appendChild(el);
    });
}

function displayProblems(problems) {
    const gridEl = document.getElementById('problem-grid');
    const categoryTitle = document.getElementById('category-title');

    categoryTitle.textContent = t('cat_' + state.currentCategory, state.currentLang);

    const filtered = state.filteredDifficulty === 'all'
        ? problems
        : problems.filter(p => p.difficulty === state.filteredDifficulty);

    gridEl.innerHTML = '';

    filtered.forEach(problem => {
        const el = document.createElement('div');
        el.className = 'problem-card' + (state.solvedProblems.has(problem.id) ? ' solved' : '');
        el.innerHTML = `
            <div class="problem-card-title">${problem['title_' + state.currentLang]}</div>
            <div class="problem-card-meta">
                <span class="difficulty-badge ${problem.difficulty}">${t(problem.difficulty, state.currentLang)}</span>
            </div>
        `;
        el.addEventListener('click', () => {
            loadProblemDetail(problem.id);
        });
        gridEl.appendChild(el);
    });
}

function displayProblemDetail(problem) {
    document.getElementById('problem-id').textContent = problem.id;
    document.getElementById('problem-difficulty').textContent = t(problem.difficulty, state.currentLang);
    document.getElementById('problem-difficulty').className = `difficulty-badge ${problem.difficulty}`;
    document.getElementById('problem-title').textContent = problem['title_' + state.currentLang];
    document.getElementById('problem-description').textContent = problem['description_' + state.currentLang];

    // Setup code
    const setupCode = document.getElementById('setup-code');
    setupCode.textContent = problem.setup_code || '# No setup required';
    Prism.highlightElement(setupCode);

    // Hint
    const hintBox = document.getElementById('hint-box');
    const hintText = problem['hint_' + state.currentLang];
    if (hintText) {
        hintBox.textContent = hintText;
        document.getElementById('hint-btn').style.display = 'inline-flex';
    } else {
        document.getElementById('hint-btn').style.display = 'none';
    }
    hintBox.classList.add('hidden');

    // Clear previous state
    document.getElementById('user-code').value = '';
    document.getElementById('result-box').classList.add('hidden');
    document.getElementById('solution-box').classList.add('hidden');

    // Update navigation buttons
    updateNavigationButtons();
}

function showResult(isCorrect, message, details = {}) {
    const resultBox = document.getElementById('result-box');
    const resultIcon = resultBox.querySelector('.result-icon');
    const resultTitle = resultBox.querySelector('.result-title');
    const resultMessage = resultBox.querySelector('.result-message');

    resultBox.classList.remove('hidden', 'success', 'error');
    resultBox.classList.add(isCorrect ? 'success' : 'error');

    resultIcon.textContent = isCorrect ? '✅' : '❌';
    resultTitle.textContent = isCorrect ? (state.currentLang === 'ja' ? '正解！' : 'Correct!') : (state.currentLang === 'ja' ? '不正解' : 'Incorrect');

    let messageHtml = '';

    // Format error messages in a more user-friendly way
    if (!isCorrect && details.error_type) {
        if (details.error_type === 'shape') {
            messageHtml = `<div class="error-section">
                <div class="error-type">${state.currentLang === 'ja' ? '形状エラー' : 'Shape Error'}</div>
                <div class="error-detail">
                    <strong>${state.currentLang === 'ja' ? '期待される形状' : 'Expected Shape'}:</strong>
                    <code>${JSON.stringify(details.expected_shape)}</code>
                </div>
                <div class="error-detail">
                    <strong>${state.currentLang === 'ja' ? '実際の形状' : 'Actual Shape'}:</strong>
                    <code>${JSON.stringify(details.actual_shape)}</code>
                </div>
            </div>`;
        } else {
            messageHtml = message;
        }
    } else {
        // Success case - add shape info
        messageHtml = `<div>${message}</div>`;

        if (details.actual_shape) {
            messageHtml += `<div class="success-detail" style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
                <strong>${state.currentLang === 'ja' ? '結果の形状' : 'Result Shape'}:</strong>
                <code>${JSON.stringify(details.actual_shape)}</code>
            </div>`;
        }
    }

    resultMessage.innerHTML = messageHtml;

    // Show execution output if available
    const outputSection = document.getElementById('execution-output');
    const outputText = document.getElementById('output-text');

    if (outputSection && outputText) {
        if (details.execution_output && details.execution_output.trim() !== '') {
            outputSection.classList.remove('hidden');
            outputText.textContent = details.execution_output;
        } else {
            outputSection.classList.add('hidden');
        }
    }

    // Scroll to result
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}


function updateNavigationButtons() {
    const prevBtn = document.getElementById('prev-problem');
    const nextBtn = document.getElementById('next-problem');

    prevBtn.disabled = state.currentProblemIndex === 0;
    nextBtn.disabled = state.currentProblemIndex === state.currentProblems.length - 1;
}

function navigateProblem(direction) {
    const newIndex = state.currentProblemIndex + direction;
    if (newIndex >= 0 && newIndex < state.currentProblems.length) {
        const problem = state.currentProblems[newIndex];
        loadProblemDetail(problem.id);
    }
}

// View Management
function showWelcome() {
    document.getElementById('welcome-screen').classList.remove('hidden');
    document.getElementById('problem-list-view').classList.add('hidden');
    document.getElementById('problem-detail-view').classList.add('hidden');
}

function showProblemList() {
    document.getElementById('welcome-screen').classList.add('hidden');
    document.getElementById('problem-list-view').classList.remove('hidden');
    document.getElementById('problem-detail-view').classList.add('hidden');
}

function showProblemDetail() {
    document.getElementById('welcome-screen').classList.add('hidden');
    document.getElementById('problem-list-view').classList.add('hidden');
    document.getElementById('problem-detail-view').classList.remove('hidden');
}

function showLoading(show) {
    document.getElementById('loading-overlay').classList.toggle('hidden', !show);
}

// Progress Management
function loadProgress() {
    const saved = localStorage.getItem('tensorMarathonProgress');
    if (saved) {
        state.solvedProblems = new Set(JSON.parse(saved));
        updateProgressDisplay();
    }
}

function saveProgress() {
    localStorage.setItem('tensorMarathonProgress', JSON.stringify([...state.solvedProblems]));
    updateProgressDisplay();
}

function markAsSolved(problemId) {
    state.solvedProblems.add(problemId);
    saveProgress();

    // Update the current problem card if in list view
    const problemCards = document.querySelectorAll('.problem-card');
    problemCards.forEach(card => {
        const cardProblem = state.currentProblems.find(p =>
            card.textContent.includes(p['title_' + state.currentLang])
        );
        if (cardProblem && cardProblem.id === problemId) {
            card.classList.add('solved');
        }
    });
}

function updateStats(totalProblems) {
    document.getElementById('total-problems').textContent = totalProblems;
    document.getElementById('total-count').textContent = totalProblems;
    updateProgressDisplay();
}

function updateProgressDisplay() {
    const totalElement = document.getElementById('total-count');
    const total = parseInt(totalElement.textContent) || 100;
    const solved = state.solvedProblems.size;

    document.getElementById('solved-count').textContent = solved;

    const percentage = total > 0 ? Math.round((solved / total) * 100) : 0;
    document.getElementById('user-progress').textContent = `${percentage}%`;
}

// Gemini AI Features
let geminiEnabled = false;

async function checkGeminiAvailability() {
    try {
        const response = await fetch(`${API_BASE}/api/gemini/enabled`);
        const data = await response.json();
        geminiEnabled = data.enabled;
        updateGeminiUI();
    } catch (error) {
        console.error('Error checking Gemini availability:', error);
    }
}

function updateGeminiUI() {
    const geminiButtons = document.querySelectorAll('.gemini-feature');
    geminiButtons.forEach(btn => {
        if (geminiEnabled) {
            btn.style.display = 'inline-flex';
        } else {
            btn.style.display = 'none';
        }
    });
}





async function getAIHint() {
    if (!state.currentProblem || !geminiEnabled) return;

    showLoading(true);
    try {
        const userCode = document.getElementById('user-code').value.trim();
        const response = await fetch(`${API_BASE}/api/gemini/hint`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                problem_id: state.currentProblem.id,
                language: state.currentLang,
                user_code: userCode || null
            })
        });

        if (!response.ok) throw new Error('Failed to generate hint');

        const data = await response.json();
        const hintBox = document.getElementById('hint-box');
        // Parse markdown if marked is available
        const hintContent = typeof marked !== 'undefined' ? marked.parse(data.hint) : data.hint;
        hintBox.innerHTML = `<strong>🤖 AI Hint:</strong><br><div class="markdown-content">${hintContent}</div>`;
        hintBox.classList.remove('hidden');
    } catch (error) {
        console.error('Error generating hint:', error);
        showResult(false, state.currentLang === 'ja' ? 'ヒントの生成に失敗しました' : 'Failed to generate hint');
    } finally {
        showLoading(false);
    }
}

// Initialize Gemini features on page load
document.addEventListener('DOMContentLoaded', () => {
    checkGeminiAvailability();
    initTheme();
});


// Internationalization - import function 't' from i18n.js
function updateI18n(lang) {
    // Update all elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        el.innerHTML = t(key, lang);
    });

    // Update stats badge
    updateProgressDisplay();
}

// Confirmation Modal
function showConfirm(message, onConfirm) {
    const modal = document.getElementById('confirmation-modal');
    document.getElementById('confirm-message').textContent = message;

    // Update labels based on language
    const cancelBtn = document.getElementById('cancel-confirm-btn');
    const confirmBtn = document.getElementById('confirm-action-btn');

    cancelBtn.textContent = state.currentLang === 'ja' ? 'キャンセル' : 'Cancel';
    confirmBtn.textContent = state.currentLang === 'ja' ? '実行' : 'Confirm';

    // Setup confirm action
    confirmBtn.onclick = () => {
        onConfirm();
        closeConfirm();
    };

    modal.classList.remove('hidden');
}

window.closeConfirm = function () {
    document.getElementById('confirmation-modal').classList.add('hidden');
};

// Theme Handling
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    document.getElementById('theme-toggle').addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';

        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        updateThemeIcon(next);
    });
}

function updateThemeIcon(theme) {
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        btn.textContent = theme === 'dark' ? '🌙' : '☀️';
    }
}
