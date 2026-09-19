const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return window.location.origin;
    }
  }
  return 'http://127.0.0.1:8000';
};

const API_BASE_URL = getApiBaseUrl();

export async function checkHealth() {
  const res = await fetch(`${API_BASE_URL}/api/health`);
  if (!res.ok) throw new Error('Health check failed');
  return await res.json();
}

export async function chatQuery(question, responseLanguage = null) {
  const payload = { question: question.trim() };
  if (responseLanguage && typeof responseLanguage === 'string' && responseLanguage.trim()) {
    payload.response_language = responseLanguage.trim();
  }

  const res = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || 'Failed to retrieve response from PURIVU backend.');
  }

  return await res.json();
}


export async function analyzeProduct(imageFile, question) {
  const formData = new FormData();
  formData.append('image', imageFile);
  if (question && question.trim()) {
    formData.append('question', question.trim());
  }

  const res = await fetch(`${API_BASE_URL}/api/vision`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || 'Failed to analyze product image.');
  }

  return await res.json();
}
