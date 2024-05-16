# Praktis LLM Agent
Implementasi LLM menggunakan Gemini Function Calling.

## Cara Menggunakan
- Buat env baru, python=3.12
```
python -m venv myenv
```
- Buat `.env` Environment Variables file
```
GOOGLE_APPLICATION_CREDENTIALS = path\to\credentials.json
PROJECT_ID = ""
LOCATION = ""
```
- Install dependencies
```
pip install -r requirements.txt
```
- Run streamlit web app
```
streamlit run app.py
```
Buka `http://localhost:8501/` di browser
