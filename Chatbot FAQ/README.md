# Chatbot FAQ (Flask)

Proyecto de ejemplo: un chatbot tipo FAQ con interfaz web creada en Flask. Utiliza TF-IDF (scikit-learn) para buscar la pregunta más similar en `faqs.json`.

Requisitos
- Python 3.8+
- Windows PowerShell (instrucciones abajo)

Instalación (PowerShell)
```powershell
python -m venv .venv
; .\.venv\Scripts\Activate.ps1
; pip install -r requirements.txt
```

Nota sobre modelos
- Este ejemplo usa `sentence-transformers` con el modelo por defecto `all-MiniLM-L6-v2`. La primera ejecución descargará el modelo (~20-100MB según versión). Si deseas usar otro modelo, puedes establecer la variable de entorno `ST_MODEL` antes de ejecutar la app, por ejemplo:

```powershell
$env:ST_MODEL = 'all-mpnet-base-v2'
; flask run
```

Ejecutar (PowerShell)
```powershell
$env:FLASK_APP = 'app.py'
; flask run
```

Uso
- Abre `http://127.0.0.1:5000` en tu navegador.
- Escribe una pregunta en la caja y presiona "Preguntar". El backend devuelve la respuesta más parecida según las FAQs.

Personalización
- Edita `faqs.json` para añadir tus propias preguntas y respuestas.
- Ajusta el umbral de similitud en `find_best_answer` en `app.py`.

Próximos pasos sugeridos
- Mejorar el ranking con embeddings (OpenAI, sentence-transformers).
- Añadir almacenamiento y edición de FAQs desde la interfaz.
