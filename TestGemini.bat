@echo off
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent" ^
  -H "Content-Type: application/json" ^
  -H "X-goog-api-key: your api key" ^
  -X POST ^
  -d "{\"contents\":[{\"parts\":[{\"text\":\"Explain how AI works in a few words\"}]}]}"
