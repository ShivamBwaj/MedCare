import httpx
import os
import json
import asyncio
from dotenv import load_dotenv

async def test_groq_api():
    # Load environment
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    
    print('=== GROQ API DEBUG ===')
    print(f'API Key loaded: {"YES" if api_key else "NO"}')
    
    if not api_key or api_key == 'your_groq_api_key_here':
        print('❌ API key not configured properly')
        return
    
    print(f'Key length: {len(api_key)} characters')
    print(f'Key starts with: {api_key[:15]}...')
    
    # Test simple API call
    test_prompt = "Analyze these symptoms: fever, headache, cough. Respond with JSON format."
    
    try:
        async with httpx.AsyncClient() as client:
            url = "https://api.groq.com/openai/v1/chat/completions"
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant. Always respond with valid JSON only."
                    },
                    {
                        "role": "user", 
                        "content": test_prompt
                    }
                ],
                "model": "llama3-8b-8192",
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            print(f'\n🔗 Testing URL: {url}')
            print(f'📤 Model: {payload["model"]}')
            print(f'🔑 Auth header: Bearer {api_key[:10]}...')
            
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            print(f'\n📥 Response Status: {response.status_code}')
            
            if response.status_code == 200:
                result = response.json()
                ai_text = result["choices"][0]["message"]["content"]
                print(f'✅ SUCCESS! Response length: {len(ai_text)} chars')
                print(f'📝 AI Response:\n{ai_text}')
                
                # Try to parse JSON
                try:
                    if ai_text.startswith('```json'):
                        ai_text = ai_text.replace('```json', '').replace('```', '').strip()
                    
                    parsed_json = json.loads(ai_text)
                    print(f'\n🎯 Parsed JSON successfully:')
                    print(json.dumps(parsed_json, indent=2))
                except json.JSONDecodeError as e:
                    print(f'❌ JSON parsing failed: {e}')
                
            else:
                print(f'❌ API Error: {response.status_code}')
                print(f'Error details: {response.text}')
                
                # Check for common errors
                if response.status_code == 401:
                    print('🔑 Authentication failed - check your API key')
                elif response.status_code == 429:
                    print('⏰ Rate limit exceeded - wait and try again')
                elif response.status_code == 400:
                    print('📝 Bad request - check payload format')
                
    except httpx.TimeoutException:
        print('❌ Request timed out')
    except httpx.ConnectError:
        print('❌ Connection failed - check internet connection')
    except Exception as e:
        print(f'❌ Unexpected error: {e}')

if __name__ == "__main__":
    asyncio.run(test_groq_api())
