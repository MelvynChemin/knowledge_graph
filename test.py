import ollama

# Text-only example
resp = ollama.chat(
    model='llava:7b-v1.5-q4_1',
    messages=[{'role': 'user', 'content': 'Give me three use cases for LLaVA.'}],
)
print(resp['message']['content'])

# Vision example (image path)
with open('./images/presidentielles.jpg', 'rb') as f:
    img_bytes = f.read()

resp = ollama.chat(
    model='llava:7b-v1.5-q4_1',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': [img_bytes],
    }],
)
print(resp['message']['content'])
