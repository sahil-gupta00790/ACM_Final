import pre_processing
import requests
import base64




API_KEY = 'AIzaSyDkcWpKkUSy3LG4jwBDZYpCYpbhD9WQC0Q'
API_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent'

def get_text_from_image(image_path: str) -> str:
    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': API_KEY
    }

    # Read and encode the image
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Prepare the request payload
    data = {
        "contents": [{
            "parts": [
                {"text": "Identify and extract all text visible in this image. Return only the extracted text, without any additional commentary.Don't give repetations and don't give something which might not be relevant to shop name"},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": encoded_image
                    }
                }
            ]
        }]
    }

    # Make the API request
    response = requests.post(API_ENDPOINT, headers=headers, json=data)

    if response.status_code == 200:
        # Extract the text from the response
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
if __name__ == "__main__":
    ab=pre_processing.preprocess_image(image_path='image.jpg')
    image_path = "results.jpg"
    extracted_text = get_text_from_image(image_path)
    print("Extracted text:")
    print(extracted_text)