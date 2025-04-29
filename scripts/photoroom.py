import http.client
import mimetypes
import os
import uuid
import argparse

# Please replace with your own apiKey
API_KEY = os.getenv("PHOTOROOM_API_KEY")  # Fetch from environment variable

def remove_background(input_image_path, output_image_path):
    # Define multipart boundary
    boundary = '----------{}'.format(uuid.uuid4().hex)

    # Get mimetype of image
    content_type, _ = mimetypes.guess_type(input_image_path)
    if content_type is None:
        content_type = 'application/octet-stream'  # Default type if guessing fails

    # Prepare the POST data
    with open(input_image_path, 'rb') as f:
        image_data = f.read()
    filename = os.path.basename(input_image_path)

    body = (
    f"--{boundary}\r\n"
    f"Content-Disposition: form-data; name=\"image_file\"; filename=\"{filename}\"\r\n"
    f"Content-Type: {content_type}\r\n\r\n"
    ).encode('utf-8') + image_data + f"\r\n--{boundary}--\r\n".encode('utf-8')
    
    # Set up the HTTP connection and headers
    conn = http.client.HTTPSConnection('sdk.photoroom.com')

    headers = {
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'x-api-key': API_KEY
    }

    # Make the POST request
    conn.request('POST', '/v1/segment', body=body, headers=headers)
    response = conn.getresponse()

    # Handle the response
    if response.status == 200:
        response_data = response.read()
        with open(output_image_path, 'wb') as out_f:
            out_f.write(response_data)
        print("Image saved to", output_image_path)
    else:
        print(f"Error: {response.status} - {response.reason}")
        print(response.read())

    # Close the connection
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove background from an image using PhotoRoom API')
    parser.add_argument('--input', '-i', required=True, help='Path to input image folder')
    parser.add_argument('--output', '-o', required=True, help='Path to output image folder')
    
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file)
        output_path=os.path.join(output_folder, file)
        remove_background(input_path, output_path)