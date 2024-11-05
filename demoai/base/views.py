from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import joblib
import json
from django.views.decorators.csrf import csrf_exempt


# Load environment variables
load_dotenv()
API_KEY = os.getenv('API_KEY_OPENAI')

model_path = os.path.join(os.path.dirname(__file__), 'random_forest_classifier.pkl')
model_col = os.path.join(os.path.dirname(__file__), 'model_columns.pkl')

model = joblib.load(model_path)
model_columns = joblib.load(model_col)

# Set OpenAI API key
client = OpenAI(api_key = API_KEY)

@csrf_exempt


def chatbot_view(request):
    if request.method == 'POST':
        # Parse JSON data
        try:
            data = json.loads(request.body)
            user_message = data.get('Prompt')  # Use get() for safety
            business_model = data.get('BusinessModel')  # Retrieve the business model
        except json.JSONDecodeError:
            return JsonResponse({'message': '', 'response': "Invalid JSON format"}, status=400)

        # Customize system message based on the selected business model
        system_message = f"You are a helpful assistant specialized in the {business_model} business."

        # Generate response using ChatGPT (chat-based API)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Specify the model
            messages=[
                {"role": "system", "content": system_message},  # Use the customized system message
                {"role": "user", "content": user_message}
            ]
        )
        response = completion.choices[0].message.content.strip()

        # Return the response as JSON
        return JsonResponse({'message': user_message, 'response': response})

    # In case of GET request, just return an empty JSON
    return JsonResponse({'message': '', 'response': ''})



#127.0.0.1:8000/predict
@csrf_exempt
def predict(req):
    if req.method == 'POST':
        try:
            # Parse JSON data from the request body
            data = json.loads(req.body.decode('utf-8'))
            '''
            RV_Type: '',
    Manufacturer: '',
    Model_Name: '',
    Trim: '',
    Leveling_System:'',
    Jack:'',
    Year: '',
            '''
            input_data = {
                'RV Type': [data.get('RV_Type')],
                'Manufacturer': [data.get('Manufacturer')],
                'Model Name': [data.get('Model_Name')],
                'Trim': [data.get('Trim')],
                'Leveling System': [data.get('Leveling_System')],
                'jack': [data.get('Jack')],
                'Year': [data.get('Year')]
            }
            
            test_df = pd.DataFrame(input_data)
            print(test_df)
            
            # Handle categorical encoding
            categorical_columns = ['RV Type', 'Manufacturer', 'Model Name', 'Trim', 'Leveling System']
            test_encoded = pd.get_dummies(test_df, columns=categorical_columns)
            test_encoded = test_encoded.reindex(columns=model_columns, fill_value=0)

            # Make the prediction
            prediction = model.predict(test_encoded)
            prediction_result = prediction[0]
            print(prediction_result)

            return JsonResponse({'prediction': prediction_result})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
 