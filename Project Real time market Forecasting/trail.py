from google import genai
import base64
#from google.generativeai import Gemini
api="AIzaSyClfw6ZTb6a_p-l4ziRcCt32YJmiQFwvIQ"
ai_prompt = (
                            "You are a stock market technical analyst. Analyze the chart provided in the image, "
                            "taking into account the price action and the technical indicators overlaid on the chart. "
                            "Provide your trading recommendation (buy, hold, or sell) along with detailed reasoning."
                        )
with open(r"C:\Users\AJITESH\Downloads\Screenshot 2025-02-08 055137.png", "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            image_data_uri = f"data:image/png;base64,{image_data}"
try:
    client = genai.Client(api_key=api)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[ai_prompt,image_data_uri]
    )
    print("Raw AI Response (text-only):", response.text)
except Exception as e:
    print(f"Error during text-only AI analysis: {e}")
