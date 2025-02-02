import os
import json
import requests
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def generate_itinerary(query):
    """
    Uses ChatCompletion API to generate a travel itinerary (with bullet-pointed locations)
    """
    messages = [
        {"role": "system", "content": "You are a helpful travel assistant."},
        {"role": "user", "content": (
            f"Create a travel itinerary based on the following preferences: {query}\n"
            "List the key locations (cities, landmarks, attractions) in bullet points."
        )}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using the base model here
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        itinerary = response.choices[0].message.content.strip()
        return itinerary
    except Exception as e:
        print("Error generating itinerary from OpenAI:", e)
        return None

def parse_locations(itinerary):
    """
    Parses bullet-pointed itinerary text to extract location names.
    """
    lines = itinerary.splitlines()
    locations = []
    for line in lines:
        line = line.strip()
        # Consider lines that start with common bullet characters or numbers
        if line.startswith("-") or line.startswith("*") or (line and line[0].isdigit()):
            location = line.lstrip("-*0123456789. ").strip()
            if location:
                locations.append(location)
    return locations

def get_place_details(place):
    """
    Uses the Google Places Text Search API to retrieve details for a given location.
    """
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": place,
        "key": GOOGLE_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("results"):
            return data["results"][0]
        else:
            return None
    except Exception as e:
        print(f"Error fetching details for {place}:", e)
        return None

def call_create_route_function(places_details):
    """
    Uses AI function calling to get an optimized route (ordered list) along with an explanation.
    We supply a function schema for 'create_route' so that ChatGPT returns a structured response.
    """
    # Build a details string from the places
    details_str = ""
    for detail in places_details:
        if detail:
            details_str += (
                f"Name: {detail.get('name', 'N/A')}\n"
                f"Address: {detail.get('formatted_address', 'N/A')}\n"
                f"Rating: {detail.get('rating', 'N/A')}\n\n"
            )
    
    # The prompt instructs the model to create a structured route.
    messages = [
        {"role": "system", "content": "You are a travel planning assistant that specializes in optimizing travel routes."},
        {"role": "user", "content": (
            "Based on the following list of places and their details, determine the best order to visit them "
            "considering geographic proximity and ratings. "
            "Return the result by calling the function 'create_route' with two fields: 'route' (an ordered list of place names) "
            "and 'explanation' (a short rationale). \n\n"
            f"{details_str}"
        )}
    ]
    
    # Define the function schema for function calling.
    functions = [
        {
            "name": "create_route",
            "description": "Generates an optimized travel route given a list of places",
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of place names in the recommended visiting order."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "A brief explanation of the recommended route."
                    }
                },
                "required": ["route", "explanation"]
            }
        }
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using the base model here
            messages=messages,
            functions=functions,
            function_call="auto",  # Let the model decide whether to call the function.
            max_tokens=200,
            temperature=0.7,
        )
        
        # Check if the model has returned a function call.
        message = response.choices[0].message
        if message.get("function_call"):
            # The arguments come as a JSON string; parse them.
            arguments = json.loads(message["function_call"]["arguments"])
            return arguments
        else:
            # Fallback: return text if a function call was not used.
            print("Function call was not used. Response text:")
            print(message.get("content", ""))
            return None
    except Exception as e:
        print("Error determining best route via function calling:", e)
        return None

def generate_google_maps_url(route):
    """
    Builds a Google Maps directions URL from an ordered list of stops.
    The URL format uses '/dir/' followed by the stops separated by '/'.
    """
    base_url = "https://www.google.com/maps/dir/"
    # Replace spaces with '+' to URL-encode the place names.
    stops = [place.replace(" ", "+") for place in route]
    # Join stops with '/' to form the URL path.
    url = base_url + "/".join(stops)
    return url

def main():
    # Get travel preferences from the user.
    query = input("Enter your travel preferences (e.g., 'beach holiday in Europe with art and culture'): ")
    
    print("\nGenerating itinerary...")
    itinerary = generate_itinerary(query)
    if not itinerary:
        print("Could not generate itinerary.")
        return
    
    print("\n--- Generated Itinerary ---")
    print(itinerary)
    
    # Parse the itinerary to extract locations.
    locations = parse_locations(itinerary)
    if not locations:
        print("No locations could be parsed from the itinerary.")
        return
    
    print("\nFetching details for each location using Google Places:")
    places_details = []
    for loc in locations:
        details = get_place_details(loc)
        if details:
            name = details.get("name", "N/A")
            address = details.get("formatted_address", "N/A")
            rating = details.get("rating", "N/A")
            print(f"\n{name}\nAddress: {address}\nRating: {rating}")
            places_details.append(details)
        else:
            print(f"\nNo details found for {loc}.")
    
    if len(places_details) < 2:
        print("Not enough locations to determine a route.")
        return

    # Call the function to get an optimized route using AI function calling.
    print("\nDetermining the best route using AI function calling...")
    route_info = call_create_route_function(places_details)
    if route_info:
        route = route_info.get("route")
        explanation = route_info.get("explanation")
        print("\n--- Recommended Route ---")
        print("Route Order:", route)
        print("Explanation:", explanation)
        
        # Generate a Google Maps URL from the route list.
        maps_url = generate_google_maps_url(route)
        print("\nOpen this URL in your browser to see the route on Google Maps:")
        print(maps_url)
    else:
        print("Could not determine the best route.")

if __name__ == "__main__":
    main()
