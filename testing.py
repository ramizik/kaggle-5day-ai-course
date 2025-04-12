import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FunctionDeclaration, Tool
import vertexai.preview.generative_models as generative_models # For safety settings etc.
import faiss
import numpy as np
import base64 # For image/audio data
import json
import requests # For external APIs (weather, etc.)
from datetime import datetime
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

#WEATHER_API_KEY = "4"
#GOOGLE_API_KEY = "A"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
#client = genai.Client(api_key=GOOGLE_API_KEY)
#PROJECT_ID = "" # Replace with your Project ID
LOCATION = "us-central1" # Or your preferred region
MODEL_NAME = "gemini-2.0-flash-001"
IMAGE_MODEL_NAME = "gemini-2.0-flash-001"
EMBEDDING_MODEL_NAME = "textembedding-gecko@003" # Or another embedding model

class RequestState(TypedDict):
    "State representing the traveler's request conversation."

    # The chat conversation.
    messages: Annotated[list, add_messages]
    # The traveler's in-progress request.
    request: list[str]
    # Flag indicating that request has been processed and completed.
    finished: bool
    user_profile: dict
    location: str
    recommendation: dict
    
# The system instruction defines how the chatbot is expected to behave and includes
# rules for when to call different functions, as well as rules for the conversation, such
# as tone and what is permitted for discussion.
TRAVELAGENT_SYSINT = {
    "role": "system",
    "content": "You are a TravelAIAgent, an interactive travel companion. A human will ask you for personalized travel advice and suggestions during their trip. "
    "You will assist them with planning, recommendations, cultural guidance, and other travel-related tasks — but only within the scope of travel (no off-topic discussion, "
    "though you can freely chat about travel experiences, destinations, and helpful advice)."
    "\n\n"
    "Learn the traveler's preferences such as budget, interests (e.g., history, food, nightlife, adventure, relaxation), accessibility needs, and pace of travel. "
    "You can infer or confirm these preferences from the user's messages and update the user profile accordingly. "
    "Use these preferences along with the current time, location, and weather to tailor your suggestions."
    "\n\n"
    "The user may ask for specific functions like:\n"
    "- Getting current weather: call get_weather\n"
    "- Finding events nearby: call find_events\n"
    "- Discovering places: call find_places\n"
    "- Translating or describing text or images: use translate_text or describe_image\n"
    "- Retrieving local tips or hidden gems: call retrieve_local_info (this uses your curated knowledge base)\n"
    "- Creating a short itinerary: build flexible suggestions with time estimates (no tool call needed)\n"
    "\n"
    "You may update preferences and context gradually through the conversation. Do not repeat known information unless asked. "
    "When suggesting places or activities, explain *why* they are a good fit for the traveler, based on what you know about them."
    "\n\n"
    "Always clarify if a user request is ambiguous or if you're not sure of a location or preference. "
    "If a tool or capability is unavailable, break the fourth wall and inform the user that the feature hasn't been implemented yet. "
    "End the conversation warmly when the user says goodbye or signals they're done."
    "Always return structured responses in JSON format. For example, when recommending places, use:\n"
    "{\"type\": \"recommendation\", \"location\": \"Barcelona\", \"suggestions\": [ ... ]}"
}

# This is the message with which the system opens the conversation.
WELCOME_MSG = "Welcome to the TravelAIAgent. Type `q` to quit. How can I help you today?"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Chatbot Node
def chatbot(state: RequestState) -> RequestState:
    """LangGraph chatbot node — handles user input and infers request."""
    # Create message history with properly formatted messages
    message_history = [TRAVELAGENT_SYSINT] + state["messages"]
    
    try:
        response = llm.invoke(message_history)
        # Extract the text content
        response_text = response.content
        
        try:
            # Try to parse as JSON first
            parsed = json.loads(response_text)
            
            # If we got a weather request, update location
            if parsed.get("action") == "get_weather" and parsed.get("location"):
                state["location"] = parsed["location"]
                return {
                    **state,
                    "messages": state["messages"] + [{"role": "assistant", "content": response_text}],
                    "request": ["get_weather"],
                    "finished": False
                }
                
            response_dict = {
                "role": "assistant",
                "content": response_text
            }
        except json.JSONDecodeError:
            response_dict = {
                "role": "assistant",
                "content": response_text
            }
            
    except Exception as e:
        response_dict = {
            "role": "assistant",
            "content": f"Error during model call: {e}"
        }
        return {
            **state,
            "messages": state["messages"] + [response_dict],
            "request": [],
            "finished": False
        }

    messages = state["messages"] + [response_dict]
    
    # Extract request type from parsed response if available
    try:
        parsed = json.loads(response_text)
        request_type = parsed.get("action")
    except Exception:
        parsed = {}
        request_type = None

    if not request_type:
        follow_up = {
            "role": "system",
            "content": "What would you like to do next? (Options: get_weather, find_events, find_places)"
        }
        messages.append(follow_up)
        request = []
    else:
        request = [request_type]
    
    return {
        "messages": messages,
        "request": request,
        "user_profile": state.get("user_profile", {}),
        "location": state.get("location", ""),
        "recommendation": state.get("recommendation", {}),
        "finished": False
    }

def profile_collector(state: RequestState) -> RequestState:
    """Asks user for preferences/experiences and stores structured data in user_profile."""
    message_history = [TRAVELAGENT_SYSINT] + state["messages"]

    response = llm.invoke(message_history)
    response_text = response.content  # Extract text

    # Try extracting user profile data from the LLM response
    try:
        parsed = json.loads(response_text)
        user_profile_update = parsed.get("user_profile", {})
        
        # Add validation for expected profile fields
        if user_profile_update:
            valid_fields = ["interests", "budget", "travel_style", "accessibility_needs", "pace"]
            user_profile_update = {k: v for k, v in user_profile_update.items() if k in valid_fields}
    except Exception:
        user_profile_update = {}

    # Merge with existing profile, preserving existing values
    current_profile = state.get("user_profile", {})
    updated_profile = {**current_profile, **user_profile_update}

    return {
        **state,
        "messages": state["messages"] + [response],
        "user_profile": updated_profile,
        "request": ["chatbot"],  # Signal to send back to chatbot
    }

def get_weather(state: RequestState) -> RequestState:
    """Get real-time weather information for a location using OpenWeather API."""
    location = state.get("location", "Unknown")
    
    try:
        # Make API call to OpenWeather
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        
        # Format weather data for LLM
        weather_context = {
            "location": location,
            "current_weather": {
                "description": weather_data["weather"][0]["description"],
                "temperature": round(weather_data["main"]["temp"]),
                "feels_like": round(weather_data["main"]["feels_like"]),
                "humidity": weather_data["main"]["humidity"],
                "wind_speed": round(weather_data["wind"]["speed"] * 3.6),
                "clouds": weather_data["clouds"]["all"]
            }
        }
        
        # Ask LLM to summarize weather information
        weather_prompt = [
            TRAVELAGENT_SYSINT,
            {
                "role": "user",
                "content": f"Here is the current weather data: {json.dumps(weather_context)}. Please provide a natural, helpful summary for a traveler, including any relevant travel recommendations based on the weather."
            }
        ]
        
        weather_summary = llm.invoke(weather_prompt)
        
        weather_text = weather_summary.content  # Extract text
        
        try:
            # Try to parse LLM response as JSON
            summary_content = json.loads(weather_text)
        except json.JSONDecodeError:
            # If not JSON, wrap in our standard format
            summary_content = {
                "type": "weather_response",
                "message": weather_text
            }
        
        msg = {
            "role": "assistant",
            "content": json.dumps(summary_content)
        }

        return {
            **state,
            "messages": state["messages"] + [msg],
            "recommendation": weather_context,
            "request": [],
            "finished": False
        }
        
    except requests.RequestException as e:
        error_msg = {
            "role": "assistant", 
            "content": json.dumps({
                "type": "error",
                "message": f"Sorry, I couldn't get the weather information for {location}. Error: {str(e)}"
            })
        }
        return {
            **state,
            "messages": state["messages"] + [error_msg],
            "request": [],
            "finished": False
        }
    
def find_events(state: RequestState) -> RequestState:
    """Find events based on location and user interests."""
    location = state.get("location", "Unknown")
    user_profile = state.get("user_profile", {})
    interests = user_profile.get("interests", [])
    
    try:
        # Mock events data - in production, this would call an events API
        mock_events = [
            {
                "title": "Sushi Festival",
                "category": "food",
                "date": "2024-03-22",
                "time": "12:00-20:00",
                "location": "Central Park",
                "price": "Free",
                "description": "Experience authentic Japanese cuisine"
            },
            {
                "title": "Historical Walking Tour",
                "category": "history",
                "date": "2024-03-23",
                "time": "10:00-12:00",
                "location": "Old Town",
                "price": "$25",
                "description": "Discover the city's rich history"
            }
        ]

        # Filter events based on user interests
        filtered_events = [e for e in mock_events if e["category"] in interests] if interests else mock_events

        events_response = {
            "type": "events",
            "location": location,
            "matched_interests": interests,
            "events": filtered_events,
            "total_found": len(filtered_events)
        }

        # Create a user-friendly message
        event_descriptions = "\n".join(
            f"- {e['title']} on {e['date']} at {e['time']}\n  {e['description']}\n  Location: {e['location']}, Price: {e['price']}"
            for e in filtered_events
        )

        msg = {
            "role": "assistant",
            "content": json.dumps({
                "type": "events_response",
                "message": f"Here are some events in {location} that match your interests:\n\n{event_descriptions}"
                if filtered_events else
                f"I couldn't find any events in {location} matching your interests at the moment."
            })
        }

        return {
            **state,
            "messages": state["messages"] + [msg],
            "recommendation": events_response,
            "request": [],
            "finished": False
        }
    except Exception as e:
        error_msg = {
            "role": "assistant",
            "content": json.dumps({
                "type": "error",
                "message": f"Sorry, I couldn't find any events: {str(e)}"
            })
        }
        return {**state, "messages": state["messages"] + [error_msg], "finished": False}
    


def log_interaction(data: Dict[Any, Any], interaction_type: str):
    """Log interaction details to file while keeping JSON structure."""
    try:
        # Create a deep copy that's JSON serializable
        if isinstance(data, dict):
            serializable_data = {}
            for k, v in data.items():
                if k == "messages" and isinstance(v, list):
                    # Process each message and make it serializable
                    serializable_messages = []
                    for msg in v:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            # Already in correct format
                            serializable_messages.append(msg)
                        else:
                            # Try to extract role and content
                            try:
                                serializable_messages.append({
                                    "role": getattr(msg, "role", "unknown"),
                                    "content": getattr(msg, "content", str(msg))
                                })
                            except Exception:
                                # Fallback for completely unknown types
                                serializable_messages.append({
                                    "role": "unknown", 
                                    "content": str(msg)
                                })
                    serializable_data[k] = serializable_messages
                else:
                    # Handle other types normally
                    serializable_data[k] = v
            
            json_data = json.dumps(serializable_data, indent=2, default=str)
        else:
            json_data = json.dumps({"data": str(data)}, indent=2)
    except Exception as e:
        # Ultimate fallback
        json_data = json.dumps({"error": f"Failed to serialize: {str(e)}"}, indent=2)
    
    logging.info(f"{interaction_type}: {json_data}")

def format_response_for_user(response_content: str) -> str:
    """Convert JSON response to human-readable format."""
    try:
        # Try to parse as JSON first
        response_dict = json.loads(response_content)
        
        # Extract the relevant message based on response structure
        if "response" in response_dict:
            return response_dict["response"]
        elif "message" in response_dict:
            return response_dict["message"]
        elif "weather" in response_dict:
            return response_dict["weather"]
        else:
            # If no recognized field, return the first value we find
            return next(iter(response_dict.values()))
            
    except json.JSONDecodeError:
        # If not JSON, return as is
        return response_content

def interactive_chat():
    # MAIN
    graph_builder = StateGraph(RequestState)

    # Add all nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("weather", get_weather)
    graph_builder.add_node("events", find_events)
    
    # Set entry point
    graph_builder.set_entry_point("chatbot")
    
    # Add edges
    graph_builder.add_edge("weather", "chatbot")
    graph_builder.add_edge("events", "chatbot")
    
    # Add conditional edges with proper routing logic
    def router(state):
        if state["request"] and state["request"][0] == "get_weather":
            return "weather"
        elif state["request"] and state["request"][0] == "find_events":
            return "events"
        else:
            # Don't route back to chatbot when there's no specific request
            return END
    
    # Fix the conditional edges to prevent infinite recursion
    graph_builder.add_conditional_edges("chatbot", router)
    
    # Compile the graph
    chat_graph = graph_builder.compile()

    # Save Mermaid diagram
    mermaid_code = chat_graph.get_graph().draw_mermaid()
    with open("travel_agent_graph.mmd", "w") as f:
        f.write(mermaid_code)

    # Set up logging
    logging.basicConfig(
        filename=f'travel_agent_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    print("\nWelcome to Travel AI Assistant! (Type 'quit' to exit)")
    print("-" * 50)
    
    # Initialize state correctly
    state = {
        "messages": [],
        "request": [],
        "finished": False,
        "user_profile": {},
        "location": "",
        "recommendation": {}
    }
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye! Have a great trip! 👋")
                break
            
            # Log user input
            log_interaction({"user_input": user_input}, "USER_INPUT")
            
            # Update state with user message
            state["messages"].append({
                "role": "user",
                "content": user_input
            })
            
            # Process through graph
            state = chat_graph.invoke(state, {"recursion_limit": 100})
            
            # Log the raw state
            log_interaction(state, "INTERNAL_STATE")
            
            # Get the last assistant message - FIXED VERSION
            assistant_messages = []
            system_messages = []
            
            # Safely process all messages
            for msg in state["messages"]:
                # Handle different message formats
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    # Standard dict format
                    if msg["role"] == "assistant":
                        assistant_messages.append(msg)
                    elif msg["role"] == "system":
                        system_messages.append(msg)
                else:
                    # Handle LangChain message objects
                    try:
                        role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
                        content = getattr(msg, "content", str(msg))
                        
                        if role == "assistant" or role == "ai":
                            assistant_messages.append({"role": "assistant", "content": content})
                        elif role == "system":
                            system_messages.append({"role": "system", "content": content})
                    except Exception:
                        # Skip messages we can't process
                        continue
            
            # Display the most recent assistant message
            if assistant_messages:
                last_message = assistant_messages[-1]
                human_readable = format_response_for_user(last_message["content"])
                print("\nAssistant:", human_readable)
                
                # Log the response
                log_interaction({
                    "raw_response": last_message["content"],
                    "formatted_response": human_readable
                }, "ASSISTANT_RESPONSE")
            
            # Display the most recent system message (options)
            if system_messages:
                last_system = system_messages[-1]
                print("\nOptions:", format_response_for_user(last_system["content"]))
                
        except Exception as e:
            error_msg = f"Error during chat: {str(e)}"
            print("\nSorry, I encountered an error. Please try again.")
            logging.error(error_msg)
            import traceback
            logging.error(traceback.format_exc())  # Log the full traceback for debugging

if __name__ == "__main__":
    interactive_chat()