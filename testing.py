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


os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
#client = genai.Client(api_key=GOOGLE_API_KEY)
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

def setup_logging():
    """Set up enhanced logging that captures all levels including debug."""
    log_filename = f'travel_agent_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Configure the root logger to capture everything
    logging.basicConfig(
        level=logging.DEBUG,  # Capture debug level and above
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler for everything
            logging.FileHandler(log_filename),
            # Console handler just for info and above
            logging.StreamHandler()
        ]
    )
    
    # Create a logger for our app
    logger = logging.getLogger("travel_agent")
    logger.setLevel(logging.DEBUG)
    
    print(f"Logging to {log_filename}")
    logger.info(f"=== Starting Travel Agent Session ===")
    
    return logger

# Create logger at the top of the file
logger = setup_logging()

# Chatbot Node
def chatbot(state: RequestState) -> RequestState:
    """LangGraph chatbot node — handles user input and infers request."""
    # Store the original user messages to ensure their roles don't change
    user_messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_messages.append(msg.get("content", ""))
    
    # Make sure we use the original message array with proper roles
    message_history = [TRAVELAGENT_SYSINT] + state["messages"]
    
    try:
        response = llm.invoke(message_history)
        response_text = response.content
        logger.debug(f"Raw LLM response: {response_text[:100]}...")
        
        # Flag to track if we found a weather request
        weather_found = False
        location = None
        
        # First, try to detect the weather request pattern in any format
        if "get_weather" in response_text or "weather" in response_text:
            logger.debug("Weather-related text found in response")
            
            # Extract location using a more flexible regex pattern that handles nested structures
            import re
            
            # Try different patterns to find the location
            patterns = [
                r'"location"\s*:\s*"([^"]+)"',  # Direct pattern: "location": "Place"
                r'"parameters"\s*:\s*\{[^}]*"location"\s*:\s*"([^"]+)"',  # Nested pattern with parameters
                r'location["\']?\s*[=:]\s*["\']([^"\']+)["\']'  # More general pattern
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response_text)
                if matches:
                    location = matches[0]
                    logger.debug(f"Found location with pattern {pattern}: {location}")
                    weather_found = True
                    break
            
            # If we found a location, route to weather
            if location:
                state["location"] = location
                logger.debug(f"Setting location to {location} and routing to weather")
                
                # Don't add the LLM's initial weather report to messages 
                # The actual weather response will be added by get_weather
                return {
                    **state,
                    "request": ["get_weather"],
                    "finished": False
                }
        
        # If we reach here, try with JSON parsing as a fallback
        try:
            parsed = json.loads(response_text)
            
            # Direct field check
            action = parsed.get("action") or parsed.get("tool_code") or parsed.get("type") or parsed.get("tool_name")
            location = parsed.get("location")
            
            # Check nested parameters
            if not location and "parameters" in parsed and isinstance(parsed["parameters"], dict):
                location = parsed["parameters"].get("location")
            
            # Check nested response field with a tool_code block
            if "response" in parsed and isinstance(parsed["response"], str):
                response_content = parsed["response"]
                
                # First try to find tool_code block
                if "tool_code" in response_content:
                    try:
                        tool_code_start = response_content.find("tool_code")
                        json_str = response_content[tool_code_start:].split("```")[0]
                        # Extract the actual JSON part
                        if "{" in json_str:
                            json_part = json_str[json_str.find("{"):].replace("\\\"", "\"")
                            tool_data = json.loads(json_part)
                            
                            # Check tool_data for location
                            if "location" in tool_data:
                                location = tool_data["location"]
                            elif "parameters" in tool_data and "location" in tool_data["parameters"]:
                                location = tool_data["parameters"]["location"]
                            
                            if location:
                                weather_found = True
                    except:
                        logger.debug("Failed to parse tool_code block")
            
            # If we found a weather action and location, prep for routing
            if (action in ["get_weather", "weather"] and location) or weather_found:
                state["location"] = location
                logger.debug(f"Setting location to {location} and routing to weather")
                return {
                    **state,
                    "messages": state["messages"] + [{"role": "assistant", "content": response_text}],
                    "request": ["get_weather"],
                    "finished": False
                }
                
        except json.JSONDecodeError:
            logger.debug("JSON decode error in main parsing")
        
        # If we reach here, no weather request was identified
        response_dict = {
            "role": "assistant",
            "content": response_text
        }
            
    except Exception as e:
        logger.error(f"Error in chatbot: {str(e)}")
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

    # Continue with standard processing if no weather request detected
    messages = state["messages"] + [response_dict]
    
    # Need to check again for action in the response
    try:
        parsed = json.loads(response_text)
        request_type = parsed.get("action") or parsed.get("tool_code") or parsed.get("type")
    except Exception:
        parsed = {}
        request_type = None

    if not request_type:
        follow_up = {
            "role": "assistant", 
            "content": "What would you like to do next? (Options: get_weather, find_events, find_places)"
        }
        messages.append(follow_up)
        request = []
    else:
        request = [request_type]
    
    # Filter out any messages with empty content to prevent empty messages in state
    filtered_messages = [msg for msg in messages if not (isinstance(msg, dict) and msg.get("content", "").strip() == "")]
    
    # Restore user message roles by checking content against original user messages
    for i, msg in enumerate(filtered_messages):
        if isinstance(msg, dict) and msg.get("content") in user_messages:
            filtered_messages[i] = {
                "role": "user",
                "content": msg.get("content")
            }
    
    return {
        "messages": filtered_messages,
        "request": request,
        "user_profile": state.get("user_profile", {}),
        "location": state.get("location", ""),
        "recommendation": state.get("recommendation", {}),
        "finished": False
    }

def profile_collector(state: RequestState) -> RequestState:
    """Asks user for preferences/experiences and stores structured data in user_profile."""
    # Store original user messages to preserve their roles
    user_messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_messages.append(msg.get("content", ""))
            
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
    
    # Convert response to a proper message dictionary
    response_dict = {
        "role": "assistant",
        "content": response_text
    }
    
    # Create new message list with preserved user roles
    messages = state["messages"] + [response_dict]
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("content") in user_messages:
            messages[i] = {
                "role": "user",
                "content": msg.get("content")
            }

    return {
        **state,
        "messages": messages,
        "user_profile": updated_profile,
        "request": ["chatbot"],  # Signal to send back to chatbot
    }

def get_weather(state: RequestState) -> RequestState:
    """Get real-time weather information for a location using OpenWeather API."""
    location = state.get("location", "Unknown")
    logger.debug(f"WEATHER: Getting weather for location: {location}")
    
    # Store original user messages to preserve their roles
    user_messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_messages.append(msg.get("content", ""))
    
    try:
        # Make API call to OpenWeather - this part works fine
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        
        logger.debug(f"WEATHER: Making API call to OpenWeather with params: {params}")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        logger.debug(f"WEATHER: Got response from OpenWeather: {str(weather_data)[:100]}...")
        
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
        
        logger.debug(f"WEATHER: Formatted weather context: {json.dumps(weather_context)}")
        
        # Create a simple text prompt
        weather_prompt_text = f"""You are a helpful travel assistant. 
Here is the current weather data for {location}:
Description: {weather_data["weather"][0]["description"]}
Temperature: {round(weather_data["main"]["temp"])}°C (feels like {round(weather_data["main"]["feels_like"])}°C)
Humidity: {weather_data["main"]["humidity"]}%
Wind Speed: {round(weather_data["wind"]["speed"] * 3.6)} km/h
Cloud Cover: {weather_data["clouds"]["all"]}%

Please provide a helpful, conversational summary of this weather information for a traveler, 
including appropriate clothing suggestions and activity recommendations based on these conditions.
Keep your response concise and natural-sounding. Do not use JSON formatting in your response."""

        logger.debug(f"WEATHER: Sending prompt to LLM: {weather_prompt_text[:100]}...")
        
        # Use a single string prompt
        weather_summary = llm.invoke(weather_prompt_text)
        weather_text = weather_summary.content
        
        logger.debug(f"WEATHER: Got summary from LLM: {weather_text[:100]}...")
        
        # HERE'S THE FIX - Create a properly formatted JSON response
        # No need for the LLM to format as JSON, we'll do it ourselves
        summary_content = {
            "type": "weather_response",
            "message": weather_text
        }
        
        # Convert to JSON string
        json_content = json.dumps(summary_content)
        logger.debug(f"WEATHER: Formatted JSON response: {json_content[:100]}...")
        
        msg = {
            "role": "assistant",
            "content": json_content
        }
        
        # Create message list with preserved user roles
        messages = state["messages"] + [msg]
        for i, message in enumerate(messages):
            if isinstance(message, dict) and message.get("content") in user_messages:
                messages[i] = {
                    "role": "user",
                    "content": message.get("content")
                }

        return {
            **state,
            "messages": messages,
            "recommendation": weather_context,
            "request": [],
            "finished": False
        }
    except requests.RequestException as e:
        logger.error(f"WEATHER ERROR: Request error: {str(e)}")
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
    except Exception as e:
        logger.error(f"WEATHER ERROR: General error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())  # Log full traceback
        error_msg = {
            "role": "assistant", 
            "content": json.dumps({
                "type": "error",
                "message": f"Sorry, I encountered an error processing weather data for {location}. Error: {str(e)}"
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
    
    # Store original user messages to preserve their roles
    user_messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_messages.append(msg.get("content", ""))
    
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

        assistant_message = {
            "role": "assistant",
            "content": json.dumps({
                "type": "events_response",
                "message": f"Here are some events in {location} that match your interests:\n\n{event_descriptions}"
                if filtered_events else
                f"I couldn't find any events in {location} matching your interests at the moment."
            })
        }
        
        # Create message list with preserved user roles
        messages = state["messages"] + [assistant_message]
        for i, message in enumerate(messages):
            if isinstance(message, dict) and message.get("content") in user_messages:
                messages[i] = {
                    "role": "user",
                    "content": message.get("content")
                }

        return {
            **state,
            "messages": messages,
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
        
        # Create message list with preserved user roles for error case
        messages = state["messages"] + [error_msg]
        for i, message in enumerate(messages):
            if isinstance(message, dict) and message.get("content") in user_messages:
                messages[i] = {
                    "role": "user", 
                    "content": message.get("content")
                }
                
        return {**state, "messages": messages, "finished": False}

def log_interaction(data: Dict[Any, Any], interaction_type: str, user_messages=None):
    """Log interaction details to file while keeping JSON structure."""
    # Initialize user_messages if not provided
    if user_messages is None:
        user_messages = []
    
    # Keep track of user messages from USER_INPUT interactions
    if interaction_type == "USER_INPUT" and "user_input" in data:
        user_messages.append(data["user_input"])
    
    try:
        # Create a deep copy that's JSON serializable
        if isinstance(data, dict):
            serializable_data = {}
            for k, v in data.items():
                if k == "messages" and isinstance(v, list):
                    # Process each message and make it serializable
                    serializable_messages = []
                    for msg in v:
                        # Check content against known user messages to ensure proper role assignment
                        if isinstance(msg, dict) and msg.get("content") in user_messages:
                            # This is a user message - force user role
                            serializable_messages.append({
                                "role": "user",
                                "content": msg.get("content")
                            })
                        # Preserve user roles at all times
                        elif isinstance(msg, dict) and msg.get("role") == "user":
                            serializable_messages.append(msg)
                        elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                            # Only modify non-user messages to maintain consistency
                            if msg["role"] not in ["user", "system"]:
                                # Standardize on assistant role
                                msg_copy = msg.copy()
                                msg_copy["role"] = "assistant"
                                serializable_messages.append(msg_copy)
                            else:
                                # Keep system messages as is
                                serializable_messages.append(msg)
                        else:
                            # Try to extract role and content
                            try:
                                # Get the proper role
                                role = getattr(msg, "role", None) or getattr(msg, "type", None)
                                content = getattr(msg, "content", str(msg))
                                
                                # Force user role if content matches known user inputs
                                if content in user_messages:
                                    role = "user"
                                # Map to standard roles
                                elif role == "ai" or role == "model" or role is None:
                                    role = "assistant"
                                # Keep user role if present
                                elif role == "user":
                                    role = "user"
                                # Use assistant role for everything else except system
                                elif role != "system":
                                    role = "assistant"
                                
                                serializable_messages.append({
                                    "role": role,
                                    "content": content
                                })
                            except Exception:
                                # Fallback for completely unknown types
                                content = str(msg)
                                role = "user" if content in user_messages else "assistant"
                                
                                serializable_messages.append({
                                    "role": role,
                                    "content": content
                                })
                    serializable_data[k] = serializable_messages
                else:
                    # Handle other types normally
                    serializable_data[k] = v
            
            # Special handling for user_input - ensure it's always recorded as a user message
            if interaction_type == "USER_INPUT" and "user_input" in serializable_data:
                serializable_data = {
                    "messages": [{
                        "role": "user", 
                        "content": serializable_data["user_input"]
                    }]
                }
            
            json_data = json.dumps(serializable_data, indent=2, default=str)
        else:
            json_data = json.dumps({"data": str(data)}, indent=2)
    except Exception as e:
        # Ultimate fallback
        json_data = json.dumps({"error": f"Failed to serialize: {str(e)}"}, indent=2)
    
    logger.info(f"{interaction_type}: {json_data}")
    
    # Return the user_messages list for potential future use
    return user_messages

def format_response_for_user(response_content: str) -> str:
    """Convert JSON response to human-readable format."""
    # For debugging
    logger.debug(f"FORMAT: Input response content: {response_content[:100]}...")
    
    # Skip processing if empty
    if not response_content or response_content.strip() == "":
        return "No response available."
    
    # Handle potential encoding issues and extract from code blocks
    try:
        # Clean up code blocks if present
        if "```" in response_content:
            parts = response_content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # This is inside a code block
                    if part.strip().startswith("json"):
                        response_content = part.replace("json", "", 1).strip()
                        break
        
        # Try JSON parsing with multiple approaches
        try:
            # Direct JSON parse
            response_dict = json.loads(response_content)
            
            # Check for weather_response format first
            if response_dict.get("type") == "weather_response" and "message" in response_dict:
                return response_dict["message"]
            
            # Check for other standard fields
            for field in ["message", "response", "content", "weather"]:
                if field in response_dict and isinstance(response_dict[field], str):
                    return response_dict[field]
                    
            # If no standard field, return any string value
            for value in response_dict.values():
                if isinstance(value, str) and len(value) > 5:
                    return value
                    
        except json.JSONDecodeError:
            # Try regex extraction if JSON parsing fails
            import re
            
            # Try to extract weather_response directly
            weather_pattern = r'"type"\s*:\s*"weather_response".*?"message"\s*:\s*"([^"]+)"'
            weather_match = re.search(weather_pattern, response_content, re.DOTALL)
            if weather_match:
                return weather_match.group(1).replace('\\n', '\n').replace('\\', '')
                
            # Try to extract any message field
            message_pattern = r'"message"\s*:\s*"([^"]+)"'
            message_match = re.search(message_pattern, response_content, re.DOTALL)
            if message_match:
                return message_match.group(1).replace('\\n', '\n').replace('\\', '')
    except Exception as e:
        logger.debug(f"FORMAT ERROR: {str(e)}")
    
    # Return as is if all else fails
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
        """Route to the appropriate node based on the request."""
        logger.debug(f"ROUTER: request={state.get('request')}, location={state.get('location')}")
        
        # Check if we've just come from the weather node (recommendation would be populated)
        if state.get("recommendation") and "current_weather" in state.get("recommendation", {}):
            logger.debug(f"Weather data already in state, not routing back to weather")
            return END
        
        # Only route to weather if we have an explicit request
        if state.get("request") and len(state["request"]) > 0:
            request_type = state["request"][0]
            
            if request_type in ["get_weather", "weather"]:
                if state.get("location"):
                    logger.debug(f"Routing to weather with location: {state.get('location')}")
                    return "weather"
                else:
                    logger.debug("Weather request but no location found")
            elif request_type == "find_events":
                return "events"
        
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
    
    # Store all user messages to verify message integrity
    user_messages = []
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye! Have a great trip! 👋")
                break
            
            # Log user input properly as a user message
            log_interaction({"user_input": user_input}, "USER_INPUT", user_messages)
            
            # Save the original user input
            user_message = {
                "role": "user",
                "content": user_input
            }
            
            # Keep track of all user messages
            user_messages.append(user_input)
            
            # Update state with user message - make sure the role is explicitly set as "user"
            state["messages"].append(user_message)
            
            # Process through graph
            state = chat_graph.invoke(state, {"recursion_limit": 100})
            
            # Ensure user message role integrity is maintained after graph processing
            if "messages" in state:
                # Fix all user messages by checking content against our stored list
                for i, msg in enumerate(state["messages"]):
                    if isinstance(msg, dict) and msg.get("content") in user_messages:
                        # Ensure it has user role
                        state["messages"][i] = {
                            "role": "user",
                            "content": msg.get("content")
                        }
                
                # Filter out any empty messages
                state["messages"] = [
                    msg for msg in state["messages"] 
                    if not (isinstance(msg, dict) and msg.get("content", "").strip() == "")
                ]
            
            # Log the raw state
            log_interaction(state, "INTERNAL_STATE", user_messages)
            
            # DIRECT APPROACH for displaying responses
            displayed = False
            
            # If we have weather data in the state, prioritize displaying that
            if state.get("recommendation") and "current_weather" in state.get("recommendation", {}):
                # Find the proper weather response message (from API call)
                for msg in list(reversed(state["messages"]))[:3]:  # Look at recent messages
                    if isinstance(msg, dict) and "content" in msg:
                        try:
                            content_obj = json.loads(msg["content"])
                            if content_obj.get("type") == "weather_response":
                                weather_msg = content_obj.get("message", "")
                                if weather_msg:
                                    print(f"\nAssistant: {weather_msg}")
                                    logger.debug(f"Displayed weather response: {weather_msg[:50]}...")
                                    displayed = True
                                    break
                        except:
                            continue
            
            # If no weather response was displayed, show most recent relevant message
            if not displayed:
                # First try to find assistant messages
                for msg in reversed(state["messages"]):
                    if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                        content = msg.get("content", "").strip()
                        if content:
                            formatted = format_response_for_user(content)
                            print(f"\nAssistant: {formatted}")
                            logger.debug(f"Displayed assistant message: {formatted[:50]}...")
                            displayed = True
                            break
                
                # If no assistant message, check for system messages (options)
                if not displayed:
                    for msg in reversed(state["messages"]):
                        if isinstance(msg, dict) and msg.get("role") == "system" and msg.get("content"):
                            # Skip options messages as we display them separately
                            if "would you like to do next" in msg.get("content", "").lower():
                                continue
                            
                            # Show other system messages
                            content = msg.get("content", "").strip()
                            if content:
                                print(f"\nSystem: {content}")
                                logger.debug(f"Displayed system message: {content[:50]}...")
                                displayed = True
                                break
            
            # Fallback message if nothing else was displayed
            if not displayed:
                print("\nAssistant: I'll help you with your travel plans. What would you like to know?")
            
            # Always show options
            print("\nOptions: What would you like to do next? (Options: get_weather, find_events, find_places)")
            
        except Exception as e:
            error_msg = f"Error during chat: {str(e)}"
            print("\nSorry, I encountered an error. Please try again.")
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    interactive_chat()