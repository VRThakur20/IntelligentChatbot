from flask import Flask, request, render_template, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
from functools import lru_cache
import logging
import os
from werkzeug.utils import secure_filename
import torch
from datetime import datetime
import json
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import pygame  # Use pygame instead of playsound
import time
import random

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    logger.error("FLASK_SECRET_KEY environment variable not set")
    raise ValueError("FLASK_SECRET_KEY environment variable not set")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_audio'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Initialize pygame mixer
pygame.mixer.init()


# Initialize models with caching
@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    return pipeline(
        "text-classification",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=0 if torch.cuda.is_available() else -1
    )


@lru_cache(maxsize=1)
def get_chatbot_models():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return tokenizer, model


# Initialize models
sentiment_analyzer = get_sentiment_analyzer()
chatbot_tokenizer, chatbot_model = get_chatbot_models()


class ConversationHistory:

    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history

    def add_interaction(self, user_input, bot_response, sentiment):
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'sentiment': sentiment
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_history(self):
        return self.history


conversation_history = ConversationHistory()

# Containers for jokes, music lyrics, and inspirational quotes
jokes = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "Why don't skeletons fight each other? They don't have the guts.",
    "What do you call fake spaghetti? An impasta!",
    "Why did the bicycle fall over? It was two-tired!",
    "What do you call cheese that isn't yours? Nacho cheese!",
    "Why can't your nose be 12 inches long? Because then it would be a foot!"
]

songs = [
    "🎵 Don't stop believin', hold on to that feelin' 🎵",
    "🎵 We will, we will rock you! 🎵",
    "🎵 Let it be, let it be, let it be, let it be 🎵",
    "🎵 I'm walking on sunshine, whoa-oh 🎵",
    "🎵 Sweet Caroline, bum bum bum 🎵",
    "🎵 Shake it off, shake it off 🎵",
    "🎵 I want to break free 🎵"
]

quotes = [
    "The best way to predict the future is to invent it. - Alan Kay",
    "Life is 10% what happens to us and 90% how we react to it. - Charles R. Swindoll",
    "The only way to do great work is to love what you do. - Steve Jobs",
    "Success is not the key to happiness. Happiness is the key to success. - Albert Schweitzer",
    "Believe you can and you're halfway there. - Theodore Roosevelt",
    "Your time is limited, don't waste it living someone else's life. - Steve Jobs",
    "The only limit to our realization of tomorrow is our doubts of today. - Franklin D. Roosevelt"
]

# def generate_response(user_input):
#     try:
#         # Handle reminders
#         if "set a reminder for" in user_input.lower():
#             try:
#                 # Extract time and message from the input
#                 parts = user_input.lower().split("set a reminder for")
#                 if len(parts) > 1:
#                     time_and_message = parts[1].strip()
#                     if "with a message" in time_and_message:
#                         time_part, message_part = time_and_message.split("with a message", 1)
#                         reminder_time = datetime.strptime(time_part.strip(), "%I:%M %p")  # Example: "2:00 PM"

#                         # Ask for date and day
#                         return "Please provide the date and day for the reminder in the format '12 February, Wednesday, 2025'."

#             except ValueError:
#                 return "The time format is incorrect. Please use the format '2:00 PM'."
#             except Exception as e:
#                 logger.error(f"Error parsing reminder command: {str(e)}")
#                 return "I couldn't understand the reminder time or message. Please try again."

#         # Handle date and day input for reminders
#         if "," in user_input and any(day in user_input.lower() for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
#             try:
#                 # Adjust parsing logic to handle the correct format
#                 parts = user_input.split(", ")
#                 if len(parts) == 3:
#                     date_str, day_str, year_str = parts
                    
#                     reminder_date = datetime.strptime(f"{date_str} {year_str}", "%d %B %Y")
                    
#                     if reminder_date.strftime("%A").lower() != day_str.strip().lower():
#                         return "The day does not match the date. Please try again."

#                     # Combine date and time for the reminder
#                     reminder_datetime = reminder_date.replace(hour=reminder_time.hour, minute=reminder_time.minute)
#                     if reminder_datetime < datetime.now():
#                         return "The reminder time is in the past. Please provide a future date and time."

#                     # Schedule the reminder
#                     trigger = DateTrigger(run_date=reminder_datetime)
#                     scheduler.add_job(send_reminder, trigger, args=[message_part.strip()])

#                     return f"Reminder set for {reminder_datetime.strftime('%d %B %Y (%A) %I:%M %p')} with message: '{message_part.strip()}'."
#                 else:
#                     return "Invalid date format. Please use the format '12 February, Wednesday, 2025'."
#             except ValueError:
#                 return "Invalid date format. Please use the format '12 February, Wednesday, 2025'."
#             except Exception as e:
#                 logger.error(f"Error parsing date and day: {str(e)}")
#                 return "I couldn't understand the date or day. Please try again."


# def generate_response(user_input):
#     global reminder_time, message_part  # Ensure these variables are accessible

#     try:
#         # Handle reminders
#         if "set a reminder for" in user_input.lower():
#             try:
#                 # Extract time and message from the input
#                 parts = user_input.lower().split("set a reminder for")
#                 if len(parts) > 1:
#                     time_and_message = parts[1].strip()
#                     if "with a message" in time_and_message:
#                         time_part, message_part = time_and_message.split("with a message", 1)
#                         reminder_time = datetime.strptime(time_part.strip(), "%I:%M %p")  # Example: "2:00 PM"
#                         # Ask for date and day
#                         return "Please provide the date and day for the reminder in the format '21 April, Monday, 2025'."
#             except ValueError:
#                 return "The time format is incorrect. Please use the format '2:00 PM'."
#             except Exception as e:
#                 logger.error(f"Error parsing reminder command: {str(e)}")
#                 return "I couldn't understand the reminder time or message. Please try again."
        
#         # Handle date and day input for reminders
#         if "," in user_input:
#             try:
#                 # Split the input by commas and clean up each part
#                 date_parts = [part.strip() for part in user_input.split(",")]
                
#                 if len(date_parts) == 3:
#                     date_str = date_parts[0].strip()  # "21 April"
#                     day_str = date_parts[1].strip()  # "Monday"
#                     year_str = date_parts[2].strip()  # "2025"
                    
#                     # Remove any quotes that might be in the string
#                     date_str = date_str.replace('"', '').replace("'", '')
#                     day_str = day_str.replace('"', '').replace("'", '')
#                     year_str = year_str.replace('"', '').replace("'", '')
                    
#                     # Create the full date string without quotes
#                     full_date_str = f"{date_str} {year_str}"
                    
#                     try:
#                         # Try to parse the date with the day first format
#                         reminder_date = datetime.strptime(full_date_str, "%d %B %Y")
#                     except ValueError:
#                         try:
#                             # Alternative: Try month first format
#                             reminder_date = datetime.strptime(full_date_str, "%B %d %Y")
#                         except ValueError:
#                             # More detailed error message
#                             return f"Could not parse date '{full_date_str}'. Please use the format 'day month, weekday, year' like '21 April, Monday, 2025'."
                    
#                     # Make sure reminder_time exists
#                     if 'reminder_time' not in locals() and 'reminder_time' not in globals():
#                         reminder_time = datetime.now().replace(hour=12, minute=0)  # Default to noon
                    
#                     # Make sure message_part exists
#                     if 'message_part' not in locals() and 'message_part' not in globals():
#                         message_part = "No message provided"
                    
#                     # Check if the day matches
#                     actual_day = reminder_date.strftime("%A")
#                     if actual_day.lower() != day_str.lower():
#                         return f"The day does not match the date. {reminder_date.strftime('%d %B %Y')} is a {actual_day}, not a {day_str}."
                    
#                     # Combine date and time for the reminder
#                     reminder_datetime = reminder_date.replace(hour=reminder_time.hour, minute=reminder_time.minute)
                    
#                     if reminder_datetime < datetime.now():
#                         return "The reminder time is in the past. Please provide a future date and time."
                    
#                     # Schedule the reminder
#                     # Commented out for testing:
#                     # trigger = DateTrigger(run_date=reminder_datetime)
#                     # scheduler.add_job(send_reminder, trigger, args=[message_part.strip()])
                    
#                     return f"Reminder set for {reminder_datetime.strftime('%d %B %Y (%A) %I:%M %p')} with message: '{message_part.strip()}'."
                
#                 else:
#                     return f"Expected 3 parts in date format but got {len(date_parts)}. Please use the format '21 April, Monday, 2025'."
            
#             except Exception as e:
#                 logger.error(f"Error in date parsing: {str(e)}")
#                 return f"Error: {str(e)}. Please use the format '21 April, Monday, 2025'."
        
#         return "I'm not sure what you're asking. You can set a reminder by saying 'Set a reminder for [time] with a message [your message]'"
    
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         return f"An unexpected error occurred: {str(e)}"

#         # Handle greetings
#         greetings = ["hey", "hello", "hi", "greetings"]
#         if user_input.lower() in greetings:
#             return "Hey! How can I help you?"

#         # Handle sad messages
#         if "i am sad" in user_input.lower():
#             return "Well, I can enlighten your mood. Should I tell you a joke, play some music, or share an inspirational quote?"

#         # Handle jokes
#         if "tell me a joke" in user_input.lower():
#             return random.choice(jokes)

#         # Handle music
#         if ("play some music", "other music") in user_input.lower():
#             return random.choice(songs)

#         # Handle inspirational quotes
#         if "share an inspirational quote" in user_input.lower():
#             return random.choice(quotes)

#         # Encode user input
#         input_ids = chatbot_tokenizer.encode(
#             user_input + chatbot_tokenizer.eos_token,
#             return_tensors="pt"
#         )
#         if torch.cuda.is_available():
#             input_ids = input_ids.to('cuda')

#         # Generate response
#         response_ids = chatbot_model.generate(
#             input_ids,
#             max_length=100,
#             pad_token_id=chatbot_tokenizer.eos_token_id,
#             temperature=0.7,
#             top_k=50,
#             top_p=0.9,
#             do_sample=True
#         )
#         response = chatbot_tokenizer.decode(
#             response_ids[:, input_ids.shape[-1]:][0],
#             skip_special_tokens=True
#         )

#         # Ensure the response is not a direct echo of the input
#         if response.strip().lower() == user_input.strip().lower():
#             response = "I'm here to help. Could you tell me more about how you're feeling?"

#         return response
def generate_response(user_input):
    global reminder_time, message_part  # Ensure these variables are accessible

    try:
        # Handle greetings first
        greetings = ["hey", "hello", "hi", "greetings"]
        if user_input.lower() in greetings:
            return "Hey! How can I help you?"

        # Handle sad messages
        if "i am sad" in user_input.lower():
            return "Well, I can enlighten your mood. Should I tell you a joke, play some music, or share an inspirational quote?"

        # Handle jokes
        if "tell me a joke" in user_input.lower():
            return random.choice(jokes)

        # Handle music
        if any(phrase in user_input.lower() for phrase in ("play some music", "other music")):
            return random.choice(songs)

        # Handle inspirational quotes
        if "share an inspirational quote" in user_input.lower():
            return random.choice(quotes)

        # Handle reminders
        if "set a reminder for" in user_input.lower():
            try:
                # Extract time and message from the input
                parts = user_input.lower().split("set a reminder for")
                if len(parts) > 1:
                    time_and_message = parts[1].strip()
                    if "with a message" in time_and_message:
                        time_part, message_part = time_and_message.split("with a message", 1)
                        reminder_time = datetime.strptime(time_part.strip(), "%I:%M %p")  # Example: "2:00 PM"
                        # Ask for date and day
                        return "Please provide the date and day for the reminder in the format '21 April, Monday, 2025'."
            except ValueError:
                return "The time format is incorrect. Please use the format '2:00 PM'."
            except Exception as e:
                logger.error(f"Error parsing reminder command: {str(e)}")
                return "I couldn't understand the reminder time or message. Please try again."
        
        # Handle date and day input for reminders
        if "," in user_input:
            try:
                # Split the input by commas and clean up each part
                date_parts = [part.strip() for part in user_input.split(",")]
                
                if len(date_parts) == 3:
                    date_str = date_parts[0].strip()  # "21 April"
                    day_str = date_parts[1].strip()  # "Monday"
                    year_str = date_parts[2].strip()  # "2025"
                    
                    # Remove any quotes that might be in the string
                    date_str = date_str.replace('"', '').replace("'", '')
                    day_str = day_str.replace('"', '').replace("'", '')
                    year_str = year_str.replace('"', '').replace("'", '')
                    
                    # Create the full date string without quotes
                    full_date_str = f"{date_str} {year_str}"
                    
                    try:
                        # Try to parse the date with the day first format
                        reminder_date = datetime.strptime(full_date_str, "%d %B %Y")
                    except ValueError:
                        try:
                            # Alternative: Try month first format
                            reminder_date = datetime.strptime(full_date_str, "%B %d %Y")
                        except ValueError:
                            # More detailed error message
                            return f"Could not parse date '{full_date_str}'. Please use the format 'day month, weekday, year' like '21 April, Monday, 2025'."
                    
                    # Make sure reminder_time exists
                    if 'reminder_time' not in locals() and 'reminder_time' not in globals():
                        reminder_time = datetime.now().replace(hour=12, minute=0)  # Default to noon
                    
                    # Make sure message_part exists
                    if 'message_part' not in locals() and 'message_part' not in globals():
                        message_part = "No message provided"
                    
                    # Check if the day matches
                    actual_day = reminder_date.strftime("%A")
                    if actual_day.lower() != day_str.lower():
                        return f"The day does not match the date. {reminder_date.strftime('%d %B %Y')} is a {actual_day}, not a {day_str}."
                    
                    # Combine date and time for the reminder
                    reminder_datetime = reminder_date.replace(hour=reminder_time.hour, minute=reminder_time.minute)
                    
                    if reminder_datetime < datetime.now():
                        return "The reminder time is in the past. Please provide a future date and time."
                    
                    # Schedule the reminder
                    trigger = DateTrigger(run_date=reminder_datetime)
                    scheduler.add_job(send_reminder, trigger, args=[message_part.strip()])
                    
                    return f"Reminder set for {reminder_datetime.strftime('%d %B %Y (%A) %I:%M %p')} with message: '{message_part.strip()}'."
                
                else:
                    return f"Expected 3 parts in date format but got {len(date_parts)}. Please use the format '21 April, Monday, 2025'."
            
            except Exception as e:
                logger.error(f"Error in date parsing: {str(e)}")
                return f"Error: {str(e)}. Please use the format '21 April, Monday, 2025'."
        
        # If no specific commands were matched, use the chatbot model
        try:
            # Encode user input
            input_ids = chatbot_tokenizer.encode(
                user_input + chatbot_tokenizer.eos_token,
                return_tensors="pt"
            )
            if torch.cuda.is_available():
                input_ids = input_ids.to('cuda')

            # Generate response
            response_ids = chatbot_model.generate(
                input_ids,
                max_length=100,
                pad_token_id=chatbot_tokenizer.eos_token_id,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
            response = chatbot_tokenizer.decode(
                response_ids[:, input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )

            # Ensure the response is not a direct echo of the input
            if response.strip().lower() == user_input.strip().lower():
                response = "I'm here to help. Could you tell me more about how you're feeling?"

            return response
        except Exception as e:
            logger.error(f"Error generating chatbot response: {str(e)}")
            return "I'm not sure what you're asking. You can set a reminder by saying 'Set a reminder for [time] with a message [your message]'"
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now."


def analyze_sentiment(user_input):
    try:
        result = sentiment_analyzer(user_input)[0]
        sentiment_mapping = {
            '5 stars': "Happy",
            '4 stars': "Happy",
            '3 stars': "Neutral",
            '2 stars': "Sad",
            '1 star': "Sad"
        }
        sentiment = sentiment_mapping.get(result['label'], "Neutral")
        return sentiment, float(result['score'])
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return "Neutral", 0.0


def convert_voice_to_text(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand audio")
        return None
    except sr.RequestError as e:
        logger.error(f"Speech recognition error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error in voice conversion: {str(e)}")
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        sentiment, score = analyze_sentiment(user_input)
        response = generate_response(user_input)

        conversation_history.add_interaction(user_input, response, sentiment)

        return jsonify({
            "input_text": user_input,
            "classification": sentiment,
            "score": score,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/voice_chat", methods=["POST"])
def voice_chat():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = convert_voice_to_text(file_path)
        os.remove(file_path)  # Clean up file

        if not text:
            return jsonify({"error": "Could not understand the audio"}), 400

        sentiment, score = analyze_sentiment(text)
        response = generate_response(text)

        conversation_history.add_interaction(text, response, sentiment)

        return jsonify({
            "input_text": text,
            "classification": sentiment,
            "score": score,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in voice_chat endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/chat_history", methods=["GET"])
def get_chat_history():
    return jsonify(conversation_history.get_history())


@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        conversation_history.history.clear()
        return jsonify({"message": "Chat history cleared successfully"}), 200
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/download_history", methods=["GET"])
def download_history():
    try:
        history = conversation_history.get_history()
        response = jsonify(history)
        response.headers['Content-Disposition'] = 'attachment; filename=chat_history.json'
        return response
    except Exception as e:
        logger.error(f"Error downloading chat history: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# Function to send reminder
def send_reminder(user_input):
    logger.info(f"Reminder: {user_input}")
    pygame.mixer.music.load('audio/reminder.mp3')  # Updated path to your audio file
    pygame.mixer.music.play()


@app.route("/set_reminder", methods=["POST"])
def set_reminder():
    try:
        message = request.form.get("message", "").strip()
        reminder_time = request.form.get("time", "").strip()  # Expected format: 'YYYY-MM-DD HH:MM:SS'

        if not message or not reminder_time:
            return jsonify({"error": "Message and time are required"}), 400

        # Schedule the reminder
        trigger = DateTrigger(run_date=reminder_time)
        scheduler.add_job(send_reminder, trigger, args=[message])

        return jsonify({"message": "Reminder set successfully", "time": reminder_time}), 200
    except Exception as e:
        logger.error(f"Error in set_reminder endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say the reminder time (e.g., 'set a reminder for 1:50 PM'):")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError:
        print("Could not request results; check your network connection.")


def parse_command(command):
    # Simple parsing logic to extract time from the command
    if "set a reminder for" in command:
        time_str = command.split("set a reminder for ")[1]
        reminder_time = datetime.strptime(time_str, "%I:%M %p")
        now = datetime.now()
        reminder_time = reminder_time.replace(year=now.year, month=now.month, day=now.day)
        if reminder_time < now:
            reminder_time = reminder_time.replace(day=now.day + 1)
        return reminder_time
    return None


@app.route("/voice_set_reminder", methods=["POST"])
def voice_set_reminder():
    try:
        command = recognize_speech()
        if command:
            reminder_time = parse_command(command)
            if reminder_time:
                set_reminder(reminder_time, "This is your reminder!")
                return jsonify({"message": f"Reminder set for {reminder_time.strftime('%I:%M %p')}"})
            else:
                return jsonify({"error": "Could not parse the reminder time."}), 400
        else:
            return jsonify({"error": "Could not understand the audio."}), 400
    except Exception as e:
        logger.error(f"Error in voice_set_reminder endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    scheduler.shutdown()
    
    try:
        # Keep the script running
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        pass
