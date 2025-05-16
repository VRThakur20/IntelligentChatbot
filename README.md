**🤖 Intelligent Chatbot with Sentiment Analysis and Task Scheduling**

A Flask-based AI chatbot that enables emotionally aware, human-like conversations and supports natural language-based task scheduling with audio reminders. Built using BERT for sentiment detection and DialoGPT for dialogue generation, the system is accessible through both text and voice input, and designed for productivity and inclusivity.

**🔧Features**

1. Sentiment Analysis: Detects emotional tone (Happy, Neutral, Sad) using BERT.
2. Conversational AI: Uses DialoGPT to generate intelligent and context-aware replies.
3. Task Scheduling: Set reminders with natural language and receive audio alerts.
4. Voice Input Support: Converts speech to text using Google Speech Recognition.
5. Audio Notifications: Reminder alerts played via Pygame TTS.
6. Conversation History: View, download, and clear past chat sessions.
7. Web Interface: Simple, intuitive frontend built with HTML/CSS and Flask templates.
    

📂 **Project Structure**

├── app.py                    # Main Flask app with routes and logic
├── templates/
│   └── index.html            # Frontend HTML file
├── audio/
│   └── reminder.mp3          # Audio file for reminders
├── temp_audio/               # Temporary storage for uploaded voice files
├── .env                      # Environment variables (e.g., Flask secret key)
├── requirements.txt          # Python dependencies


🚀 **Getting Started**

Python 3.9+

pip package manager

-Installation:

<pre> 
git clone https://github.com/yourusername/intelligent-chatbot.git
cd intelligent-chatbot
pip install -r requirements.txt
 </pre>

-Setup:

<pre> 
Create a .env file:

ini

FLASK_SECRET_KEY=your_secret_key
</pre>

Make sure to add a reminder audio file in audio/reminder.mp3.

Run the App

<pre>
python app.py
Visit http://localhost:5000/ in your browser to start chatting.  </pre>

🧠 **Technologies Used**

-Flask – Backend web framework
-Hugging Face Transformers
-BERT – Sentiment classification
-DialoGPT – Conversation generation
-Google Speech Recognition – Voice-to-text input
-APScheduler – Background task scheduling
-Pygame – Voice/audio playback
-Torch (PyTorch) – AI model inference

**📷Sample Use Cases**

User Input	Chatbot Behavior
-"Hi"	Responds with a greeting
-"I'm feeling sad"	Offers a joke, music, or quote
-"Tell me a joke"	Delivers a random joke
-"Set a reminder for 2:00 PM with a message Drink water"	Schedules a reminder
-"21 April, Monday, 2025"	Confirms and finalizes the reminder

**🛡 Limitations**

-Only supports English.
-Performance may vary with strong accents or noisy audio input.
-Reminders are scheduled locally (no cloud sync).

**📌Future Enhancements**

-Multilingual support using mBERT
-Calendar/email integration
-PWA or mobile app version
-Sentiment detection from voice tone

**👨‍💻Author**

**Van Raj Thakur**

**Supervised by: Ms. Richa Nigam**

📃 License
**This project is licensed under the MIT License.**
