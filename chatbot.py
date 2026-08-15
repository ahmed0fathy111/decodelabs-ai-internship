# =========================================
# Project 1: Rule-Based AI Chatbot
# DecodeLabs Internship - Batch 2026
# =========================================

# Knowledge base: dictionary of intents -> responses (O(1) lookup)
responses = {
    'hello': "Hi there! How can I help you today?",
    'hi': "Hello! What can I do for you?",
    'how are you': "I'm just a bunch of if-else logic, but I'm running fine!",
    'what is your name': "I'm DecodeBot, your friendly rule-based assistant.",
    'help': "Sure! You can say hello, ask how I'm doing, ask my name, or type 'bye' to exit.",
    'thank you': "You're welcome!",
    'thanks': "Anytime!",
    'bye': "Goodbye! Have a great day.",
    'exit': "Goodbye! Have a great day.",
}

# Words that should trigger the loop to break
exit_commands = {'bye', 'exit', 'quit'}

def sanitize(text):
    """Clean raw input: lowercase + strip whitespace."""
    return text.lower().strip()

def get_response(user_input):
    """Look up the sanitized input in the knowledge base with a fallback."""
    return responses.get(user_input, "I do not understand. Type 'help' for options.")

def run_chatbot():
    print("DecodeBot: Hello! Type 'bye' or 'exit' to end our chat.\n")

    while True:  # The Heartbeat: infinite loop until kill command
        raw_input_text = input("You: ")
        clean_input = sanitize(raw_input_text)

        if clean_input in exit_commands:
            print(f"DecodeBot: {responses.get(clean_input)}")
            break  # Kill Command

        reply = get_response(clean_input)
        print(f"DecodeBot: {reply}")

if __name__ == "__main__":
    run_chatbot()