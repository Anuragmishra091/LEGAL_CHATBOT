from chatbot import LegalChatbot

def test_queries():
    chatbot = LegalChatbot()
    
    test_questions = [
        "What is the definition and punishment for murder under BNS?",
        "Explain offenses against women under Bharatiya Nyaya Sanhita.",
        
        # BNSS prompts
        "What is the procedure for arrest under BNSS?",
        "Explain the provisions for anticipatory bail under BNSS.",
        
        # BSA prompts
        "What types of evidence are admissible under Bharatiya Sakshya Adhiniyam?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {question}")
        print('='*70)
        
        result = chatbot.ask(question)
        print(result['answer'])
        print(f"\nSources: {', '.join([s['act'] for s in result['sources']])}")

if __name__ == "__main__":
    test_queries()