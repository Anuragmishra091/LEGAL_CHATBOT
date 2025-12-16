import streamlit as st
from main import LegalChatbot

st.set_page_config(page_title="Indian Legal Chatbot", page_icon="⚖️")

# Initialize chatbot
@st.cache_resource
def load_chatbot():
    return LegalChatbot()



## Aise hi
chatbot = load_chatbot()

st.title("⚖️ Indian Legal Chatbot")
st.caption("Powered by BNS, BNSS & BSA + RAG")  # Updated caption

# Sidebar with info
with st.sidebar:
    st.header("📚 Legal Documents")
    st.write("This chatbot covers:")
    st.write("- **BNS** - Bharatiya Nyaya Sanhita (Criminal Law)")
    st.write("- **BNSS** - Bharatiya Nagarik Suraksha Sanhita (Criminal Procedure)")
    st.write("- **BSA** - Bharatiya Sakshya Adhiniyam (Evidence Law)")
    st.divider()
    st.caption("These replaced IPC, CrPC & Evidence Act in 2024")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching legal database..."):
            result = chatbot.ask(prompt)
        
        st.markdown(result['answer'])
        
        # Show sources in expander
        with st.expander("📚 View Sources"):
            for i, src in enumerate(result['sources'], 1):
                st.write(f"**{i}. {src['act']}, Section {src['section']}**")
                st.write(f"Relevance: {src['relevance_score']:.2%}")
                st.caption(src['text'][:300] + "...")
                st.divider()
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result['answer']
    })