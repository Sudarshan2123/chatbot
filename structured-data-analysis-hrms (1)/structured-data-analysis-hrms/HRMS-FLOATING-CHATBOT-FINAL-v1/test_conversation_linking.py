"""
Test script to demonstrate conversation linking functionality
Run this after starting your FastAPI server to test the conversation context features
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5050"  # Adjust if your server runs on different port
TEST_USER_ID = "test_user_123"

def test_conversation_linking():
    """Test conversation linking with a realistic HR scenario"""
    
    print("🚀 Testing Conversation Linking with Redis Session Management")
    print("=" * 60)
    
    # 1. Start a new session
    print("\n1. Starting new session...")
    response = requests.post(f"{BASE_URL}/session/start/{TEST_USER_ID}")
    print(f"Session start response: {response.json()}")
    
    # 2. First query - ask about an employee
    print("\n2. First query - asking about employee details...")
    chat_data = {
        "input": "Show me details for employee code 101373",
        "lang": "en-US"
    }
    response = requests.post(f"{BASE_URL}/chat2", json=chat_data)
    first_response = response.json()
    print(f"First response: {first_response.get('answer', 'No answer')[:200]}...")
    
    time.sleep(1)  # Small delay to ensure proper ordering
    
    # 3. Follow-up query using pronouns (should link to previous context)
    print("\n3. Follow-up query using context reference...")
    chat_data = {
        "input": "What is his salary?",
        "lang": "en-US"
    }
    response = requests.post(f"{BASE_URL}/chat2", json=chat_data)
    followup_response = response.json()
    print(f"Follow-up response: {followup_response.get('answer', 'No answer')[:200]}...")
    
    time.sleep(1)
    
    # 4. Another context-dependent query
    print("\n4. Another context-dependent query...")
    chat_data = {
        "input": "Show me his department details",
        "lang": "en-US"
    }
    response = requests.post(f"{BASE_URL}/chat2", json=chat_data)
    dept_response = response.json()
    print(f"Department response: {dept_response.get('answer', 'No answer')[:200]}...")
    
    # 5. Check session statistics
    print("\n5. Checking session statistics...")
    response = requests.get(f"{BASE_URL}/session/stats/{TEST_USER_ID}")
    stats = response.json()
    print(f"Session stats: {json.dumps(stats, indent=2)}")
    
    # 6. Test with a new employee (context should switch)
    print("\n6. Asking about a different employee...")
    chat_data = {
        "input": "Now show me details for employee code 101374",
        "lang": "en-US"
    }
    response = requests.post(f"{BASE_URL}/chat2", json=chat_data)
    new_emp_response = response.json()
    print(f"New employee response: {new_emp_response.get('answer', 'No answer')[:200]}...")
    
    time.sleep(1)
    
    # 7. Follow-up about the new employee
    print("\n7. Follow-up about the new employee...")
    chat_data = {
        "input": "What about her performance rating?",
        "lang": "en-US"
    }
    response = requests.post(f"{BASE_URL}/chat2", json=chat_data)
    perf_response = response.json()
    print(f"Performance response: {perf_response.get('answer', 'No answer')[:200]}...")
    
    # 8. Final session stats
    print("\n8. Final session statistics...")
    response = requests.get(f"{BASE_URL}/session/stats/{TEST_USER_ID}")
    final_stats = response.json()
    print(f"Final session stats: {json.dumps(final_stats, indent=2)}")
    
    # 9. End session
    print("\n9. Ending session...")
    response = requests.post(f"{BASE_URL}/session/end/{TEST_USER_ID}")
    print(f"Session end response: {response.json()}")
    
    print("\n✅ Conversation linking test completed!")
    print("\nKey features demonstrated:")
    print("- Session management with Redis")
    print("- Context-aware follow-up questions")
    print("- Entity extraction and reference resolution")
    print("- Session statistics and monitoring")

def test_streaming_with_context():
    """Test streaming responses with conversation context"""
    
    print("\n🌊 Testing Streaming with Context")
    print("=" * 40)
    
    # Start session
    requests.post(f"{BASE_URL}/session/start/{TEST_USER_ID}_stream")
    
    # First query
    print("\n1. First query (streaming)...")
    chat_data = {
        "input": "Tell me about employee 101373",
        "lang": "en-US"
    }
    
    response = requests.post(f"{BASE_URL}/chat-stream", json=chat_data, stream=True)
    print("Streaming response:")
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                content = decoded_line[6:]  # Remove 'data: ' prefix
                if content != '[DONE]':
                    print(content, end='', flush=True)
    
    print("\n\n2. Follow-up query (streaming)...")
    chat_data = {
        "input": "What is his current position?",
        "lang": "en-US"
    }
    
    response = requests.post(f"{BASE_URL}/chat-stream", json=chat_data, stream=True)
    print("Streaming follow-up response:")
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                content = decoded_line[6:]
                if content != '[DONE]':
                    print(content, end='', flush=True)
    
    print("\n\n✅ Streaming with context test completed!")

def test_session_management():
    """Test session management features"""
    
    print("\n⚙️ Testing Session Management Features")
    print("=" * 40)
    
    # Create multiple test sessions
    test_users = ["user_1", "user_2", "user_3"]
    
    print("\n1. Creating multiple sessions...")
    for user in test_users:
        response = requests.post(f"{BASE_URL}/session/start/{user}")
        print(f"Started session for {user}: {response.json()['status']}")
        
        # Add some conversation to each session
        chat_data = {
            "input": f"Hello, I'm {user}",
            "lang": "en-US"
        }
        requests.post(f"{BASE_URL}/chat2", json=chat_data)
    
    # Check active sessions
    print("\n2. Checking active sessions...")
    response = requests.get(f"{BASE_URL}/session/active")
    active_sessions = response.json()
    print(f"Active sessions count: {active_sessions['count']}")
    print(f"Session details: {json.dumps(active_sessions['active_sessions'], indent=2)}")
    
    # Cleanup expired sessions
    print("\n3. Cleaning up sessions...")
    response = requests.post(f"{BASE_URL}/session/cleanup")
    cleanup_result = response.json()
    print(f"Cleanup result: {cleanup_result}")
    
    # End all test sessions
    print("\n4. Ending all test sessions...")
    for user in test_users:
        response = requests.post(f"{BASE_URL}/session/end/{user}")
        print(f"Ended session for {user}: {response.json()['status']}")
    
    print("\n✅ Session management test completed!")

if __name__ == "__main__":
    try:
        print("Starting conversation linking tests...")
        print("Make sure your FastAPI server is running on http://localhost:5050")
        
        # Test basic conversation linking
        test_conversation_linking()
        
        # Test streaming with context
        test_streaming_with_context()
        
        # Test session management
        test_session_management()
        
        print("\n🎉 All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server.")
        print("Please make sure your FastAPI server is running on http://localhost:5050")
    except Exception as e:
        print(f"❌ Error during testing: {e}")