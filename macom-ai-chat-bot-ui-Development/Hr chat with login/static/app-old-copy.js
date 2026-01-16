
const API_BASE_URL = 'http://localhost:5050';
// Store OTP related state
let otpSent = false;
let otpExpireTimer = null;

// Function to validate employee code
function validateEmployeeCode(code) {
    return code && code.toString().length > 0;
}

// Function to validate OTP
function validateOTP(otp) {
    return otp && otp.length === 4 && /^\d+$/.test(otp);
}

// Function to send OTP

// Function to verify OTP
async function verifyUserAndLogin(userName, password) {
    try {
        debugger;
        // Replace with your actual API endpoint
        const Login = `${API_BASE_URL}/login`
        const response = await fetch(Login, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ userName, password })
        });

        if (!response.ok) {
            throw new Error('Failed to verify OTP');
        }
        debugger;
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error verifying OTP:', error);
        throw error;
    }
}





document.addEventListener('DOMContentLoaded', () => {

    const User_NameInput = document.getElementById('User_Name');
    const PasswordInput = document.getElementById('Password');
    const form = document.querySelector('form');
 

    

if (form) {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const User_Name = User_NameInput.value;
        const Password = PasswordInput.value;

        try {
            const response = await verifyUserAndLogin(User_Name, Password);
            if (response.status=="success") {
                // Replace with your actual success handling
                window.location.href = '/static/chat.html';
            } else {
                alert('Invalid OTP. Please try again.');
            }
        } catch (error) {
            alert('Login failed. Please try again.');
        }
    });
}

});


const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');


        function addMessage(content, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            
            const avatar = document.createElement('div');
            avatar.className = `avatar ${isUser ? 'user-avatar' : 'bot-avatar'}`;
            avatar.textContent = isUser ? 'U' : 'B';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.textContent = content;
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            chatContainer.appendChild(messageDiv);
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function simulateBotResponse(userMessage) {
            // Simulate bot thinking time
            sendButton.disabled = true;
            setTimeout(() => {
                // Simple response logic - you can replace this with actual API calls
                const responses = [
                    "I understand you're saying: " + userMessage,
                    "That's interesting! Tell me more.",
                    "I'm here to help. What else would you like to know?",
                    "Let me think about that..."
                ];
                const randomResponse = responses[Math.floor(Math.random() * responses.length)];
                addMessage(randomResponse, false);
                sendButton.disabled = false;
            }, 1000);
        }

        function sendMessage() {
            const message = userInput.value.trim();
            if (message) {
                addMessage(message, true);
                userInput.value = '';
                simulateBotResponse(message);
            }
        }

        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        // Auto-resize textarea as user types
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });