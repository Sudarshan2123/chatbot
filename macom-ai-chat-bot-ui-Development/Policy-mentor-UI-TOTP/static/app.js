// // Constants
const CONFIG = {
    API_BASE_URL: 'https://policy-mentor-446976656513.us-central1.run.app',
    //API_BASE_URL: 'http://localhost:5050',
    OTP_LENGTH: 4,
    TYPING_DELAY: 1000
};

// Token Manager for secure token handling
class TokenManager {
    static storeToken(token) {
        sessionStorage.setItem('auth_token', token);
    }
    
    static getToken() {
        return sessionStorage.getItem('auth_token');
    }
    
    static removeToken() {
        sessionStorage.removeItem('auth_token');
    }
}

class AuthService {
    static state = {
        isLoggedIn: false,
        accessToken: null
    };

    static encryptCredentials(username, password) {
        // In a real application, use proper encryption libraries
        const shift = 5;
        
        function encryptString(str, shift) {
            // Using browser's btoa() instead of Buffer
            return btoa(str)
                .split('')
                .map(char => {
                    const code = char.charCodeAt(0);
                    return String.fromCharCode(code + shift);
                })
                .join('');
        }
        
        return {
            userName: encryptString(username, shift),
            password: encryptString(password, shift)
        };
    }

    static async login(userName, password) {
        try {
            const encrypted = this.encryptCredentials(userName, password);
            const response = await fetch(`${CONFIG.API_BASE_URL}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    userName: encrypted.userName, 
                    password: encrypted.password 
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const authHeader = response.headers.get('Authorization');
            const token = authHeader?.startsWith('Bearer ') ? authHeader.substring(7) : null;
            const data = await response.json();

            if (data.status === "success" && token) {
                this.state.isLoggedIn = true;
                this.state.accessToken = token;
                return { data, token };
            }
            
            throw new Error('Login failed: Invalid response format');
        } catch (error) {
            console.error('Authentication error:', error);
            throw new Error('Login failed: Please check your credentials and try again.');
        }
    }
    static async auth_token(auth_token) {
        try {
            debugger;
            // const encrypted = this.encryptCredentials(userName, password);
            const response = await fetch(`${CONFIG.API_BASE_URL}/validate-token-ldap`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${auth_token}`, // Ensure accessToken is correctly set
                },
                body: JSON.stringify({ 
                }),
                mode: 'cors',
                
            });
            debugger;
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const authHeader = response.headers.get('Authorization');
            const token = authHeader?.startsWith('Bearer ') ? authHeader.substring(7) : null;
            const data = await response.json();

            if (data.status === "success" && token) {
                this.state.isLoggedIn = true;
                this.state.accessToken = token;
                return { data, token };
            }
            
            throw new Error('Login failed: Invalid response format');
        } catch (error) {
            console.error('Authentication error:', error);
            throw new Error('Login failed: Please check your credentials and try again.');
        }
    }

    static validateCredentials(username, password) {
        return Boolean(username?.trim() && password?.trim());
    }
}

const languageConfig = {
    1: {
        code: 'en',
        name: 'English',
        apiParam: 'en-US'
    },
    2: {
        code: 'ml',
        name: 'Malayalam',
        apiParam: 'ml-IN'
    },
    3: {
        code: 'hi',
        name: 'Hindi',
        apiParam: 'hi-IN'
    }
};

class ChatService {
    
    static async getBotResponse(message) {
        try {
            const lang_code = sessionStorage.getItem('lang_code') || 'en-US';
            // Make the fetch request
            const response = await fetch(`${CONFIG.API_BASE_URL}/chat`, {
                method: 'POST',
                body: JSON.stringify({
                    input: encodeURIComponent(message),
                    lang:encodeURIComponent(lang_code)
                }),
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${TokenManager.getToken()}`, // Ensure accessToken is correctly set
                },
            });

            // Check if the response is successful
            if (!response.ok) {
                throw new Error('Failed to fetch response');
            }

            // Extract the token from headers if available
            const authHeader = response.headers.get('Authorization');
            if (authHeader && authHeader.startsWith('Bearer ')) {
                const token = authHeader.substring(7); // Remove 'Bearer ' from the start
                TokenManager.storeToken(token); // Store the new token
            }

            // Parse the JSON response
            const data = await response.json();
            const parser = new DOMParser();
            const decodedAnswer = parser.parseFromString(data.answer, 'text/html').body.textContent;
            // Check if the 'answer' key exists in the response
            if (data && data.answer) {
                return decodedAnswer; // Return the 'answer' value
            } else {
                throw new Error('Answer key not found in the response');
            }

        } catch (error) {
            console.error('Error in getBotResponse:', error);
            window.location.href = '/Macom-RBI-BOT';
            return { error: 'An error occurred while fetching the response.' };
        }
    }
}


class ChatUI {
    constructor() {
        //this.signOutLink = document.querySelector('.menu-item .menu-link:not(.text-hover-chatbot) + .menu-link');
        this.signOutLink = Array.from(document.querySelectorAll('.menu-item .menu-link.text-hover-chatbot[href="#"]'))
        .find(link => link.textContent.trim() === 'Sign Out');
        this.newChatButton = document.getElementById('kt_toolbar_primary_button');
        this.messageContainer = document.querySelector('.chatbox-messages');
        this.messageInput = document.querySelector('.chatbox .form-control');
        this.userNameDisplay = document.querySelector('.card-label.fw-bolder.fs-3.mb-1');
        this.languageSelect = document.querySelector('select[data-control="select2"]');
        this.microphoneButton = document.querySelector('.btn-chatbox-audio');
        this.asideToggle = document.getElementById('kt_aside_toggle');
        this.aside = document.getElementById('kt_aside');
        this.state = {
            isRecording: false,
            language: 'en',
            isLoading :false
        };
        this.setupEventListeners();
        this.initializeChatbot();

    }
    
    initializeSidebarToggle() {
        if (this.asideToggle && this.aside) {
            // Add click event to toggle the 'active' class
            this.asideToggle.addEventListener('click', () => {
                this.aside.classList.toggle('active');
            });
            
            // Optional: Close the sidebar when clicking outside of it
            document.addEventListener('click', (event) => {
                if (!this.aside.contains(event.target) && event.target !== this.asideToggle) {
                    this.aside.classList.remove('active');
                }
            });
        }
    }
    setupEventListeners() {
        debugger;
        if (this.messageInput) {
            this.messageInput.addEventListener('keypress', this.handleKeyPress.bind(this));
        }

        if (this.microphoneButton) {
            this.microphoneButton.addEventListener('click', this.handleMicrophoneClick.bind(this));
        }
        if (this.signOutLink) {
            this.signOutLink.addEventListener('click', (e) => this.handlesignout(e));
        }
        if (this.newChatButton) {
            this.newChatButton.addEventListener('click', (e) => this.handlenewchat(e));
        }
        // if (this.languageSelect) {
        // this.languageSelect.addEventListener('change', (event) => this.changeLanguage(event));
        // }
        if (this.languageSelect) {
            // For select2 dropdowns, you need to use the select2:select event
            $(this.languageSelect).on('select2:select', (event) => this.changeLanguage(event));
            // Also listen for regular change events as backup
            this.languageSelect.addEventListener('change', (event) => this.changeLanguage(event));
        }
  
    }

        // Clear all messages in the chat box


    async handlenewchat(e) {
            e.preventDefault();
            // You're using messageContainer and chatboxMessages variables 
            // that don't exist in this scope
            
            // Should use this.messageContainer instead
            this.messageContainer.innerHTML = '';
            
            // Add back the initial greeting message
            const greetingMessage = document.createElement('span');
            greetingMessage.className = 'chatbox-msgitem-rcv mb-2';
            greetingMessage.textContent = 'Hai, What can I do for you?';
            this.messageContainer.appendChild(greetingMessage);
            this.refresh_chat();
        }
    
    async refresh_chat() {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/clear_history`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${TokenManager.getToken()}`,
                },
            });
    
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
    
            const authHeader = response.headers.get('Authorization');
            const token = authHeader?.startsWith('Bearer ') ? authHeader.substring(7) : null;
            TokenManager.storeToken(token);
            const data = await response.json();
        } catch (error) {
            console.error('Authentication error:', error);
            throw new Error('refresh failed: Please check your credentials and try again.');
        }
    }
    
    changeLanguage(event) {
        debugger;
        const languageMap = {
            '1': 'en-US',
            '2': 'ml-IN',
            '3': 'hi-IN',
        };
        
        const selectedLanguage = event.target.value;
        this.state.language = languageMap[selectedLanguage] || 'en-US';
        sessionStorage.setItem('lang_code', this.state.language);
        console.log('Language changed to:', this.state.language);
        
        // If currently recording, restart recognition with new language
        if (this.state.isRecording) {
            this.stopVoiceRecognition().then(() => {
                this.startVoiceRecognition();
            });
        }
    }

    showLoadingIndicator() {
        if (!this.messageContainer || this.isLoading) return;
        
        this.isLoading = true;
        const loadingElement = document.createElement('div');
        loadingElement.className = 'chatbox-loading';
        loadingElement.id = 'chatLoadingIndicator';
        
        // Create three dots
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'dot';
            loadingElement.appendChild(dot);
        }
        
        this.messageContainer.appendChild(loadingElement);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }

    hideLoadingIndicator() {
        if (!this.messageContainer) return;
        
        const loadingElement = document.getElementById('chatLoadingIndicator');
        if (loadingElement) {
            loadingElement.remove();
            this.isLoading = false;
        }
    }
    
    async handlesignout(e) {
        e.preventDefault();
        sessionStorage.removeItem('User_Name');
        sessionStorage.removeItem('auth_token');
        window.location.href = '/Macom-RBI-BOT';
    }
    
    async handleMicrophoneClick(e) {
        e.preventDefault();
        
        if (!('webkitSpeechRecognition' in window)) {
            console.error('Speech recognition not supported in this browser');
            return;
        }

        if (this.state.isRecording) {
            await this.stopVoiceRecognition();
        } else {
            await this.startVoiceRecognition();
        }
    }
    
    async startVoiceRecognition() {
        try {
            this.recognition = new webkitSpeechRecognition();
            this.recognition.lang = this.state.language;
            this.recognition.interimResults = false;
            this.recognition.maxAlternatives = 1;
        
            // Start recording
            this.recognition.start();
            this.state.isRecording = true;
            this.microphoneButton?.classList.add('recording');
        
            // Handle speech recognition results
            this.recognition.onresult = (event) => {
                const speechResult = event.results[0][0].transcript;
                console.log('Result received:', speechResult);
                this.handleSpeechResult(speechResult);
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.stopVoiceRecognition();
            };
        } catch (error) {
            console.error('Error starting voice recognition:', error);
            this.state.isRecording = false;
        }
    }
    
    async stopVoiceRecognition() {
        if (this.recognition) {
            try {
                this.recognition.stop();
            } catch (error) {
                console.error('Error stopping recognition:', error);
            } finally {
                this.state.isRecording = false;
                this.microphoneButton?.classList.remove('recording');
                console.log('Voice recognition stopped');
            }
        }
    }
    
    handleSpeechResult(speechResult) {
        if (speechResult && this.messageInput) {
            this.messageInput.value = speechResult;
            this.sendMessage('voice');
        }
    }

    handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage('Text');
        }
    }
    setGreeting(userName) {
        const hour = new Date().getHours();
        let greeting = '';
        if (hour < 12) greeting = 'Good Morning';
        else if (hour < 18) greeting = 'Good Afternoon';
        else greeting = 'Good Evening';
        
        // Capitalize the first letter of userName
        const capitalizedUserName = userName.charAt(0).toUpperCase() + userName.slice(1);
        
        if (this.userNameDisplay) {
            this.userNameDisplay.textContent = `${greeting}, ${capitalizedUserName}!`;
        }
    }

    initializeChatbot() {
        const userName = sessionStorage.getItem('User_Name') || 'User';
        this.setGreeting(userName);
    }

   

    async sendMessage(type) {
        debugger;
        const message = this.messageInput?.value.trim();
        if (!message) return;

        this.addMessageToChat(message, 'snd');
        if (this.messageInput) {
            this.messageInput.value = '';
        }
        this.showLoadingIndicator();

        try {
            const response = await ChatService.getBotResponse(message);
            this.hideLoadingIndicator();
            this.addMessageToChat(response, 'rcv');
            
        } catch (error) {
            console.error('Error getting bot response:', error);
            this.hideLoadingIndicator();
            this.addMessageToChat('Sorry, there was an error processing your message.', 'rcv');
        }
    }


    addMessageToChat(message, type) {
        if (!this.messageContainer) return;
        
        const messageElement = document.createElement('div');
        messageElement.className = `chatbox-msgitem-${type} mb-2`;
        
        // Parse markdown to HTML using marked
        messageElement.innerHTML = marked.parse(message);
        
        // Apply some basic sanitization to prevent XSS attacks
        // Note: For production use, consider using a dedicated sanitizer like DOMPurify
        messageElement.querySelectorAll('script').forEach(script => script.remove());
        
        this.messageContainer.appendChild(messageElement);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
}



// document.addEventListener('DOMContentLoaded', () => {
//     // Login form handling
//     const loginForm = document.querySelector('form');
//     if (loginForm) {
//         const signInButton = loginForm.querySelector('button[type="submit"]');
        
//         loginForm.addEventListener('submit', async (e) => {
//             e.preventDefault();
            
//             // Change button state to loading
//             const originalButtonText = signInButton.innerHTML;
//             signInButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Signing in...';
//             signInButton.disabled = true;
            
//             const username = document.getElementById('User_Name')?.value;
//             sessionStorage.setItem('User_Name', username);
//             const password = document.getElementById('Password')?.value;

//             if (!AuthService.validateCredentials(username, password)) {
//                 // Reset button state
//                 signInButton.innerHTML = originalButtonText;
//                 signInButton.disabled = false;
//                 alert('Please enter valid credentials');
//                 return;
//             }

//             try {
//                 const response = await AuthService.login(username, password);
//                 TokenManager.storeToken(response.token);
//                 window.location.href = '/static/chat.html';
//             } catch (error) {
//                 // Reset button state on error
//                 signInButton.innerHTML = originalButtonText;
//                 signInButton.disabled = false;
//                 alert(error.message);
//             }
//         });
//     }

//     // Chat page initialization code remains unchanged
//     if (window.location.pathname === '/static/chat.html') {
//         const token = TokenManager.getToken();
//         if (!token) {
//             window.location.href = '/Macom-RBI-BOT';
//             return;
//         }

//         new ChatUI();
//     }
// });

document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.querySelector('form');
if (loginForm) {
    const signInButton = loginForm.querySelector('button[type="submit"]');
    
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Change button state to loading
        const originalButtonText = signInButton.innerHTML;
        signInButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Signing in...';
        signInButton.disabled = true;

        // Redirect to desired URL after a short delay to show loading state
        setTimeout(() => {
            // Replace with your desired redirect URL
            window.location.href = '/ldap_authenticate';
        }, 500); // 500ms delay to show loading state
    });
}
    // Function to get query parameter
    function getQueryParam(name) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(name);
    }

    // Check if we're on the redirect page
    if (window.location.pathname.includes('/redirect')) {
        // Get token from URL only on the redirect page
        const token = getQueryParam('token');
        
        // Async function to handle authentication
        async function handleAuthentication() {
            // Validate token exists
            if (!token) {
                console.error('No token found in URL');
                window.location.href = '/Macom-RBI-BOT'; // Redirect if no token
                return;
            }

            try {
                const response = await AuthService.auth_token(token);
                TokenManager.storeToken(response.token);
                window.location.href = '/static/chat.html';
            } catch (error) {
                console.error('Authentication error:', error);
                alert(error.message || 'Authentication failed');
                window.location.href = '/Macom-RBI-BOT'; // Redirect on auth failure
            }
        }

        // Trigger authentication
        handleAuthentication();
    }

    // Chat page initialization code
    if (window.location.pathname === '/static/chat.html') {
        const token = TokenManager.getToken();
        if (!token) {
            window.location.href = '/Macom-RBI-BOT';
            return;
        }

        new ChatUI();
    }
});