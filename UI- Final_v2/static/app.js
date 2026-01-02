// Constants
const CONFIG = (() => {
    try {
        const config = {
            // API_BASE_URL: 'http://localhost:5050',
            API_BASE_URL: 'https://macbot.mactech.net.in/policy',
            OTP_LENGTH: 6,
            TYPING_DELAY: 1000
        };
        
        // Validate API_BASE_URL
        if (!config.API_BASE_URL || typeof config.API_BASE_URL !== 'string') {
            throw new Error('Invalid API_BASE_URL configuration');
        }
        
        return config;
    } catch (error) {
        console.error('Configuration error:', error);
        throw new Error('Failed to initialize application configuration');
    }
})();

// Validate CONFIG
if (!CONFIG.API_BASE_URL) {
    throw new Error('API_BASE_URL is required in configuration');
}

// Token Manager
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

// Authentication Service
class AuthService {
    static state = {
        isLoggedIn: false,
        accessToken: null
    };

    static encryptCredentials(username, password) {
        try {
            if (!username || !password) {
                throw new Error('Username and password are required');
            }
            
            const shift = 5; // Use consistent shift value

            function encryptString(str, shift) {
                try {
                    return btoa(str)
                        .split('')
                        .map(char => {
                            const code = char.charCodeAt(0);
                            return String.fromCharCode(code + shift);
                        })
                        .join('');
                } catch (error) {
                    throw new Error('Failed to encrypt string: ' + error.message);
                }
            }

            return {
                userName: encryptString(username, shift),
                password: encryptString(password, shift)
            };
        } catch (error) {
            console.error('Encryption error:', error);
            throw new Error('Failed to encrypt credentials: ' + error.message);
        }
    }

    static async login(userName, password) {
        try {
            debugger;
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
                // Attempt to parse error message from response if available
                const errorData = await response.json().catch(() => ({}));
                if (response.status === 400 || response.status === 404) {
                    throw new Error('Invalid username or password. Please try again.');
                }
                throw new Error(errorData.detail || 'Login failed. Please check your credentials.');
            }

            const authHeader = response.headers.get('Authorization');
            const token = authHeader?.startsWith('Bearer ') ? authHeader.substring(7) : null;
            const data = await response.json();
            const user_status = data.user_Status
            debugger;
            // Assuming the 'data' object from the login response contains the actual user ID
            // If your login endpoint returns the user ID in the response body, e.g., data.user_id
            // you should store it here. For now, we'll rely on the 'User_Name' from sessionStorage.
            // If your backend sends the user ID in the login response, it's better to use that.
            // Example: if (data.user_id) sessionStorage.setItem('loggedInUserId', data.user_id);

            if (data.status === "success" && token && user_status==="active") {
                this.state.isLoggedIn = true;
                this.state.accessToken = token;
                return { data, token,status: 'active'};
            }else if (data.status === "success" && token && user_status==="Inactive") {
                sessionStorage.setItem('User_Name', userName);
                TokenManager.storeToken(token); // Store the token received from login
                return { success: true, status: 'inactive_otp', token: token, data: data };
                
            } else {
                return { success: false, status: 'failed', message: 'Login failed: Invalid response format or token missing.' };
            }
        } catch (error) {
            console.error('Authentication error:', error);
            // Provide a more user-friendly error message
            throw new Error(error.message || 'Login failed: Please check your credentials and try again.');
        }
    }

    static validateCredentials(username, password) {
        return Boolean(username?.trim() && password?.trim());
    }

    static async verifyOtp(otp) {
        //const url = '${CONFIG.API_BASE_URL}/proxy-verify-otp';
        //const otp = otp
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/proxy-verify-otp`,{
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ otp: otp })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            // Assuming your backend sends a 'status' field in the response for OTP verification
            if (data.status === "active") {
                // OTP verified successfully on the backend
                this.state.isLoggedIn = true; // Update client-side login state
                // No new token is typically received here; the existing one is still valid after activation
                return { success: true, message: 'OTP verified successfully!' };
            } else {
                return { success: false, message: data.detail || 'Invalid OTP. Please try again.' };
            }
        } catch (error) {
            console.error('OTP verification error:', error);
            return { success: false, message: error.message || 'An error occurred during OTP verification.' };
        }
    }
}


// Language Configuration
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

// Chat Service
class ChatService {
    static async getBotResponse(message) {
        try {
            debugger;
            const lang_code = sessionStorage.getItem('lang_code') || 'en-US';
            const response = await fetch(`${CONFIG.API_BASE_URL}/chat2`, {
                method: 'POST',
                body: JSON.stringify({
                    input: encodeURIComponent(message),
                    lang: encodeURIComponent(lang_code)
                }),
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${TokenManager.getToken()}`,
                },
            });

            if (!response.ok) {
                // If response is not OK, try to get more details from the error body
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Failed to fetch response: HTTP status ${response.status}`);
            }

            const authHeader = response.headers.get('Authorization');
            if (authHeader && authHeader.startsWith('Bearer ')) {
                const token = authHeader.substring(7);
                TokenManager.storeToken(token);
            }

            const data = await response.json();
            const parser = new DOMParser();
            const decodedAnswer = parser.parseFromString(data.answer, 'text/html').body.textContent;
            if (data && data.answer) {
                return decodedAnswer;
            } else {
                throw new Error('Answer key not found in the response');
            }
        } catch (error) {
            console.error('Error in getBotResponse:', error);
            // Redirect only if the error is due to authentication or major API failure
            if (error.message.includes('HTTP error! status: 401') || error.message.includes('Failed to fetch')) {
                window.location.href = '/static/index.html'; // Redirect to login
            }
            return { error: 'An error occurred while fetching the response.' };
        }
    }

    /**
     * Fetches chat history for a given user ID with pagination.
     * @param {string} userId - The ID of the user whose chat history is to be fetched.
     * @param {number} [offset=0] - The starting offset for the result set.
     * @param {number} [limit=10] - The maximum number of messages to retrieve per page.
     * @returns {Promise<{messages?: Array<Object>, error?: string}>} - An object containing either the messages or an error.
     */
    static async fetchChatHistory(userId, offset = 0, limit = 10) {
        debugger;
        if (!userId) {
            console.error('Error: User ID is required to fetch chat history.');
            return { error: 'User not logged in or user ID not available.' };
        }

        try {
            // Construct the URL using the dynamically provided userId
            const url = `${CONFIG.API_BASE_URL}/chat_history/${userId}?limit=${limit}&offset=${offset}`;
            
            console.log('Fetching chat history for URL:', url);

            const response = await fetch(url, {
                method: 'POST', // The FastAPI endpoint is a POST request
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${TokenManager.getToken()}` 
                },
                // No body is needed for this GET-like POST request as parameters are in URL
            });

            if (!response.ok) {
                const error = await response.json();
                console.error('Error fetching chat history:', error);
                // Return a more specific error message if available from the backend
                return { error: error.detail || 'Failed to load chat history.' };
            }

            const data = await response.json();
            // The response structure from the backend is { user_id: string, messages: Array<ChatMessage> }
            // We are interested in the 'messages' array.
            return { messages: data.messages };
        } catch (error) {
            console.error('Error in fetchChatHistory:', error);
            return { error: 'An unexpected error occurred while fetching chat history.' };
        }
    }
}

// Chat UI
class ChatUI {
    constructor() {
        this.messageContainer = document.querySelector('.chatbox-messages');
        this.messageInput = document.querySelector('.chatbox .form-control');
        this.userNameDisplay = document.querySelector('.card-label.fw-bolder.fs-3.mb-1');
        this.languageSelect = document.querySelector('select[data-control="select2"]');
        this.microphoneButton = document.querySelector('.btn-chatbox-audio');
        this.chatHistoryContainer = document.getElementById('chat-history-container');
        this.viewMoreButton = document.querySelector('#kt_aside_secondary_footer a');
        this.allMessages = [];
        this.displayedMessageCount = 0;
        this.messagesToLoadMore = 10;
        // Get the user ID from sessionStorage, which is set during login
        this.userId = sessionStorage.getItem('User_Name'); // Use 'User_Name' as the user ID
        // Removed hardcoded this.sessionId
        this.state = {
            isRecording: false,
            language: 'en',
            isLoading: false
        };
        this.setupEventListeners();
        this.initializeChatbot();
        this.loadInitialChatHistory(); // Load chat history on initialization
    }

    setupEventListeners() {
        if (this.messageInput) {
            this.messageInput.addEventListener('keypress', this.handleKeyPress.bind(this));
        }

        if (this.microphoneButton) {
            this.microphoneButton.addEventListener('click', this.handleMicrophoneClick.bind(this));
        }
        if (this.languageSelect) {
            $(this.languageSelect).on('select2:select', (event) => this.changeLanguage(event));
            this.languageSelect.addEventListener('change', (event) => this.changeLanguage(event));
        }
        if (this.viewMoreButton) {
            this.viewMoreButton.addEventListener('click', (event) => {
                event.preventDefault();
                this.loadMoreMessages();
            });
        }
    }

    changeLanguage(event) {
        const languageMap = {
            '1': 'en-US',
            '2': 'ml-IN',
            '3': 'hi-IN',
        };

        const selectedLanguage = event.target.value;
        this.state.language = languageMap[selectedLanguage] || 'en-US';
        sessionStorage.setItem('lang_code', this.state.language);
        console.log('Language changed to:', this.state.language);

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

            this.recognition.start();
            this.state.isRecording = true;
            this.microphoneButton?.classList.add('recording');

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

        if (this.userNameDisplay) {
            this.userNameDisplay.textContent = `${greeting}, ${userName}!`;
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

        this.addMessageToChat(message, 'snd'); // Changed type from 'snd' to 'user' for consistency
        if (this.messageInput) {
            this.messageInput.value = '';
        }
        this.showLoadingIndicator();

        try {
            const response = await ChatService.getBotResponse(message);
            this.hideLoadingIndicator();
            this.addMessageToChat(response, 'rcv'); // Changed type from 'rcv' to 'ai' for consistency
        } catch (error) {
            console.error('Error getting bot response:', error);
            this.hideLoadingIndicator();
            this.addMessageToChat('Sorry, there was an error processing your message.', 'ai');
        }
    }

    addMessageToChat(message, type) {
        if (!this.messageContainer) return;
        
        const messageElement = document.createElement('div');
        messageElement.className = `chatbox-msgitem-${type} mb-2`;
        
        // Convert markdown to HTML
        const htmlMessage = this.convertMarkdownToHTML(message);
        messageElement.innerHTML = htmlMessage;
        
        this.messageContainer.appendChild(messageElement);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
    
    convertMarkdownToHTML(text) {
        let html = text;
        
        // Convert **bold** to <strong>
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convert lines
        const lines = html.split('\n');
        let result = [];
        let inList = false;
        let inTable = false;
        let tableRowCount = 0;
        
        let isTableContext = false;
        
        for (let i = 0; i < lines.length; i++) {
            const trimmed = lines[i].trim();
            
            // End table on empty line
            if (!trimmed) {
                if (inTable) {
                    result.push('</table>');
                    inTable = false;
                    isTableContext = false;
                    tableRowCount = 0;
                }
                continue;
            }
            
            // Check if next line is a table separator to identify table start
            const nextLine = i + 1 < lines.length ? lines[i + 1].trim() : '';
            if (nextLine.match(/^:?-+:?$/) || nextLine === ':------------------------------------------------') {
                isTableContext = true;
                if (!inTable) {
                    if (inList) {
                        result.push('</ul>');
                        inList = false;
                    }
                    result.push('<table>');
                    inTable = true;
                    tableRowCount = 0;
                }
                // Parse header row with pipe separators
                const headerCells = trimmed.split('|').map(cell => cell.trim()).filter(cell => cell);
                result.push('<tr>');
                headerCells.forEach(cell => {
                    result.push(`<th>${cell}</th>`);
                });
                result.push('</tr>');
                tableRowCount++;
                i++; // Skip the separator line
                continue;
            }
            
            // Skip table separator lines
            if (trimmed.match(/^:?-+:?$/) || trimmed === ':------------------------------------------------') {
                continue;
            }
            
            // Handle table rows - improved detection
            const pipeCount = (trimmed.match(/\|/g) || []).length;
            const tabCount = (trimmed.match(/\t/g) || []).length;
            
            if ((pipeCount >= 1 && trimmed.includes('|')) || (tabCount >= 1 && trimmed.includes('\t')) || (isTableContext && inTable)) {
                
                // Skip separator lines even in table context
                if (trimmed.match(/^:?-+:?$/) || trimmed === ':------------------------------------------------') {
                    continue;
                }
                
                if (!inTable) {
                    if (inList) {
                        result.push('</ul>');
                        inList = false;
                    }
                    result.push('<table>');
                    inTable = true;
                }
                
                // Convert table row
                let cells;
                if (trimmed.includes('\t')) {
                    cells = trimmed.split('\t').filter(cell => cell.trim());
                } else if (trimmed.includes('|')) {
                    cells = trimmed.split('|').map(cell => cell.trim()).filter(cell => cell);
                } else if (isTableContext && inTable) {
                    // Single column table row
                    result.push('<tr><td>' + trimmed + '</td></tr>');
                    continue;
                } else {
                    cells = [trimmed];
                }
                
                // Check if it's a header row
                const isHeaderRow = cells.some(cell => 
                    /^(policy|detail|attribute|name|field|information|value|type|description|property)s?$/i.test(cell.trim())
                );
                
                tableRowCount++;
                // Skip the 2nd row
                if (tableRowCount === 2) {
                    continue;
                }
                
                result.push('<tr>');
                cells.forEach(cell => {
                    const tag = isHeaderRow ? 'th' : 'td';
                    result.push(`<${tag}>${cell.trim()}</${tag}>`);
                });
                result.push('</tr>');
            }
            // Handle bullet points
            else if (trimmed.startsWith('* ')) {
                if (inTable) {
                    result.push('</table>');
                    inTable = false;
                }
                if (!inList) {
                    result.push('<ul>');
                    inList = true;
                }
                result.push(`<li>${trimmed.substring(2)}</li>`);
            }
            // Handle regular text
            else {
                if (inList) {
                    result.push('</ul>');
                    inList = false;
                }
                if (inTable) {
                    result.push('</table>');
                    inTable = false;
                    isTableContext = false;
                    tableRowCount = 0;
                }
                result.push(trimmed);
            }
        }
        
        // Close any open tags
        if (inList) {
            result.push('</ul>');
        }
        if (inTable) {
            result.push('</table>');
        }
        
        return result.join('');
    }

    async loadInitialChatHistory() {
        // Use this.userId (from sessionStorage) to fetch history
        const historyData = await ChatService.fetchChatHistory(this.userId, 0);
        if (historyData.error) {
            this.displayErrorMessage(historyData.error);
            return;
        }
        // Clear existing messages before displaying new ones to avoid duplicates on initial load
        if (this.chatHistoryContainer) {
            this.chatHistoryContainer.innerHTML = ''; 
        }
        this.allMessages = historyData.messages || [];
        this.displayMessages(this.allMessages.slice(0, this.messagesToLoadMore));
        this.displayedMessageCount = Math.min(this.allMessages.length, this.messagesToLoadMore);
        this.updateViewMoreButtonVisibility();
    }

    async loadMoreMessages() {
        // Use this.userId (from sessionStorage) to fetch more history
        const historyData = await ChatService.fetchChatHistory(this.userId, this.displayedMessageCount);
        if (historyData.error) {
            this.displayErrorMessage(historyData.error);
            return;
        }
        const newMessages = historyData.messages || [];
        // Prepend new messages to the beginning of the chat history container
        // to maintain chronological order when loading older messages.
        const fragment = document.createDocumentFragment();
        newMessages.reverse().forEach(message => { // Reverse to add oldest first
            const messageDiv = this.createMessageElement(message);
            fragment.appendChild(messageDiv);
        });
        this.chatHistoryContainer.prepend(fragment); // Add new messages to the top

        this.allMessages = newMessages.concat(this.allMessages); // Update allMessages array
        this.displayedMessageCount += newMessages.length;
        this.updateViewMoreButtonVisibility();
    }

    displayErrorMessage(message) {
        if (this.chatHistoryContainer) {
            this.chatHistoryContainer.innerText = ` ${message}`;
        }
        if (this.viewMoreButton) {
            this.viewMoreButton.style.display = 'none';
        }
    }

    createMessageElement(message) { // Helper function to create message div
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(message.role); // 'user' or 'ai'
        const createdAt = message.created_at ? new Date(message.created_at).toLocaleString() : 'N/A';
        
        const roleElement = document.createElement('strong');
        roleElement.textContent = message.role + ':';
        
        const contentElement = document.createTextNode(' ' + message.content + ' ');
        
        const timeElement = document.createElement('small');
        timeElement.textContent = '(' + createdAt + ')';
        
        messageDiv.appendChild(roleElement);
        messageDiv.appendChild(contentElement);
        messageDiv.appendChild(timeElement);
        
        return messageDiv;
    }

    displayMessages(messages) {
        if (!this.chatHistoryContainer) return;

        // Clear existing messages only if it's an initial load or full refresh
        // For 'loadMoreMessages', we prepend, so no clearing here.
        // This function is primarily used by loadInitialChatHistory
        messages.forEach(message => {
            const messageDiv = this.createMessageElement(message);
            this.chatHistoryContainer.appendChild(messageDiv);
        });
        // Scroll to bottom for initial load
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }

    updateViewMoreButtonVisibility() {
        // This logic needs to be adjusted if `allMessages` is not truly all messages
        // but only the currently displayed ones.
        // A better approach for "View More" is to check if the number of messages
        // returned by the API is less than the requested limit, indicating no more history.
        // For now, I'll keep the existing logic but note this for future improvement.
        if (this.viewMoreButton) {

            const hasMore = this.allMessages.length >= this.messagesToLoadMore; // Simplified
            this.viewMoreButton.style.display = hasMore ? 'block' : 'none';
        }
    }
    
    startNewChat() {
        // Clear current chat messages
        if (this.messageContainer) {
            this.messageContainer.innerHTML = '';
        }
        
        // Clear chat history container
        if (this.chatHistoryContainer) {
            this.chatHistoryContainer.innerHTML = '';
        }
        
        // Reset message arrays
        this.allMessages = [];
        this.displayedMessageCount = 0;
        
        // Clear input field
        if (this.messageInput) {
            this.messageInput.value = '';
            this.messageInput.focus();
        }
        
        // Hide view more button
        if (this.viewMoreButton) {
            this.viewMoreButton.style.display = 'none';
        }
        
        // Show welcome message like Gemini
        this.showWelcomeMessage();
    }
    
    showWelcomeMessage() {
        const userName = sessionStorage.getItem('User_Name') || 'User';
        const welcomeMessages = [
            `Hello ${userName}! How can I help you today?`,
            `Hi ${userName}! What would you like to know?`,
            `Welcome back ${userName}! Ask me anything.`,
            `Hello! I'm here to assist you with any questions.`
        ];
        
        const randomMessage = welcomeMessages[Math.floor(Math.random() * welcomeMessages.length)];
        this.addMessageToChat(randomMessage, 'rcv');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Login form handling
const loginForm = document.querySelector('form');
const otpSection = document.getElementById('otpSection');
const otpInput = document.getElementById('otpInput');
const messageDisplay = document.getElementById('messageDisplay');
const errorModalElement = document.getElementById('errorModal');
const errorModalMessage = document.getElementById('errorModalMessage');

// Helper function to show messages
function showMessage(text, type = 'info') {
    if (messageDisplay) {
        messageDisplay.innerHTML = `<div class="alert alert-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} py-2" role="alert">
            <small>${text}</small>
        </div>`;
    }
}

// Helper function to show error modal
function showErrorModal(message) {
    if (errorModalMessage) errorModalMessage.textContent = message;
    if (errorModalElement && typeof bootstrap !== 'undefined' && bootstrap.Modal) {
        const errorModal = new bootstrap.Modal(errorModalElement);
        errorModal.show();
    }
}

// --- Login Form Submission Logic ---
if (loginForm) {
    const signInButton = loginForm.querySelector('button[type="submit"]');

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const originalButtonText = signInButton.innerHTML;
        signInButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Signing in...';
        signInButton.disabled = true;

        const username = document.getElementById('User_Name')?.value;
        sessionStorage.setItem('User_Name', username);
        const password = document.getElementById('Password')?.value;

        if (messageDisplay) messageDisplay.innerHTML = '';

        if (!AuthService.validateCredentials(username, password)) {
            showErrorModal('Please enter valid credentials.');
            signInButton.innerHTML = originalButtonText;
            signInButton.disabled = false;
            return;
        }

        try {
            const result = await AuthService.login(username, password);
            if (result.status === 'active') {
                TokenManager.storeToken(result.token);
                signInButton.innerHTML = originalButtonText;
                signInButton.disabled = false;
                window.location.href = '/static/chat.html';
            } else if (result.status === 'inactive_otp') {
                // Show OTP section inline instead of modal
                otpSection.style.display = 'block';
                if (otpInput) otpInput.value = '';
                showMessage('Please check your email for the OTP code.', 'info');
                signInButton.innerHTML = originalButtonText;
                signInButton.disabled = false;
            } else {
                showErrorModal(result.message || 'Login failed. Please try again.');
                signInButton.innerHTML = originalButtonText;
                signInButton.disabled = false;
            }
        } catch (error) {
            console.error('Login error:', error);
            showErrorModal(error.message || 'An unexpected error occurred during login.');
            signInButton.innerHTML = originalButtonText;
            signInButton.disabled = false;
        }
    });
}

// --- OTP Verification Button Logic ---
const otpVerifyButton = document.getElementById('otpVerifyButton');

if (otpVerifyButton) {
    otpVerifyButton.addEventListener('click', async () => {
        const otp = otpInput?.value.trim();

        if (messageDisplay) messageDisplay.innerHTML = '';

        if (!otp) {
            showMessage('Please enter the OTP.', 'error');
            return;
        }

        if (typeof CONFIG !== 'undefined' && otp.length !== CONFIG.OTP_LENGTH) {
            showMessage(`OTP must be ${CONFIG.OTP_LENGTH} digits long.`, 'error');
            return;
        }

        const originalOtpButtonText = otpVerifyButton.innerHTML;
        otpVerifyButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Verifying...';
        otpVerifyButton.disabled = true;

        try {
            const verificationResult = await AuthService.verifyOtp(otp);

            if (verificationResult.success) {
                showMessage('OTP verified successfully!', 'success');
                setTimeout(() => {
                    window.location.href = '/static/chat.html';
                }, 1500);
            } else {
                showMessage(verificationResult.message || 'Invalid OTP. Please try again.', 'error');
                if (otpInput) otpInput.value = '';
            }
        } catch (error) {
            console.error('Error during OTP verification API call:', error);
            showMessage('An unexpected error occurred during OTP verification.', 'error');
        } finally {
            otpVerifyButton.innerHTML = originalOtpButtonText;
            otpVerifyButton.disabled = false;
        }
    });
}

if (otpInput) {
    otpInput.addEventListener('keypress', async (e) => {
        if (e.key === 'Enter') {
            otpVerifyButton?.click();
        }
    });
}



    // Chat page initialization
    if (window.location.pathname === '/static/chat.html') {
        const token = TokenManager.getToken();
        const userId = sessionStorage.getItem('User_Name'); // Get the user ID from session storage
        if (!token || !userId) { // Check for both token and userId
            window.location.href = '/static/index.html'; // Redirect to login if not authenticated or user ID missing
            return;
        }
        const chatUI = new ChatUI();
        
        // Initialize dark mode only on chat page
        initializeDarkMode();
        
        // Add New Chat button functionality
        const newChatButton = document.getElementById('kt_toolbar_primary_button');
        if (newChatButton) {
            newChatButton.addEventListener('click', () => {
                chatUI.startNewChat();
            });
        }
        
        // Initialize user profile dropdown
        initializeUserMenu();
    }
    
    // Dark mode functionality
    function initializeDarkMode() {
        const darkModeBtn = document.getElementById('darkModeToggleBtn');
        const body = document.body;
        
        // Check for saved dark mode preference
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        
        // Apply saved preference
        if (isDarkMode) {
            body.setAttribute('data-theme', 'dark');
            if (darkModeBtn) {
                darkModeBtn.innerHTML = '<i class="fa fa-sun"></i> Light Mode';
            }
        }
        
        // Add event listener for button
        if (darkModeBtn) {
            darkModeBtn.addEventListener('click', function() {
                const isCurrentlyDark = body.getAttribute('data-theme') === 'dark';
                
                if (isCurrentlyDark) {
                    body.removeAttribute('data-theme');
                    localStorage.setItem('darkMode', 'false');
                    this.innerHTML = '<i class="fa fa-moon"></i> Dark Mode';
                } else {
                    body.setAttribute('data-theme', 'dark');
                    localStorage.setItem('darkMode', 'true');
                    this.innerHTML = '<i class="fa fa-sun"></i> Light Mode';
                }
            });
        }
    }
    
    // User menu functionality
    function initializeUserMenu() {
        const userMenuToggle = document.getElementById('kt_header_user_menu_toggle');
        const userMenu = userMenuToggle?.querySelector('.menu');
        
        if (userMenuToggle && userMenu) {
            userMenuToggle.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                
                // Toggle menu visibility
                if (userMenu.style.display === 'block') {
                    userMenu.style.display = 'none';
                } else {
                    userMenu.style.display = 'block';
                    userMenu.style.position = 'absolute';
                    userMenu.style.top = '100%';
                    userMenu.style.right = '0';
                    userMenu.style.zIndex = '1000';
                }
            });
            
            // Close menu when clicking outside
            document.addEventListener('click', function(e) {
                if (!userMenuToggle.contains(e.target)) {
                    userMenu.style.display = 'none';
                }
            });
        }
    }
});


