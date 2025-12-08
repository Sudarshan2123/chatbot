const CONFIG = (() => {
    try {
        const config = {
            API_BASE_URL: 'http://localhost:5050',
            OTP_LENGTH: 6,
            TYPING_DELAY: 1000
        };
        
        if (!config.API_BASE_URL || typeof config.API_BASE_URL !== 'string') {
            throw new Error('Invalid API_BASE_URL configuration');
        }
        
        return config;
    } catch (error) {
        console.error('Configuration error:', error);
        throw new Error('Failed to initialize application configuration');
    }
})();

if (!CONFIG.API_BASE_URL) {
    throw new Error('API_BASE_URL is required in configuration');
}

class TokenManager {
    static storeToken(token) {
        if (!token) {
            console.error('Token is null or undefined');
            return false;
        }
        
        sessionStorage.setItem('auth_token', token);
        
        // Verify it was stored
        const stored = sessionStorage.getItem('auth_token');
        return !!stored;
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
        accessToken: null,
        csrfToken: null
    };

    static async getCSRFToken() {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/csrf-token`);
            const data = await response.json();
            this.state.csrfToken = data.csrf_token;
            return data.csrf_token;
        } catch (error) {
            console.error('Failed to get CSRF token:', error);
            return null;
        }
    }

    static encryptCredentials(username, password) {
        try {
            if (!username || !password) {
                throw new Error('Username and password are required');
            }
            
            const shift = 5;

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
            const csrfToken = await this.getCSRFToken();
            const encrypted = this.encryptCredentials(userName, password);
            const response = await fetch(`${CONFIG.API_BASE_URL}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({
                    userName: encrypted.userName,
                    password: encrypted.password
                })
            });

            if (!response.ok) {

                const errorData = await response.json().catch(() => ({}));
                if (response.status === 400 || response.status === 404) {
                    throw new Error('Invalid username or password. Please try again.');
                }
                throw new Error(errorData.detail || 'Login failed. Please check your credentials.');
            }

            const authHeader = response.headers.get('Authorization');
            debugger;
            const token = authHeader?.startsWith('Bearer ') ? authHeader.substring(7) : null;
            const data = await response.json();
            const user_status = data.user_Status


            if (data.status === "success" && token && user_status==="active") {
                this.state.isLoggedIn = true;
                this.state.accessToken = token;
                TokenManager.storeToken(token);
                sessionStorage.setItem('User_Name', userName);
                return { data: data, token: token, status: 'active'};
            } else if (data.status === "success" && token && user_status==="Inactive") {
                sessionStorage.setItem('User_Name', userName);
                TokenManager.storeToken(token);
                return { success: true, status: 'inactive_otp', token: token, data: data };
                
            } else {
                return { success: false, status: 'failed', message: 'Login failed: Invalid response format or token missing.' };
            }
        } catch (error) {
            console.error('Authentication error:', error);

            throw new Error(error.message || 'Login failed: Please check your credentials and try again.');
        }
    }

    static validateCredentials(username, password) {
        return Boolean(username?.trim() && password?.trim());
    }

    static async verifyOtp(otp) {
        try {
            const csrfToken = await this.getCSRFToken();
            const response = await fetch(`${CONFIG.API_BASE_URL}/proxy-verify-otp`,{
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({ otp: otp })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.status === "active") {
                this.state.isLoggedIn = true;
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
            debugger;
            const token = TokenManager.getToken();
            const csrfToken = await AuthService.getCSRFToken();
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
                    'Authorization': `Bearer ${token}`,
                    'X-CSRFToken': csrfToken
                },
            });

            if (!response.ok) {
    
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
            if (error.message.includes('HTTP error! status: 401') || error.message.includes('Failed to fetch')) {
               window.location.href = '/static/index.html';
            }
            return { error: 'An error occurred while fetching the response.' };
        }
    }


    static async fetchChatHistory(userId, offset = 0, limit = 10) {
        if (!userId) {
            console.error('Error: User ID is required to fetch chat history.');
            return { error: 'User not logged in or user ID not available.' };
        }

        try {
            debugger;

            const url = `${CONFIG.API_BASE_URL}/chat_history/${userId}?limit=${limit}&offset=${offset}`;
            debugger;
            console.log('Fetching chat history for URL:', url);

            const csrfToken = await AuthService.getCSRFToken();
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${TokenManager.getToken()}`,
                    'X-CSRFToken': csrfToken
                },
            });

            if (!response.ok) {
                const error = await response.json();
                console.error('Error fetching chat history:', error);

                return { error: error.detail || 'Failed to load chat history.' };
            }
                        
            const authHeader = response.headers.get('Authorization');
            if (authHeader && authHeader.startsWith('Bearer ')) {
                const token = authHeader.substring(7);
                TokenManager.storeToken(token);
            }
            const data = await response.json();


            if (!Array.isArray(data.messages) || data.messages.length === 0) {
                return { error: `No history found for this user ${userId}` };
            }
            else{
                return { messages: data.messages };
            }

        } catch (error) {
            console.error('Error in fetchChatHistory:', error);
            return { error: 'An unexpected error occurred while fetching chat history.' };
        }
    }
}


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
        this.userId = sessionStorage.getItem('User_Name');
        this.state = {
            isRecording: false,
            language: 'en',
            isLoading: false
        };
        this.setupEventListeners();
        this.initializeChatbot();
        this.loadInitialChatHistory();
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
        if (!this.messageContainer || this.state.isLoading) return;

        this.state.isLoading = true;
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
            this.state.isLoading = false;
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
                this.handleSpeechResult(speechResult);
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.stopVoiceRecognition();
            };

            this.recognition.onend = () => {
                this.state.isRecording = false;
                this.microphoneButton?.classList.remove('recording');
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
            event.stopPropagation();

            if (this.state.isRecording) {
                this.stopVoiceRecognition();
            }
            this.sendMessage('text');
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
        const message = this.messageInput?.value.trim();
        if (!message) return;


        if (this.state.isRecording) {
            await this.stopVoiceRecognition();
        }

        this.addMessageToChat(message, 'snd');
        if (this.messageInput) {
            this.messageInput.value = '';
        }
        this.showLoadingIndicator();

        try {
            const response = await ChatService.getBotResponse(message);
            this.hideLoadingIndicator();
            if (response && !response.error) {
                this.addMessageToChat(response, 'rcv');
            } else {
                this.addMessageToChat(response.error || 'Sorry, there was an error processing your message.', 'rcv');
            }
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
        messageElement.textContent = message;
        this.messageContainer.appendChild(messageElement);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }

    async loadInitialChatHistory() {

        const historyData = await ChatService.fetchChatHistory(this.userId, 0);
        if (historyData.error) {
            this.displayErrorMessage(historyData.error);
            return;
        }

        if (this.chatHistoryContainer) {
            this.chatHistoryContainer.innerHTML = '';
        }
        this.allMessages = historyData.messages || [];
        this.displayMessages(this.allMessages.slice(0, this.messagesToLoadMore));
        this.displayedMessageCount = Math.min(this.allMessages.length, this.messagesToLoadMore);
        this.updateViewMoreButtonVisibility();
    }

    async loadMoreMessages() {

        const historyData = await ChatService.fetchChatHistory(this.userId, this.displayedMessageCount);
        if (historyData.error) {
            this.displayErrorMessage(historyData.error);
            return;
        }
        const newMessages = historyData.messages || [];

        const fragment = document.createDocumentFragment();
        newMessages.reverse().forEach(message => {
            const messageDiv = this.createMessageElement(message);
            fragment.appendChild(messageDiv);
        });
        this.chatHistoryContainer.prepend(fragment);

        this.allMessages = newMessages.concat(this.allMessages);
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

    createMessageElement(message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(message.role);
        const createdAt = message.created_at ? new Date(message.created_at).toLocaleString() : 'N/A';
        messageDiv.innerHTML = `<strong>${message.role}:</strong> ${message.content} <small>(${createdAt})</small>`;
        return messageDiv;
    }

    displayMessages(messages) {
        if (!this.chatHistoryContainer) return;

        messages.forEach(message => {
            const messageDiv = this.createMessageElement(message);
            this.chatHistoryContainer.appendChild(messageDiv);
        });

        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }

    updateViewMoreButtonVisibility() {

        if (this.viewMoreButton) {
            const hasMore = this.allMessages.length > this.messagesToLoadMore;
            this.viewMoreButton.style.display = hasMore ? 'block' : 'none';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.querySelector('form');
const otpSection = document.getElementById('otpSection');
const otpInput = document.getElementById('otpInput');
const messageDisplay = document.getElementById('messageDisplay');
const errorModalElement = document.getElementById('errorModal');
const errorModalMessage = document.getElementById('errorModalMessage');

    function showMessage(text, type = 'info') {
    if (messageDisplay) {
        messageDisplay.innerHTML = `<div class="alert alert-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} py-2" role="alert">
            <small>${text}</small>
        </div>`;
    }
    }

    function showErrorModal(message) {
    if (errorModalMessage) errorModalMessage.textContent = message;
    if (errorModalElement && typeof bootstrap !== 'undefined' && bootstrap.Modal) {
        const errorModal = new bootstrap.Modal(errorModalElement);
        errorModal.show();
    }
    }

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

    if (window.location.pathname === '/static/chat.html') {
        const token = TokenManager.getToken();
        const userId = sessionStorage.getItem('User_Name');
        if (!token || !userId) {
            window.location.href = '/static/index.html';
            return;
        }
        new ChatUI();
        initializeDarkMode();
        initializeChatPageInteractions();
    }
    
    function initializeDarkMode() {
        const darkModeToggle = document.getElementById('kt_user_menu_dark_mode_toggle');
        const body = document.body;
        
        if (!darkModeToggle) {
            setTimeout(() => {
                initializeDarkMode();
            }, 200);
            return;
        }
        
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        
        if (isDarkMode) {
            body.setAttribute('data-theme', 'dark');
            darkModeToggle.checked = true;
        } else {
            body.removeAttribute('data-theme');
            darkModeToggle.checked = false;
        }
        
        darkModeToggle.removeEventListener('change', handleDarkModeChange);
        darkModeToggle.addEventListener('change', handleDarkModeChange);
        
        function handleDarkModeChange(e) {
            e.stopPropagation();
            if (this.checked) {
                body.setAttribute('data-theme', 'dark');
                localStorage.setItem('darkMode', 'true');
                console.log('Dark mode enabled');
            } else {
                body.removeAttribute('data-theme');
                localStorage.setItem('darkMode', 'false');
                console.log('Dark mode disabled');
            }
        }
        
        console.log('Dark mode initialized, toggle found:', !!darkModeToggle);
    }
    
    function initializeChatPageInteractions() {
        const userMenuToggle = document.getElementById('kt_header_user_menu_toggle');
        const newChatButton = document.getElementById('kt_toolbar_primary_button');
        
        if (userMenuToggle) {
            const userMenuElement = userMenuToggle.querySelector('.menu');
            
            if (userMenuElement) {
                userMenuToggle.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    const isVisible = userMenuElement.style.display === 'block';
                    
                    if (isVisible) {
                        userMenuElement.style.display = 'none';
                    } else {
                        userMenuElement.style.display = 'block';
                        userMenuElement.style.position = 'absolute';
                        userMenuElement.style.top = '50px';
                        userMenuElement.style.right = '10px';
                        userMenuElement.style.zIndex = '9999';
                        userMenuElement.style.backgroundColor = 'white';
                        userMenuElement.style.border = '1px solid #ccc';
                        userMenuElement.style.borderRadius = '8px';
                        userMenuElement.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
                        
                        setTimeout(() => {
                            initializeDarkMode();
                            setupLogout();
                        }, 100);
                    }
                });
                
                document.addEventListener('click', function(e) {
                    if (!userMenuToggle.contains(e.target)) {
                        userMenuElement.style.display = 'none';
                    }
                });
                
                userMenuElement.addEventListener('click', function(e) {
                    e.stopPropagation();
                });
            }
        }
        
        if (newChatButton) {
            newChatButton.addEventListener('click', function(e) {
                e.preventDefault();
                const messageContainer = document.querySelector('.chatbox-messages');
                const messageInput = document.querySelector('#message-input');
                
                if (messageContainer) {
                    messageContainer.innerHTML = '<span class="chatbox-msgitem-rcv mb-2">Hai, What can I do for you?</span>';
                }
                if (messageInput) {
                    messageInput.value = '';
                    messageInput.focus();
                }
            });
        }
        
        const asideToggle = document.getElementById('kt_aside_toggle');
        if (asideToggle) {
            asideToggle.addEventListener('click', function(e) {
                e.preventDefault();
                const aside = document.getElementById('kt_aside');
                if (aside) {
                    aside.classList.toggle('drawer-on');
                }
            });
        }
    }
    
    function setupLogout() {
        const signOutLinks = document.getElementsByClassName('menu-link');
        Array.from(signOutLinks).forEach(link => {
            if (link.textContent.trim() === 'Sign Out') {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    handleLogout();
                });
            }
        });
    }
    
    function handleLogout() {
        TokenManager.removeToken();
        sessionStorage.removeItem('User_Name');
        sessionStorage.removeItem('lang_code');
        localStorage.removeItem('darkMode');
        
        console.log('User logged out successfully');
        window.location.href = '/static/index.html';
    }
});