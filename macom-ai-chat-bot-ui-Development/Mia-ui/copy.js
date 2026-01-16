// Configuration with environment-based API URL
const CONFIG = {
    API_BASE_URL: process.env.NODE_ENV === 'production' 
        ? 'https://chat-backend-app-446976656513.us-central1.run.app'
        : 'http://localhost:5050',
    OTP_LENGTH: 4,
    TYPING_DELAY: 1000
};

/**
 * Authentication Service
 * Handles user authentication and token management
 */
class AuthService {
    static state = {
        isLoggedIn: false,
        accessToken: null
    };

    /**
     * Encrypts user credentials
     * Note: This is simple obfuscation, not secure encryption
     * @param {string} employee_code - Employee code
     * @param {string} firm_id - Firm ID
     * @returns {Object} Encrypted credentials
     */
    static encryptCredentials(employee_code, firm_id) {
        const shift = 5;
        
        function encryptString(str, shift) {
            return btoa(str)
                .split('')
                .map(char => {
                    const code = char.charCodeAt(0);
                    return String.fromCharCode(code + shift);
                })
                .join('');
        }
        
        return {
            employee_code: encryptString(employee_code, shift),
            firm_id: encryptString(firm_id, shift)
        };
    }

    /**
     * Authenticates user with the server
     * @param {string} employee_code - Employee code
     * @param {string} firm_id - Firm ID
     * @returns {Promise<Object>} Authentication result
     */
    static async login(employee_code, firm_id) {
        try {
            const encrypted = this.encryptCredentials(employee_code, firm_id);
            const response = await fetch(`${CONFIG.API_BASE_URL}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    employee_code: encrypted.employee_code, 
                    firm_id: encrypted.firm_id 
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
            throw error;
        }
    }

    /**
     * Validates that credentials are not empty
     * @param {string} username - Username
     * @param {string} password - Password
     * @returns {boolean} True if credentials are valid
     */
    static validateCredentials(username, password) {
        return Boolean(username?.trim() && password?.trim());
    }
}

/**
 * Chatbox component
 * Handles chat UI and interactions
 */
class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            refreshButton: document.querySelector('.refresh__button'),
            voiceButton: document.querySelector('.voice__button'),
            languageDropdown: document.querySelector('.language__dropdown'),
            spinner: document.querySelector('.loading-spinner'),
            loading_spinner: document.querySelector('.loader')
        };

        this.state = {
            isLoggedIn: false,
            access_token: null,
            employee_code: '100744',
            firm_id: '10001',
            isRecording: false,
            language: 'en-US', // Default language
            isActive: false,
            isBlocked: false
        };
        
        this.messages = [];
        this.audio = null;
        this.currentMessage = null;
        this.recognition = null;
    }

    /**
     * Disable all interactive buttons
     */
    disableButtons() {
        const buttons = [
            this.args.voiceButton,
            this.args.languageDropdown,
            this.args.refreshButton,
            this.args.sendButton
        ];
        
        buttons.forEach(button => {
            if (button) {
                button.disabled = true;
                button.classList.add('disabled');
            }
        });
    }
    
    /**
     * Enable all interactive buttons
     */
    enableButtons() {
        const buttons = [
            this.args.voiceButton,
            this.args.languageDropdown,
            this.args.refreshButton,
            this.args.sendButton
        ];
        
        buttons.forEach(button => {
            if (button) {
                button.disabled = false;
                button.classList.remove('disabled');
            }
        });
    }

    /**
     * Validates if token exists
     * @returns {boolean} True if token is valid
     */
    validateToken() {
        if (!this.state.access_token) {
            this.state.isBlocked = true;
            this.blockChatbox();
            console.error("Token validation failed. Chatbox is blocked.");
            return false;
        }
        return true;
    }

    /**
     * Blocks the chatbox when session expires
     */
    blockChatbox() {
        const { chatBox } = this.args;
    
        // Disable all input elements
        chatBox.querySelector('input').disabled = true;
        this.disableButtons();
        this.args.spinner.classList.remove('visible');
        
        // Show session expired message
        let blockMessage = { name: "Mia", message: "Session expired. Please refresh again to continue." };
        this.messages.push(blockMessage);
        this.updateChatText(chatBox);
    }
    
    /**
     * Unblocks the chatbox
     */
    unblockChatbox() {
        const { chatBox } = this.args;
    
        chatBox.querySelector('input').disabled = false;
        this.enableButtons();
        this.state.isBlocked = false;
    }

    /**
     * Perform login and initialize chat
     * @returns {Promise<void>}
     */
    async login() {
        try {
            this.args.loading_spinner.style.display = 'block';
            const response = await AuthService.login(this.state.employee_code, this.state.firm_id);
            
            const token = response.token;
            const data = response.data;
            
            if (data.status === "success") {
                this.state.isLoggedIn = true;
                this.state.access_token = token;
                let successMessage = { name: "Mia", message: "Hi. My name is Mia. How can I help you?" };
                this.messages.push(successMessage);
                this.updateChatText(this.args.chatBox);
                console.log('Login successful');
            } else {
                this.handleLoginFailure("Sorry, you cannot chat with Mia!");
            }
        } catch (error) {
            this.handleLoginFailure("Sorry, you cannot chat with Mia right now!");
            console.error('Login error:', error.message);
        } finally {
            this.args.loading_spinner.style.display = 'none';
        }
    }

    /**
     * Handle login failure
     * @param {string} message - Message to display
     */
    handleLoginFailure(message) {
        let failedMessage = { name: "Mia", message };
        this.messages.push(failedMessage);
        this.updateChatText(this.args.chatBox);
        console.error('Login failed');
    }

    /**
     * Initialize event listeners
     */
    display() {
        const { openButton, chatBox, sendButton, refreshButton, voiceButton, languageDropdown } = this.args;
    
        // Setup button click handlers
        openButton?.addEventListener('click', async () => {
            this.toggleState(chatBox);
            if (!this.state.isLoggedIn && !this.state.isBlocked) {
                try {
                    await this.login();
                } catch (error) {
                    console.error('Login failed:', error);
                }
            }
        });
    
        sendButton?.addEventListener('click', () => this.onSendButton(chatBox));
        refreshButton?.addEventListener('click', () => this.onRefreshButton(chatBox));
        voiceButton?.addEventListener('click', () => this.toggleVoiceRecognition(chatBox));
        languageDropdown?.addEventListener('change', (event) => this.changeLanguage(event));
    
        // Setup input field
        const inputNode = chatBox.querySelector('input');
        inputNode?.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    /**
     * Toggle chatbox visibility
     * @param {HTMLElement} chatbox - Chatbox element
     */
    toggleState(chatbox) {
        this.state.isActive = !this.state.isActive;
        chatbox.classList.toggle('chatbox--active', this.state.isActive);
    }

    /**
     * Process and send user message
     * @param {HTMLElement} chatbox - Chatbox element
     */
    onSendButton(chatbox) {
        if (!this.state.isLoggedIn || this.state.isBlocked) {
            console.error('User is not logged in or chatbox is blocked');
            return;
        }

        const textField = chatbox.querySelector('input');
        const text = textField.value.trim();
        
        if (!text) {
            return;
        }

        textField.disabled = true;
        this.disableButtons();

        // Sanitize input to prevent XSS
        const sanitizedText = this.sanitizeInput(text);

        let userMessage = { name: "User", message: sanitizedText };
        this.messages.push(userMessage);
        this.updateChatText(chatbox);
        textField.value = '';
        
        this.args.sendButton.classList.add('hidden');
        this.args.spinner.classList.add('visible');
        
        this.sendChatMessage(chatbox, sanitizedText);
    }

    /**
     * Send message to API
     * @param {HTMLElement} chatbox - Chatbox element
     * @param {string} message - User message
     */
    sendChatMessage(chatbox, message) {
        fetch(`${CONFIG.API_BASE_URL}/chat`, {
            method: 'POST',
            body: JSON.stringify({
                input: encodeURIComponent(message),
                lang: encodeURIComponent(this.state.language.split('-')[0])
            }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.state.access_token}`
            },
        })
        .then(response => this.handleApiResponse(response))
        .then(({data}) => {
            if (data) {
                const decodedAnswer = this.decodeHtmlEntities(data.answer);
                let botMessage = { name: "Mia", message: decodedAnswer };
                this.messages.push(botMessage);
                this.updateChatText(chatbox);
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox);
        })
        .finally(() => {
            this.enableButtons();
            chatbox.querySelector('input').disabled = false;
            this.args.spinner.classList.remove('visible');
            this.args.sendButton.classList.remove('hidden');
        });
    }

    /**
     * Handle API response and extract token
     * @param {Response} response - Fetch API response
     * @returns {Promise<Object>} Response data with token
     */
    async handleApiResponse(response) {
        // Extract token from headers
        let token = null;
        const authHeader = response.headers.get('Authorization');
        
        if (authHeader && authHeader.startsWith('Bearer ')) {
            token = authHeader.substring(7);
            this.state.access_token = token;
        }
        
        if (!this.validateToken()) {
            return { data: null };
        }
        
        const data = await response.json();
        return { data };
    }

    /**
     * Handle refresh button click - clear chat history
     * @param {HTMLElement} chatbox - Chatbox element
     */
    onRefreshButton(chatbox) {
        if (!this.state.isLoggedIn || this.state.isBlocked) {
            console.error('User is not logged in or chatbox is blocked');
            return;
        }

        this.args.refreshButton.classList.add('hidden');
        this.args.spinner.classList.add('visible');

        fetch(`${CONFIG.API_BASE_URL}/clear_history`, {
            method: 'POST',
            body: JSON.stringify({}),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.state.access_token}`
            },
        })
        .then(response => this.handleApiResponse(response))
        .then(({data}) => {
            if (data && data.status === 'success') {
                this.messages = [];
                this.updateChatText(chatbox);
                console.log('Chat history cleared');
            } else {
                console.error('Failed to clear chat history');
            }
        })
        .catch((error) => {
            console.error('Error:', error);
        })
        .finally(() => {
            this.args.spinner.classList.remove('visible');
            this.args.refreshButton.classList.remove('hidden');
        });
    }

    /**
     * Toggle voice recognition on/off
     * @param {HTMLElement} chatbox - Chatbox element
     */
    toggleVoiceRecognition(chatbox) {
        if (!this.state.isLoggedIn || this.state.isBlocked) {
            console.error('User is not logged in or chatbox is blocked');
            return;
        }

        if (!('webkitSpeechRecognition' in window)) {
            console.error('Speech recognition not supported in this browser');
            alert('Speech recognition is not supported in this browser');
            return;
        }

        if (this.state.isRecording) {
            this.stopVoiceRecognition();
        } else {
            this.startVoiceRecognition(chatbox);
        }
    }

    /**
     * Start voice recognition
     * @param {HTMLElement} chatbox - Chatbox element
     */
    startVoiceRecognition(chatbox) {
        this.recognition = new webkitSpeechRecognition();
        this.recognition.lang = this.state.language;
        this.recognition.interimResults = false;
        this.recognition.maxAlternatives = 1;

        this.recognition.start();
        this.state.isRecording = true;
        this.args.voiceButton.classList.add('recording');

        this.recognition.onresult = (event) => {
            const speechResult = event.results[0][0].transcript;
            console.log('Speech recognized: ' + speechResult); 
            
            let userMessage = { name: "User", message: speechResult };
            this.messages.push(userMessage);
            this.updateChatText(chatbox);

            this.args.spinner.classList.add('visible'); 
            this.sendChatMessage(chatbox, speechResult);
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error: ' + event.error);
        };

        this.recognition.onend = () => {
            this.stopVoiceRecognition();
        };
    }

    /**
     * Stop voice recognition
     */
    stopVoiceRecognition() {
        if (this.recognition) {
            this.recognition.stop();
            this.state.isRecording = false;
            this.args.voiceButton.classList.remove('recording');
        }
    }

    /**
     * Change current language
     * @param {Event} event - Change event
     */
    changeLanguage(event) {
        const languageMap = {
            en: 'en-US',
            hi: 'hi-IN',
            bn: 'bn-IN',
            te: 'te-IN',
            mr: 'mr-IN',
            ta: 'ta-IN',
            gu: 'gu-IN',
            kn: 'kn-IN',
            ml: 'ml-IN',
            pa: 'pa-IN',
            ur: 'ur-IN'
        };
        
        const selectedLanguage = event.target.value;
        this.state.language = languageMap[selectedLanguage] || 'en-US';
        console.log('Language changed to:', this.state.language);
    }

    /**
     * Toggle audio playback
     */
    toggleAudioPlayback() {
        if (this.audio) {
            if (this.audio.paused) {
                this.audio.play();
            } else {
                this.audio.pause();
            }
        }
    }

    /**
     * Convert text to speech
     * @param {string} text - Text to convert
     */
    textToSpeech(text) {
        fetch(`${CONFIG.API_BASE_URL}/text-to-speech`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.state.access_token}`
            },
            body: JSON.stringify({
                text: encodeURIComponent(text),
                language: encodeURIComponent(this.state.language)
            })
        })
        .then(response => this.handleApiResponse(response))
        .then(({data}) => {
            if (data && data.audioContent) {
                if (this.audio) {
                    this.audio.pause();
                }
                
                this.audio = new Audio("data:audio/mp3;base64," + data.audioContent);
                this.audio.addEventListener('ended', () => {
                    this.audio = null;
                    this.currentMessage = null;
                });
                
                this.audio.play();
            } else if (data && data.detail) {
                console.error('Text-to-speech error: ', data.detail);
            }
        })
        .catch(error => {
            console.error('Text-to-speech fetch error: ', error);
        });
    }

    /**
     * Update chat messages in UI
     * @param {HTMLElement} chatbox - Chatbox element
     */
    updateChatText(chatbox) {
        let html = '';
        
        // Process messages in reverse order (newest at bottom)
        this.messages.slice().reverse().forEach((item) => {
            const sanitizedText = this.sanitizeInput(item.message);
            const safeMessage = encodeURIComponent(sanitizedText).replace(/'/g, "\\'");
            
            if (item.name === "Mia") {
                html += `<div class="messages__item messages__item--visitor">
                            ${marked.parse(sanitizedText)}
                            <button class="audio-icon" style="border: none !important; cursor: pointer;" data-message="${safeMessage}">
                                <img title="Listen" class="audio-btn-img" src="../static/images/audio.svg" alt="Audio Icon" />
                            </button>
                         </div>`;
            } else {
                html += `<div class="messages__item messages__item--operator">
                            ${marked.parse(sanitizedText)}
                         </div>`;
            }
        });
    
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    
        // Attach audio playback event listeners
        this.attachAudioEventListeners(chatbox);
    }
    
    /**
     * Attach event listeners to audio buttons
     * @param {HTMLElement} chatbox - Chatbox element
     */
    attachAudioEventListeners(chatbox) {
        chatbox.querySelectorAll('.audio-icon').forEach(button => {
            button.addEventListener('click', (event) => {
                const message = decodeURIComponent(event.currentTarget.getAttribute('data-message'));
                
                // Handle audio toggle logic
                if (this.audio) {
                    if (this.currentMessage === message) {
                        this.toggleAudioPlayback();
                        return;
                    }
                    
                    this.audio.pause();
                }
                
                this.currentMessage = message;
                this.textToSpeech(message);
            });
        });
    }
    
    /**
     * Sanitize input to prevent XSS
     * @param {string} input - User input
     * @returns {string} Sanitized input
     */
    sanitizeInput(input) {
        const div = document.createElement('div');
        div.textContent = input;
        return div.innerHTML;
    }
    
    /**
     * Decode HTML entities in text
     * @param {string} html - HTML string
     * @returns {string} Decoded text
     */
    decodeHtmlEntities(html) {
        const parser = new DOMParser();
        return parser.parseFromString(html, 'text/html').body.textContent;
    }
}

// Initialize chat application
document.addEventListener('DOMContentLoaded', () => {
    const chatbox = new Chatbox();
    chatbox.display();
});