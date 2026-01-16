const API_BASE_URL = 'http://localhost:5050';
//const API_BASE_URL = 'https://hr-bot-446976656513.us-central1.run.app';
function generateRandomString(length) {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    const charactersLength = characters.length;

    for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * charactersLength));
    }

    return result;
}

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
        }
        const randomString = generateRandomString(10);
        this.state = {
            isLoggedIn: false,
            access_token: null,
            
            employee_code: randomString,
            firm_id: 'gAAAAABmsxYooKxb_ZeGezv8027MZLA3hTJdppm-_BJyp8v1PhbIjMhv-KA63AvN_ijKwWkhw8iP5pCGqJq1biMaj31fk67AQw==',
            isRecording: false,
            language: 'en-US' // Set default language to English
        };
        this.messages = [];
   
       // this.login();
    }
    generateRandomString(length) {
        const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        let result = '';
        const charactersLength = characters.length;
      
        for (let i = 0; i < length; i++) {
          result += characters.charAt(Math.floor(Math.random() * charactersLength));
        }
      
        return result;
      }

    
    disableButtons() {
        this.args.voiceButton.disabled = true;
        this.args.languageDropdown.disabled = true;
        this.args.refreshButton.disabled = true;
        
        // Add CSS styling to indicate buttons are disabled (optional)
        this.args.voiceButton.classList.add('disabled');
        this.args.languageDropdown.classList.add('disabled');
        this.args.refreshButton.classList.add('disabled');
    
    }
    
    enableButtons() {
        this.args.voiceButton.disabled = false;
        this.args.languageDropdown.disabled = false;
        this.args.refreshButton.disabled = false;
    

    
        // Remove CSS styling for enabled state
        this.args.voiceButton.classList.remove('disabled');
        this.args.languageDropdown.classList.remove('disabled');
        this.args.refreshButton.classList.remove('disabled');
    }
    validateToken() {
        debugger;
        if (!this.state.access_token) {
            this.state.isBlocked = true;
            this.blockChatbox();
            console.error("Token validation failed. Chatbox is blocked.");
 
            return false;
        }
        return true;
    }
    blockChatbox() {
        const { chatBox, sendButton, refreshButton, voiceButton, languageDropdown } = this.args;
    
        // Disable all chatbox buttons
        chatBox.querySelector('input').disabled = true;
        sendButton.disabled = true;
        refreshButton.disabled = true;
        voiceButton.disabled = true;
        languageDropdown.disabled = true;
        this.args.spinner.classList.remove('visible');
        // Optionally, show a message indicating the chatbox is blocked
        let blockMessage = { name: "Mia", message: "Session expired. Please refresh again to continue." };
        this.messages.push(blockMessage);
        this.updateChatText(chatBox);
    }
    
    unblockChatbox() {
        const { chatBox, sendButton, refreshButton, voiceButton, languageDropdown } = this.args;
    
        // Enable all chatbox buttons
        chatBox.querySelector('input').disabled = false;
        sendButton.disabled = false;
        refreshButton.disabled = false;
        voiceButton.disabled = false;
        languageDropdown.disabled = false;
    }
    
    login() {
        debugger;
        const login_url = `${API_BASE_URL}/login`; 
        // const login_url='${API_BASE_URL}/login'
        this.args.loading_spinner.style.display = 'block';
        fetch(login_url, {
            method: 'POST',
            body: JSON.stringify({
                employee_code: encodeURIComponent(this.state.employee_code),
                firm_id: encodeURIComponent(this.state.firm_id)
            }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(response => {
            // Extract the token from headers
            let token = null;
            const authHeader = response.headers.get('Authorization');
            if (authHeader && authHeader.startsWith('Bearer ') ) {
                token = authHeader.substring(7); // Remove 'Bearer ' from the start
              }
       
            return response.json().then(data => ({ data, token })); // Return both data and token
        })
        .then(({ data, token }) => {
            if (data.status === "success") {
                this.state.isLoggedIn = true;
                this.state.access_token=token
                let successMessage = { name: "Mia", message: "Ask me anything about HR policies—quick and easy answers!" };
                this.messages.push(successMessage);
                const chatBox = this.args.chatBox;
                this.updateChatText(chatBox);
                console.log('Login successful');
            } else {
                let failedMessage = { name: "Mia", message: "sorry you can not chat with MACOM AI !" };
                this.messages.push(failedMessage);
                const chatBox = this.args.chatBox;
                this.updateChatText(chatBox);
                console.error('Login failed');
                alert("Login failed: Please check your employee code and firm ID.");
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            let failedMessage = { name: "Mia", message: "sorry you can not chat with mia right now !" };
            this.messages.push(failedMessage);
            const chatBox = this.args.chatBox;
            this.updateChatText(chatBox);
            console.error('Login failed');
        })
        .finally(() => {

            // Hide the loader after the API call completes
            this.args.loading_spinner.style.display = 'none';
        });
    }

    display() {
        const { openButton, chatBox, sendButton, refreshButton, voiceButton, languageDropdown } = this.args;

        //openButton.addEventListener('click', () => this.toggleState(chatBox));
        openButton.addEventListener('click', () => {
            if (!this.state.isLoggedIn) {
                this.login(); // Call login when the chatbox button is clicked
            }
            this.toggleState(chatBox);
        });

        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        refreshButton.addEventListener('click', () => this.onRefreshButton(chatBox));

        voiceButton.addEventListener('click', () => this.toggleVoiceRecognition(chatBox));

        languageDropdown.addEventListener('change', (event) => this.changeLanguage(event));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state.isActive = !this.state.isActive;

        if (this.state.isActive) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatbox) {
        debugger;
        if (!this.state.isLoggedIn) {
            console.error('User is not logged in');
            return;
        }
        var textField = chatbox.querySelector('input');
        let text1 = textField.value;
        if (text1 === "") {
            return;
        }
        textField.disabled = true;
        this.disableButtons();

        // function sanitizeInput(input) {
        //     return DOMPurify.sanitize(input);
        // }
        function sanitizeInput(input) {
            const div = document.createElement('div');
            div.textContent = input;
            return div.innerHTML;
        }

        let sanitizedText = sanitizeInput(text1);

        let msg1 = { name: "User", message: sanitizedText };
        this.messages.push(msg1);
        this.updateChatText(chatbox);
        textField.value = '';
        this.args.sendButton.classList.add('hidden');
        this.args.spinner.classList.add('visible');
        debugger;
        const chat_apiurl_local=`${API_BASE_URL}/chat`
        this.translateText(text1, 'en', translatedText => {
            fetch(chat_apiurl_local, {
                method: 'POST',
                body: JSON.stringify({
                    // access_token: encodeURIComponent(this.state.access_token),
                    input: encodeURIComponent(translatedText)
                }),
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.state.access_token}` // Ensure accessToken is correctly set
                },
            })
            .then(response => {
                // Extract the token from headers
                let token = null;
                const authHeader = response.headers.get('Authorization');
                if (authHeader && authHeader.startsWith('Bearer ') ) {
                    token = authHeader.substring(7); // Remove 'Bearer ' from the start
                  }
                this.state.access_token = token;
                if (!this.validateToken()) {
                    return; // Stop execution if token is invalid
                }
        
                return response.json().then(data => ({ data})); 
                
            })
           
            // .then(r => r.json())
            .then(({ data}) =>{
                const parser = new DOMParser();
                const decodedAnswer = parser.parseFromString(data.answer, 'text/html').body.textContent;
                
                this.translateText(decodedAnswer, this.state.language.split('-')[0], translatedResponse => {
                    let msg2 = { name: "Mia", message: translatedResponse };
                    this.messages.push(msg2);
                    this.updateChatText(chatbox);
         
                });
            }).catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox);
            }).finally(() => {
                this.enableButtons();
                textField.disabled = false;
                this.args.spinner.classList.remove('visible');
                this.args.sendButton.classList.remove('hidden');
            });
        });
    }

    onRefreshButton(chatbox) {
        if (!this.state.isLoggedIn) {
            console.error('User is not logged in');
            return;
        }

        this.args.refreshButton.classList.add('hidden');
        this.args.spinner.classList.add('visible');

        const clear_history_url=`${API_BASE_URL}/clear_history`
        fetch(clear_history_url, {
            method: 'POST',
            body: JSON.stringify({
                access_token: encodeURIComponent(this.state.access_token),
            }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.state.access_token}`
            },
        })
        .then(response => {
            // Extract the token from headers
            let token = null;
            const authHeader = response.headers.get('Authorization');
            if (authHeader && authHeader.startsWith('Bearer ') ) {
                token = authHeader.substring(7); // Remove 'Bearer ' from the start
              }
            this.state.access_token = token
            if (!this.validateToken()) {
                return; // Stop execution if token is invalid
            }
            return response.json().then(data => ({data})); 
        })
        .then(({data}) =>{
            if (data.status === 'success') {
                this.messages = [];
                this.updateChatText(chatbox);
                console.log('Chat history cleared');
            } else {
                console.error('Failed to clear chat history');
            }
        }).catch((error) => {
            console.error('Error:', error);
        }).finally(() => {
            this.args.spinner.classList.remove('visible');
            this.args.refreshButton.classList.remove('hidden');
        });
    }

    toggleVoiceRecognition(chatbox) {
        if (!this.state.isLoggedIn) {
            console.error('User is not logged in');
            return;
        }

        if (!('webkitSpeechRecognition' in window)) {
            console.error('Speech recognition not supported in this browser');
            return;
        }
        // if (!this.state.isRecording) {
        //     this.startVoiceRecognition(chatbox);
        // } else {
        //     this.stopVoiceRecognition(chatbox);
        // }
        // if (this.state.isRecording) {
        //     this.stopVoiceRecognition();
        // } else {
        //     this.startVoiceRecognition(chatbox);
        // }
        if (this.state.isRecording) {
            this.stopVoiceRecognition(chatbox);
            return;
        }
    
        // If not recording, start voice recognition
        this.startVoiceRecognition(chatbox);
    }
//test
    // startVoiceRecognition(chatbox) {
    //     this.recognition = new webkitSpeechRecognition();
    //     this.recognition.lang = this.state.language;
    //     this.recognition.interimResults = false;
    //     this.recognition.maxAlternatives = 1;

    //     this.recognition.timeout = 1000000;  // 10 seconds
    
    //     // Optional: Add a manual timeout to stop recording after a certain duration
    //     this.recordingTimeout = setTimeout(() => {
    //         this.stopVoiceRecognition(chatbox);
    //     }, 30000);  // Stop after 30 seconds automatically
    //     this.recognition.start();
    //     this.state.isRecording = true;
    //     this.args.voiceButton.classList.add('recording');

    //     this.recognition.onresult = (event) => {
            
    //         const speechResult = event.results[0][0].transcript;
    //         console.log('Result received: ' + speechResult); 
    //         let msg1 = { name: "User", message: speechResult };
    //         this.messages.push(msg1);
    //         this.updateChatText(chatbox);

    //         this.args.spinner.classList.add('visible'); 
    //         this.translateText(speechResult, 'en', translatedText => {
    //             console.log('Translated to English: ' + translatedText); 
    //             debugger;
    //             const chaturl=`${API_BASE_URL}/chat`
    //             fetch(chaturl, {
    //                 method: 'POST',
    //                 body: JSON.stringify({
    //                     // access_token: encodeURIComponent(this.state.access_token),
    //                     input: encodeURIComponent(translatedText)
    //                 }),
    //                 mode: 'cors',
    //                 headers: {
    //                     'Content-Type': 'application/json',
    //                     'Authorization': `Bearer ${this.state.access_token}`
    //                 },
    //             })
    //             .then(response => {
    //                 // Extract the token from headers
    //                 let token = null;
    //                 const authHeader = response.headers.get('Authorization');
    //                 if (authHeader && authHeader.startsWith('Bearer ') ) {
    //                     token = authHeader.substring(7); // Remove 'Bearer ' from the start
    //                   }
    //                 this.state.access_token = token
    //                 if (!this.validateToken()) {
    //                     return; // Stop execution if token is invalid
    //                 }
    //                 return response.json().then(data => ({data})); 
    //             })
    //             .then(({data})=> {
    //                 const parser = new DOMParser();
    //                 const decodedAnswer = parser.parseFromString(data.answer, 'text/html').body.textContent;
    //                 this.translateText(decodedAnswer, this.state.language.split('-')[0], translatedResponse => {
    //                     console.log('Translated back to selected language: ' + translatedResponse); 
    //                     let msg2 = { name: "Mia", message: translatedResponse };
    //                     this.messages.push(msg2);
    //                     this.updateChatText(chatbox);
    //                 });
    //             }).catch((error) => {
    //                 console.error('Error:', error);
    //                 this.updateChatText(chatbox);
    //             }).finally(() => {
    //                 this.args.spinner.classList.remove('visible'); 
    //                 this.args.sendButton.classList.remove('hidden');
    //             });
    //         });
    //     };

    //     this.recognition.onerror = (event) => {
    //         console.error('Error occurred in recognition: ' + event.error);
    //     };

    //     this.recognition.onend = () => {
    //         this.stopVoiceRecognition();
    //     };
    // }

    // // stopVoiceRecognition() {
    // //     if (this.recognition) {
    // //         this.recognition.stop();
    // //         this.state.isRecording = false;
    // //         this.args.voiceButton.classList.remove('recording');
    // //     }
    // // }
    // stopVoiceRecognition(chatbox) {
    //     if (this.recognition) {
    //         this.recognition.stop();
    //         this.state.isRecording = false;
    //         this.args.voiceButton.classList.remove('recording');
    //         console.log('Voice recognition stopped');
    //     }
    // }

    startVoiceRecognition(chatbox) {
        this.recognition = new webkitSpeechRecognition();
        this.recognition.lang = this.state.language;
        this.recognition.interimResults = false;
        this.recognition.maxAlternatives = 1;
    
        // Start recording
        this.recognition.start();
        this.state.isRecording = true;
        this.args.voiceButton.classList.add('recording');
    
        // Handle speech recognition results
        this.recognition.onresult = (event) => {
            const speechResult = event.results[0][0].transcript;
            console.log('Result received: ' + speechResult);
            let msg1 = { name: "User", message: speechResult };
            this.messages.push(msg1);
            this.updateChatText(chatbox);
    
            this.args.spinner.classList.add('visible');
            this.translateText(speechResult, 'en', (translatedText) => {
                console.log('Translated to English: ' + translatedText);
                const chaturl = `${API_BASE_URL}/chat`;
                fetch(chaturl, {
                    method: 'POST',
                    body: JSON.stringify({
                        input: encodeURIComponent(translatedText),
                    }),
                    mode: 'cors',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.state.access_token}`,
                    },
                })
                    .then((response) => {
                        let token = null;
                        const authHeader = response.headers.get('Authorization');
                        if (authHeader && authHeader.startsWith('Bearer ')) {
                            token = authHeader.substring(7); // Extract token
                        }
                        this.state.access_token = token;
                        if (!this.validateToken()) return;
                        return response.json().then((data) => ({ data }));
                    })
                    .then(({ data }) => {
                        const parser = new DOMParser();
                        const decodedAnswer = parser.parseFromString(data.answer, 'text/html').body.textContent;
                        this.translateText(decodedAnswer, this.state.language.split('-')[0], (translatedResponse) => {
                            console.log('Translated back to selected language: ' + translatedResponse);
                            let msg2 = { name: "Mia", message: translatedResponse };
                            this.messages.push(msg2);
                            this.updateChatText(chatbox);
                        });
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                        this.updateChatText(chatbox);
                    })
                    .finally(() => {
                        this.args.spinner.classList.remove('visible');
                        this.args.sendButton.classList.remove('hidden');
                    });
            });
        };
    
        this.recognition.onerror = (event) => {
            console.error('Error occurred in recognition: ' + event.error);
        };
    
        this.recognition.onend = () => {
            console.log('Recognition ended');
        };
    }
    
    stopVoiceRecognition(chatbox) {
        if (this.recognition) {
            this.recognition.stop();
            this.state.isRecording = false;
            this.args.voiceButton.classList.remove('recording');
            console.log('Voice recognition stopped');
        }
    }
    
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
            ur: 'ur-IN',
           
        };
        const selectedLanguage = event.target.value;
        this.state.language = languageMap[selectedLanguage] || 'en-US'; // Use appropriate language codes
        console.log('Language changed to:', this.state.language);
    }

    translateText(text, targetLang, callback) {
        //const url = 'https://vapt-mia-app-kqp2s4ffna-uc.a.run.app/translate'; // Ensure this matches your FastAPI server address
        const url=`${API_BASE_URL}/translate`
        const requestBody = {
            text: encodeURIComponent(text),
            target_language:  encodeURIComponent(targetLang),
            // access_token: encodeURIComponent(this.state.access_token)
        };

        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.state.access_token}`
            },
            body: JSON.stringify(requestBody)
        })
        .then(response => {
            // Extract the token from headers
            let token = null;
            const authHeader = response.headers.get('Authorization');
            if (authHeader && authHeader.startsWith('Bearer ') ) {
                token = authHeader.substring(7); // Remove 'Bearer ' from the start
              }
            this.state.access_token = token
            if (!this.validateToken()) {
                return; // Stop execution if token is invalid
            }
            return response.json().then(data => ({data})); 

        })
        .then(({data}) => {
            if (data.translation) {
                debugger;
                
                const parser = new DOMParser();
                const decodedtext = parser.parseFromString(data.translation, 'text/html').body.textContent;
                callback(decodedtext);
                // console.log('Translation:', data.translation);
            } else {
                console.error('Error:', data.detail);
            }
        })
        .catch(error => {
            console.error('Fetch error:', error);
        });
    }
    toggleAudioPlayback() {
        if (this.audio) {
            if (this.audio.paused) {
                this.audio.play();
            } else {
                this.audio.pause();
            }
        }
    }
    textToSpeech(text) {
        debugger;
        //const url = 'https://vapt-mia-app-kqp2s4ffna-uc.a.run.app/text-to-speech'; // Ensure this matches your FastAPI server address
        const url=`${API_BASE_URL}/text-to-speech`
        const requestBody = {
            text: encodeURIComponent(text),
            language:  encodeURIComponent(this.state.language),
            // access_token: encodeURIComponent(this.state.access_token)
        };

        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.state.access_token}`
            },
            body: JSON.stringify(requestBody)
        })
        .then(response => {
            // Extract the token from headers
            let token = null;
            const authHeader = response.headers.get('Authorization');
            if (authHeader && authHeader.startsWith('Bearer ') ) {
                token = authHeader.substring(7); // Remove 'Bearer ' from the start
              }
            this.state.access_token = token
            if (!this.validateToken()) {
                return; // Stop execution if token is invalid
            }
            return response.json().then(data => ({data})); 
        })
        // .then(response => response.json())
        .then(({data}) =>{
            if (data.audioContent) {

                if (this.audio) {
                    this.audio.pause();
                }
                this.audio = new Audio("data:audio/mp3;base64," + data.audioContent);
                this.audio.addEventListener('ended', () => {
                    this.audio = null;
                    this.currentMessage = null;
                });
                this.audio.play();
            } else {
                console.error('Error: ', data.detail);
            }
        })
        .catch(error => {
            console.error('Fetch error: ', error);
        });
    
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach((item) => {
            // Sanitize the message to prevent HTML injection
          
            function sanitizeInput(input) {
                const div = document.createElement('div');
                div.textContent = input;
                return div.innerHTML;
            }
            //item.message
           
            let sanitizedText = sanitizeInput(item.message);
            // const sanitizedMessage = DOMPurify.sanitize(sanitizedText);
    
            
            if (item.name === "Mia") {
                // const formattedMessage = sanitizedMessage.replace(/[*•]/g, '');
                const safeMessage = encodeURIComponent(sanitizedText).replace(/'/g, "\\'");
    
                html += `<div class="messages__item messages__item--visitor">
                            ${marked.parse(sanitizedText)}
                            <link rel="stylesheet" href="../static/style.css">
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
    
        // Attach event listener to buttons
        chatbox.querySelectorAll('.audio-icon').forEach(button => {
            button.addEventListener('click', (event) => {
                const message = decodeURIComponent(event.currentTarget.getAttribute('data-message'));
                if (this.audio) {
                    // If the current audio is playing the same message, pause it
                    if (this.currentMessage === message) {
                        this.toggleAudioPlayback();
                        return;
                    }
                    
                    // If a different audio is playing, stop it first
                    this.audio.pause();
                }
    
                this.currentMessage = message;
                this.textToSpeech(message);
            });
        });
    }
    
    
}

const chatbox = new Chatbox();
chatbox.display();
