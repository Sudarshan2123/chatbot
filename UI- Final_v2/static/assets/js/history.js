let allMessages = []; // Store all fetched messages
    let displayedMessageCount = 0;
    const messagesToLoadMore = 5;     // Number of messages to load on "View More"
    let sessionId = '77185df2-53f3-42e4-8964-bc481393cf37'; // Replace with your actual session ID logic

    async function fetchChatHistory(sessionId, offset = 0) { // Add offset parameter
        console.log('Fetching chat history for URL:', `/chat_history/${sessionId}?limit=${messagesToLoadMore}&offset=${offset}`);
        const response = await fetch(`/chat_history/${sessionId}?limit=${messagesToLoadMore}&offset=${offset}`, {  // Include limit and offset
            method: 'POST', // Keep as POST as per the backend definition
            headers: {
                'Content-Type': 'application/json'
                // Add any necessary authentication headers here
            },
            // If your API expects a request body for POST, add it here
            // body: JSON.stringify({})
        });

        if (!response.ok) {
            const error = await response.json();
            console.error('Error fetching chat history:', error);
            document.getElementById('chat-history-container').innerText = `Error: ${error.detail || 'Failed to load chat history.'}`;
            const viewMoreButton = document.querySelector('#kt_aside_secondary_footer a');
            if (viewMoreButton) {
                viewMoreButton.style.display = 'none'; // Hide "View More" on error
            }
            return;
        }

        const data = await response.json();
        const newMessages = data.messages; // Get messages from the response
        allMessages = allMessages.concat(newMessages); // Append new messages
        displayMessages(newMessages); // Display only the *new* messages
    }

    function displayMessages(messages) { // Refactor to take messages to display
        const chatHistoryContainer = document.getElementById('chat-history-container');
        const viewMoreButton = document.querySelector('#kt_aside_secondary_footer a');

        if (messages.length > 0) {
            messages.forEach(message => {
                const messageDiv = document.createElement('div');
                messageDiv.classList.add('message');
                messageDiv.classList.add(message.role);
                const createdAt = message.created_at ? new Date(message.created_at).toLocaleString() : 'N/A';
                messageDiv.innerHTML = `<strong>${message.role}:</strong> ${message.content} <small>(${createdAt})</small>`;
                chatHistoryContainer.appendChild(messageDiv);
            });
            displayedMessageCount += messages.length;
        }

        // Hide "View More" if all messages are displayed
        if (displayedMessageCount >= allMessages.length) {
            if (viewMoreButton) {
                viewMoreButton.style.display = 'none';
            }
        } else {
            if (viewMoreButton) {
                viewMoreButton.style.display = 'block';
            }
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        fetchChatHistory(sessionId, 0); // Initial fetch with offset 0

        const viewMoreButton = document.querySelector('#kt_aside_secondary_footer a');
        if (viewMoreButton) {
            viewMoreButton.addEventListener('click', (event) => {
                event.preventDefault();
                fetchChatHistory(sessionId, displayedMessageCount); // Fetch with updated offset
            });
        }
    });