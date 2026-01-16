// Function to extract query parameters from the URL
function getQueryParam(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

// Function to make API call with the token
async function callApiWithToken() {
    try {
        // Extract the token from the URL
        const token = getQueryParam('token');
        
        if (!token) {
            throw new Error('No token found in URL');
        }

        // API endpoint - replace with your actual API URL
        const apiUrl = 'https://your-api-endpoint.com/verify';

        // Make the API call
        const response = await fetch(apiUrl, {
            method: 'POST', // or 'GET' depending on your API
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            // Uncomment and modify body if needed
            // body: JSON.stringify({
            //     token: token
            // })
        });

        // Check if the response is successful
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Parse the response
        const data = await response.json();

        // Handle successful response
        console.log('API Response:', data);

        // Redirect to the appropriate subdomain or page
        const subdomain = data.subdomain || 'example.com';
        window.location.href = `https://${subdomain}`;

    } catch (error) {
        // Handle any errors
        console.error('Error:', error);
        
        // Optional: display error message on the page
        const messageElement = document.querySelector('.message');
        if (messageElement) {
            messageElement.textContent = 'An error occurred during redirection.';
        }

        // Fallback redirect or error handling
        // window.location.href = '/error-page';
    }
}

// Call the function when the page loads
document.addEventListener('DOMContentLoaded', callApiWithToken);