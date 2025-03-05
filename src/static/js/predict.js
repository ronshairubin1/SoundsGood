// Add this function to handle different event types from the server
function handleEventFromServer(data) {
    console.log('📋 DEBUG: Received event from server:', data);
    
    if (data.event === "prediction" && data.prediction) {
        // Update UI with prediction
        updatePredictionUI(data.prediction);
    }
    else if (data.event === "speech_detected") {
        // Show speech detected but not loud enough indicator
        document.getElementById('speechDetectedIndicator').style.display = 'block';
        
        // Hide after 3 seconds
        setTimeout(() => {
            document.getElementById('speechDetectedIndicator').style.display = 'none';
        }, 3000);
        
        console.log(`📋 DEBUG: Speech detected but not loud enough. Amplitude: ${data.amplitude}, Threshold: ${data.threshold}`);
    }
    else if (data.log) {
        console.log('📋 DEBUG: Log message from server:', data.log);
    }
}

// Update the EventSource message handler to use the new function
eventSource.onmessage = function(event) {
    console.log('📋 DEBUG: Received event from stream:', event.data);
    
    if (event.data === ": heartbeat") {
        console.log('📋 DEBUG: Received heartbeat');
        return;
    }
    
    try {
        const data = JSON.parse(event.data);
        handleEventFromServer(data);
    } catch (e) {
        console.error('📋 DEBUG: Error parsing event data:', e, event.data);
    }
}; 