/**
 * AudioRecorder - A JavaScript class for recording audio from the microphone
 * 
 * This class provides a simple interface for recording audio from the user's microphone.
 * It handles browser compatibility, microphone access, and provides callbacks for
 * recording events.
 */
class AudioRecorder {
    /**
     * Constructor for AudioRecorder
     * @param {Object} options - Configuration options
     * @param {Function} options.onStart - Callback when recording starts
     * @param {Function} options.onStop - Callback when recording stops, receives the audio blob
     * @param {Function} options.onDataAvailable - Callback when audio data is available
     * @param {Function} options.onError - Callback when an error occurs
     */
    constructor(options = {}) {
        // Set default options
        this.options = {
            mimeType: 'audio/webm',
            audioBitsPerSecond: 128000,
            onStart: () => {},
            onStop: (blob) => {},
            onDataAvailable: (e) => {},
            onError: (error) => { console.error('AudioRecorder error:', error); }
        };
        
        // Override defaults with provided options
        Object.assign(this.options, options);
        
        // Initialize state
        this.mediaRecorder = null;
        this.stream = null;
        this.audioChunks = [];
        this.isRecording = false;
    }
    
    /**
     * Start recording audio
     * @returns {Promise} - Resolves when recording starts
     */
    start() {
        return new Promise((resolve, reject) => {
            if (this.isRecording) {
                return reject(new Error('Already recording'));
            }
            
            // Request microphone access
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    this.stream = stream;
                    
                    // Create MediaRecorder instance
                    const options = {
                        mimeType: this.options.mimeType,
                        audioBitsPerSecond: this.options.audioBitsPerSecond
                    };
                    
                    try {
                        this.mediaRecorder = new MediaRecorder(stream, options);
                    } catch (e) {
                        // If preferred mime type fails, create with default options
                        console.warn('Using default MediaRecorder options:', e);
                        this.mediaRecorder = new MediaRecorder(stream);
                    }
                    
                    // Set up event handlers
                    this.mediaRecorder.ondataavailable = (e) => {
                        if (e.data.size > 0) {
                            this.audioChunks.push(e.data);
                            this.options.onDataAvailable(e);
                        }
                    };
                    
                    this.mediaRecorder.onstop = () => {
                        // Create audio blob from chunks
                        const blob = new Blob(this.audioChunks, { type: 'audio/wav' });
                        
                        // Clean up
                        this.stream.getTracks().forEach(track => track.stop());
                        this.isRecording = false;
                        
                        // Call onStop callback with the audio blob
                        this.options.onStop(blob);
                    };
                    
                    // Start recording
                    this.audioChunks = [];
                    this.mediaRecorder.start();
                    this.isRecording = true;
                    
                    // Call onStart callback
                    this.options.onStart();
                    resolve();
                })
                .catch(error => {
                    this.options.onError(error);
                    reject(error);
                });
        });
    }
    
    /**
     * Stop recording audio
     */
    stop() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
        }
    }
    
    /**
     * Check if currently recording
     * @returns {Boolean} - True if recording, false otherwise
     */
    isRecordingActive() {
        return this.isRecording;
    }
    
    /**
     * Pause recording
     */
    pause() {
        if (this.mediaRecorder && this.isRecording && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.pause();
        }
    }
    
    /**
     * Resume recording after pause
     */
    resume() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'paused') {
            this.mediaRecorder.resume();
        }
    }
    
    /**
     * Cancel recording and discard data
     */
    cancel() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.audioChunks = [];
            this.isRecording = false;
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }
        }
    }
}

// Expose the recorder to the global scope
window.AudioRecorder = AudioRecorder; 