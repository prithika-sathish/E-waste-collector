document.getElementById('startButton').addEventListener('click', function() {
    const video = document.getElementById('webcam');
    video.style.display = 'block';

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            video.srcObject = stream;
        })
        .catch(function(err) {
            console.error("Error accessing webcam: " + err);
        });

    // Initialize object tracking
    const tracker = new tracking.ObjectTracker(['face']);
    tracker.setInitialScale(4);
    tracker.setStepSize(2);
    tracker.setEdgesDensity(0.1);

    tracking.track('#webcam', tracker);

    tracker.on('track', function(event) {
        event.data.forEach(function(rect) {
            console.log('Detected object at: ', rect.x, rect.y, rect.width, rect.height);
            // You can add code here to highlight the detected object
        });
    });
});