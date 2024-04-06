// Event listener for form submission
$('#query-form').submit(function(event) {
    event.preventDefault(); // Prevent default form submission
    
    // Get query and language from form inputs
    var query = $('#query').val();
    var language = $('#language').val();

    // Create JSON object with query and language
    var data = {
        query: query,
        language: language
    };

    // Send POST request to backend
    $.ajax({
        type: 'POST',
        url: '/query',
        contentType: 'application/json',
        data: JSON.stringify(data),
        success: function(data) {
            // Display results on the webpage
            var resultsDiv = $('#results');
            resultsDiv.empty(); // Clear previous results
            data.forEach(function(codeSnippet, index) {
                var codeSnippetElement = $('<p>').text('Code Snippet ' + (index + 1) + ': ' + codeSnippet);
                resultsDiv.append(codeSnippetElement);
            });
        },
        error: function(xhr, status, error) {
            console.error('Error:', error);
        }
    });
});
