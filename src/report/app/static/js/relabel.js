// Function to allow user to label image + tiles as transient objects 
// while performing error analysis
function label_image(image, element) {

    // Extract details from the image file
    file_parts = image.split('.');

    // Image filename without the tile coordinates
    filename = file_parts[0].substr(0, file_parts[0].length - 3) + '.' + file_parts[1] + '.' + file_parts[2]

    // Tile coordinates specified in the last two characters of the image filename
    tile = file_parts[0].slice(-2);

    // Display the labelled image + tile row as a CSV line
    // This allows the user to copy and paste the labelling results
    document.getElementById('labels').innerHTML += filename + ',,,,,astrosmall00_mobile,,' + tile + '<br />';
    
    // Highlight the row as having been labelled
    element.parentNode.parentNode.className = 'label';

}