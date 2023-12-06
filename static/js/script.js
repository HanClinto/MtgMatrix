// Function to toggle the search menu
function toggleSearchMenu() {
    var searchMenu = document.getElementById("searchMenu");
    var body = document.body;
    if (searchMenu.style.width === '250px') {
        searchMenu.style.width = '0';
        body.classList.remove('menu-active');
    } else {
        searchMenu.style.width = '250px';
        body.classList.add('menu-active');
    }
}

document.addEventListener('DOMContentLoaded', (event) => {
    var searchMenu = document.getElementById("searchMenu");
    // If the body has a class 'index', then we're on the homepage
    if (document.body.classList.contains('index')) {
        searchMenu.style.width = '250px';
        document.body.classList.add('menu-active');
    }
    // Else, we assume it's any other page and the menu should be closed
    else {
        searchMenu.style.width = '0';
    }
});