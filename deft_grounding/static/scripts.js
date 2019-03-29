window.onload = function() {
    groundings = document.getElementsByClassName('click-grounded');
    longforms = document.getElementsByClassName('click-ungrounded');
    for (var i = 0; i < groundings.length; i++) {
	groundings[i].addEventListener('click', function (event) {
	    document.getElementById('name-box').value =
		event.target.getAttribute("data-name");
	    document.getElementById('grounding-box').value =
		event.target.getAttribute('data-grounding');
	});
    }
    for (var i = 0; i < longforms.length; i++) {
	longforms[i].addEventListener('click', function (event) {
	    document.getElementById('name-box').value =
		event.target.getAttribute('data-longform');
	    document.getElementById('grounding-box').value =
		event.target.getAttribute('data-longform');
	});
    } 
}
