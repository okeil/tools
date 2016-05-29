/***
* simple highlighter 
***/
$(document).ready(function(){

$( ".row" ).hover(highlight, lowlight);

function highlight() {
	console.log("highlighting");
	$(this).addClass("highlighted");
}

function lowlight() {
	$(this).removeClass("highlighted");
}});
