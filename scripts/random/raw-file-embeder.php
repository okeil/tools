<?php
/**
 * Plugin Name: Raw File Embeder
 * Plugin URI: https://mwclearning.com/raw-file-embeder
 * Description: This plugin embeds text from a url containing raw text(ie github raw files). Just use [raw_embed url="https://raw.githubusercontent.com/git/git/master/zlib.ci"]
 * Version: 1.0.0
 * Author: Mark Culhane
 * Author URI: https://mwclearning.com
 * License: GPL3
 */
// [raw_embed url=""]
function raw_file_embeder($atts) {
	extract(shortcode_atts(array(
		'url' => 'no_url_found',
	), $atts));
	if ( $url == "no_url_found" ) {
		$raw_data =  "url not found!";
	} else {
		$ch = curl_init(trim($url,'"'));
		 curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
		 curl_setopt($ch, CURLOPT_HEADER, 0); 
		$raw_data = curl_exec($ch);
		if($errno = curl_errno($ch)) {
			$raw_data = curl_strerror($errno);
		}
		 curl_close($ch);
	}
	return $raw_data;
}
add_shortcode('raw_embed', 'raw_file_embeder');
?>
