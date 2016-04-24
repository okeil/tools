# Simple script to dl and convert youtube playlists - MarkC, 2013

# Prereqs - 
#		youtube-dl --> https://rg3.github.io/youtube-dl/download.html
#		avconv --> https://libav.org/download/
# on osx simply:
#				brew install youtube-dl libav

# Usage: sh GetYoutubePlaylist.sh [youtube playlist addess] [output directory]

# To get the youtube playlist address, view the playlist then click on share, copy that URL
# example: sh GetYouTubePlaylist.sh http://www.youtube.com/playlist?list=PL702CAF4AD2AED35B /home/lp1/Music/youtube
youtube-dl --extract-audio --audio-format mp3  --audio-quality 0 -o "$2%(playlist)s/%(title)s-%(id)s.%(ext)s." $1 --yes-playlist
