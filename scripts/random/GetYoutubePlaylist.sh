# Simple script to dl and convert youtube playlists - MarkC, 2013
# Usage: sh GetYoutubePlaylist.sh [youtube playlist addess] [output directory]
# To get the youtube playlist address, view the playlist then click on share, copy that URL
# example: sh GetYouTubePlaylist.sh http://www.youtube.com/playlist?list=PL702CAF4AD2AED35B /home/lp1/Music/youtube
# ytdl - abritrary extension name
outputFileName="/%(playlist)s/%(title)s"
youtube-dl -o "$2%(playlist)s/%(title)s.ytdl" --max-quality url $1
find $2 -type f -name "*.ytdl" -exec avconv -i '{}' '{}'.mp3 ;
find $2 -name *.ytdl -exec rm {} +
