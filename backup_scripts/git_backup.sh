#!/bin/bash
#USAGE: sh getAllAssetOwl.sh
cd ~/git
curl -u username:apikey -s https://api.github.com/orgs/<org>/repos?per_page=200 | ruby -rubygems -e 'require "json"; JSON.load(STDIN.read).each { |repo| %x[git clone #{repo["ssh_url"]} ]}'
for x in $(ls -ld */ | awk '{print $9}'); do
	cd $x
	pwd
	git pull -p
	cd .. 
done
