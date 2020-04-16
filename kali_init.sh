apt update
apt upgrade -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
echo 'deb [arch=amd64] https://download.docker.com/linux/debian buster stable' | sudo tee etc/apt/sources.list.d/docker.list
apt-get update
apt-get remove docker docker-engine docker.io
apt-get install docker-ce -y
systemctl start docker
systemctl enable docker
docker --version
git clone https://github.com/cloudflare/flan.git
apt auto-remove
apt update
apt upgrade -y
