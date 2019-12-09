# http://minerl.io/docs/tutorials/index.html

yes | sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
yes | sudo apt-get install openjdk-8-jdk

workon py3
pip install --upgrade minerl

# Probably should add this to zshrc file
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 
