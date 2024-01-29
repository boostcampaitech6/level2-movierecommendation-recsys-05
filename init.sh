apt update

### python 컴파일에 필요한 패키지들을 설치
PACKAGES='curl libbz2-dev libssl-dev libsqlite3-dev liblzma-dev libffi-dev libncursesw5-dev libreadline-dev build-essential libgdbm-dev libnss3-dev zlib1g-dev tk-dev'
apt install -y $PACKAGES
curl https://pyenv.run | bash

### PATH에 pyenv 등록
echo 'export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"' >> ~/.bashrc

source ~/.bashrc

### python 3.10.13 설치
pyenv install 3.10.13
pyenv global 3.10.13

### python 버전 확인
which python


### git branch 표시
echo '
parse_git_branch() {
     git branch 2> /dev/null | sed -e '\''/^[^*]/d'\'' -e '\''s/* \(.*\)/(\1)/'\''
}
export PS1="\u@\h \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ "
' >> ~/.bashrc

source ~/.bashrc
