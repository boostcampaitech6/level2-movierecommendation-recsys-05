PutTextIn() {
  ### 파일에 텍스트가 존재한지 확인 후 없으면 추가
  if ! $(grep -q $"$1" $"$2" 2> /dev/null); then
    echo $"$1" >> $"$2"
    echo >> $"$2"
  fi 
}

### apt 업데이트
apt update

### python 컴파일에 필요한 패키지들을 설치
PACKAGES='curl libbz2-dev libssl-dev libsqlite3-dev liblzma-dev libffi-dev libncursesw5-dev libreadline-dev build-essential libgdbm-dev libnss3-dev zlib1g-dev tk-dev'
apt install -y $PACKAGES

### pyenv 설치
export PYENV_ROOT="/data/ephemeral/.pyenv"
curl https://pyenv.run | bash

### PATH에 pyenv 등록
# text='export PYENV_ROOT="/data/ephemeral/.pyenv"
# [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
# eval "$(pyenv init -)"'
text=$(cat <<'EOF'
export PYENV_ROOT="/data/ephemeral/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
)

PutTextIn $"$text" ~/.bashrc


source ~/.bashrc

### python 버전 선택
default="3.9.18"
read -p "값을 입력하세요 (기본값: $default): " -e input
PYTHON_VERSION=${input:-$default}

pyenv install $PYTHON_VERSION
pyenv global $PYTHON_VERSION

### python 버전 확인
which python


### git branch 표시
text=$(cat <<'EOF'
parse_git_branch() {
     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}
export PS1="\u@\h \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ "
EOF
)

PutTextIn $"$text" ~/.bashrc
