# Environment Setting Guide

서버 할당 후 패키지 관리자 update 및 locale 설정하기.

```bash
$ apt update
$ apt-get update
$ pip install --upgrade pip
$ apt install locales
$ locale-gen en_US.UTF-8
$ update-locale LANG=en_US.UTF-8
```

할당받은 서버에서 pyenv.sh을 실행하면 pyenv가 설치됩니다.

```bash
$ bash pyenv.sh
$ source ~/.bashrc
```

