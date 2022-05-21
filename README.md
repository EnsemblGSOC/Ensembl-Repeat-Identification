<!--
 * @Author: yangtcai yangtcai@gmail.com
 * @Date: 2022-05-12 22:59:30
 * @LastEditors: yangtcai yangtcai@gmail.com
 * @LastEditTime: 2022-05-16 19:06:43
 * @FilePath: /DL-in-repeat-seq/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

# DL-in-repeat-seq

## Usage

Set up a development environment with [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://github.com/python-poetry/poetry):
```shell
pyenv install 3.9.12

pyenv virtualenv 3.9.12 repeat_identification

poetry install
```

download annotations with the following command:
```shell
python anno.py --species hg38
```
